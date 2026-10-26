# TP2Prep: TP2/EP2 bring-up preparation

Checkpoint: `/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.565-tp2` (read-only)
Overlay: `/models/dsv41/exl3-serve-tp2` (symlinks + index with engram embed keys)
Smoke: `qmesh/tp2_smoke.py` (compile-checked only; NOT run on GPU)

## Status
- Overlay built and CPU-validated: `Config.from_directory` loads as `DeepseekV41Config`, `engram_layer_ids=[1, 14]`. Index has 192,482 keys, 43 distinct files.
- No GPU run. GPU0/GPU1 belong to peers; run `tp2_smoke.py` only after release.

## Weight bytes (from safetensors headers, `[VERIFIED]`)
| Category | GiB |
|---|---|
| Experts, layers 0–39 (trellis+suh+svh+mul1) | 162.32 |
| Experts, MTP (`mtp.0..2`, 384 each) | 6.33 |
| Layer dense (attn/norm/HC/etc.) | 4.875 |
| Embed | 1.233 |
| Head | 0.617 |
| Engram projections (`k_weight`, `q_weight`, `wkv`, layers 1, 14) | 0.586 |
| MTP non-expert | 0.402 |
| Vision | 0.767 |
| Aligner | 0.137 |
| **Total** | **177.92** |

Index `total_size` matches (191,040,601,760 B). Expert totals reconcile: 162.32 + 6.33 + 0.65 sidecars = 169.30 GiB.

K per expert (trellis shape `[in/16, out/16, 16·K]`; w1/w2/w3 of one expert share K): 2.0 ×5,500; 2.5 ×4,860; 3.0 ×2,500; 3.5 ×2,500 (total 15,360 experts across 40 layers + 3 MTP layers = 47,232 trellis tensors). Per-expert bytes 8.89–15.53 MB.

## Per-GPU memory, EP2 (`[INFERENCE]` for dense split)
- Experts (contiguous split, layers 0–39): ~81.5 GiB/GPU. Measured split sums: 81.58 / 81.38.
- Dense: layer dense 4.875 assumed split in half (2.44/GPU). Embed, head, MTP non-expert, vision, aligner, engram projections assumed replicated (3.74/GPU, worst case).
- Weights per GPU (EP2 contiguous, excl. MTP experts): **~87.7 GiB**. Including MTP experts if target loads them: ~90.8 GiB.
- Excludes KV cache, activations, workspace. Reference: a whole-model single GPU would be 177.92 GiB.

## Expert balance (selection-weighted, ranking CSV `expert_ranking_2.25.csv`, 40 layers)
Score = max/mean of (selection count × bytes) per layer across the 2 devices:
| Split | mean | worst layer |
|---|---|---|
| Contiguous [0,192)/[192,384) | 1.065 | 1.158 |
| LPT by sel×bytes, equal count | 1.0001 | 1.0006 |
| LPT by sel×bytes, free | 1.0000 | 1.0002 |

Count-only (selection share) contiguous: mean 1.0529, worst 1.1266. Contiguous half 0 carries 49.6% of selections on average.
Expert byte table: `/tmp/tp2_expert_bytes.json` (key `"layer,expert"`). Analysis script: `/tmp/tp2_bal.py`.

**Recommendation:** contiguous EP2 is the stock path and runs without changes. A custom expert→device map (LPT) gives about 15% better worst-case layer balance, but it needs changes to `tp_import` (`routing_first/last`), `make_tp_allocation`, and MoE kernel indexing (`block_sparse_mlp.py`, MixedK's file). Coordinate with MixedK before touching it.

## Blockers and risks
1. **Mixed K per layer breaks `MultiLinear`.** `modules/multilinear.py:26,76` asserts uniform K across all local experts; line 93 needs trellis `16·K`. A layer contains experts of K 2.0, 2.5, 3.0, and 3.5, so `BlockSparseMLP.load_local` (`block_sparse_mlp.py` ~535–575) will fail on the fused/quant path. Needs grouping by K (per-K MultiLinear, or a per-K dispatch). WIP diff in `block_sparse_mlp.py` does not address it. Owner: MixedK.
2. **Half-integer K needs mul1** (`modules/quant/exl3.py:81`). K 2.5 and 3.5 must have `mul1` present. Checked: `mul1` count 47,232 (matches trellis count), so the sidecar exists for every expert. [VERIFIED by header count; sizes are 0.0 GiB total, so mul1 is a flag tensor.]
3. **DSpark drafter not TP-capable.** `DeepseekV41MTPModel` inherits `caps["supports_tp"] = False` (`architecture/deepseek_v4_mtp.py`). Decision: **DSpark is out of scope for TP2 bring-up.** Loading it single-GPU adds ~6.3 GiB of MTP experts plus non-expert weights on one device, which is not in the budget. Run `tp2_smoke.py` with the draft disabled. If DSpark is wanted later, load it on GPU0 as a separate component and measure memory first.

## Collectives per decode token (code-derived, `[INFERENCE]` count; not measured)
Correction to the earlier estimate of 40/token. DSV4.1 uses `TransformerBlock` (`architecture/deepseek_v41.py:259`), not `ParallelDecoderBlock`. Its `tp_import` (`modules/transformer.py` 285–313) calls attn and mlp with default `skip_reduction`, so each sublayer sets `tp_reduce=True`:
- Attention: `DSV4Attention.forward` (`modules/dsv4.py` ~1154) does `tp_collect` per call.
- MoE: `BlockSparseMLP` (`modules/block_sparse_mlp.py` ~1494) does `tp_collect` once per call.
- HC mix/apply: no collectives.

Expected: **2 all-reduces per layer, 80 per token** for 40 layers. Router is `sqrtsp`, which is replicated (`block_sparse_mlp.py:83`), so no broadcast for routing. The `ParallelDecoderBlock` single-collect path does not apply to DSV4.1.

## Backends
- **Native** (`tp_backend="native"`, used by the smoke): POSIX `shared_memory` regions G/B/R/S/LL, registered pinned and mapped. All-reduce via `ext.pg_all_reduce_cpu` (bf16 wire for fp32).
- **NCCL**: `dist.init_process_group("nccl")`, fp32 → bf16 temp unless `EXL3_TP_NCCL_FP32=1`; broadcast/gather fall back to native.

## Engram under TP
`EngramLayer.make_tp_allocation` returns `[]` (replicated). Tables stream from disk per rank (`NGramEmbedding.tp_export`). The `EngramLayerState` ring is per-rank identical. Engram embed tables (`layers.{1,14}.engram.embed.{weight,scale}`) are F8_E4M3 rows with E8M0 scales, mapped in the overlay index to `engram-only-L01/L14.safetensors` (symlinks to `/models/dsv41/exl3-serve/`).

## Smoke script
`qmesh/tp2_smoke.py`: `Model.load(tensor_p=True, tp_output_device=0, tp_backend="native", tp_options={"moe_tensor_split": False}, max_batch_size=1, ...)`. Prints load time, per-GPU `max_memory_allocated`, first-token ms, decode ms/token. Compile-checked only (`py_compile` OK). DSpark disabled.

## Checks to run after GPU release
1. `tp2_smoke.py` on GPU0+GPU1 with `CUDA_VISIBLE_DEVICES=0,1`. Record load time, per-GPU max alloc, first-token and decode ms.
2. Re-check MixedK's per-K grouping lands before running (uniform-K asserts will fail otherwise).
3. Overlay header check: each key in the index resolves to a tensor in the named file.
4. Confirm `mul1` for all 2.5/3.5 experts at load time.
