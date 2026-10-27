# TP=2 design contract (Recipe B only — 2× RTX PRO 6000 Max-Q, P7 GPU0 + GPU1)

Owner-approved 2026-10-26. This recipe ONLY. Never touch recipe A (`/home/treadall/recipe-gpu0-gx1`, branch
`recipe/gpu0-gx1-hybrid`, checkpoint `...-mixed-2.8535`, GX1/GX10, cold pack), GX1, GX2, P7 GPU2 (A4000), or
`/home/treadall/afd-tiered-packs-1017`.

## Source
- Checkpoint (read-only): `/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.565-tp2` (177.92 GiB, 41 shards)
- External Engram tables (read-only, host DDR5 ECC RAM, ONE shared copy for both ranks):
  `/models/dsv41/exl3-serve/engram-only-L01.safetensors`, `engram-only-L14.safetensors` (FP8 E4M3 + E8M0 scale)
- Expert bitrates and layer positions stay EXACTLY as quantized (hot 3.5 → cold 2.0). Nothing is requantized,
  moved or re-ranked. Every expert is cut down the middle.

## Output (pre-sharded offline)
`/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.565-tp2-shard2/{rank0,rank1}/`
(safetensors + `model.safetensors.index.json` + `config.json` with `tp_rank`, `tp_size=2` + `shard_manifest.json`).
Disk: ~179 GiB total; /models/dsv41 has ~200 GiB free → do not write anything else large there.

## Shard rules (EXL3: every cut multiple of 128 → trellis slices exact; tile = 16×16)
EXL3 linear tensors: `trellis` I16 [in/16, out/16, 16·K], `suh` F16 [in], `svh` F16 [out], `mul1`/`mcg` scalar,
optional `bias` [out]. Weight is W = diag(su)·H·T·H·diag(sv) per 128-block.
- OUTPUT split (column-parallel), rank r of 2, cols [r·out/2, (r+1)·out/2):
  trellis[:, r·out/32:(r+1)·out/32, :], svh[slice], bias[slice], suh FULL (replicated).
- INPUT split (row-parallel), rows [r·in/2, (r+1)·in/2):
  trellis[r·in/32:(r+1)·in/32, :, :], suh[slice], svh FULL, bias only on rank 0.
(This is exactly what `LinearEXL3.tp_import_split` does at load time today — exllamav3/modules/quant/exl3.py:330.)

| Component | Rule |
|---|---|
| Routed experts (trunk 40×384, MTP 3×384) w1/w3 (gate/up) | OUTPUT split (2304 → 1152) |
| Routed experts w2 (down) | INPUT split (2304 → 1152) |
| Shared experts (trunk + MTP, K=6) | same as routed |
| MoE router `ffn.gate.*` (fp16 384×5120 + bias, bias_vl) | REPLICATED |
| Attention wq_b (1280 → 64 heads×512) | OUTPUT split by heads (32 / 32) |
| Attention wo_a (8 groups, `wo_a.slice.N`) | slices 0–3 → rank0, 4–7 → rank1 |
| Attention wo_b (8192 → 5120) | INPUT split (4096 / 4096) → all-reduce |
| Attention wq_a, wkv, compressor.*, indexer.*, norms, attn sinks per head | wq_a/wkv/compressor/indexer REPLICATED; per-head tensors (e.g. sinks) split by head |
| HC mix/apply weights, layer norms | REPLICATED |
| Embedding (129280 × 5120) | split by vocab (64640 / 64640) |
| LM head | split by vocab (64640 / 64640); local top-1 + compare (no full logits gather) |
| Engram projections (layers 1, 14: q_weight, k_weight, wkv) | split like attention (heads) — verify shapes |
| Engram tables | NOT in shards; host RAM, one shared mmap/registered copy |
| DSpark drafter (mtp.0..2): attention/experts/shared | same rules as trunk; markov/confidence heads split by vocab if vocab-shaped, else replicated |
| Vision tower (32 layers, hidden 1024, 16 heads, MLP 2816) | attention by heads (8/8), MLP 1408/1408, output proj input-split |
| Aligner | split (column then row) |

## Runtime
- KV cache: sharded BY SEQUENCE across the two GPUs, **FP8** (nope pool via k_bits=8 path; rope + indexer keys
  evaluate fp8 too if quality holds). First fit test: 2M tokens.
- Collectives per decode token (bsz1): 126 (40 wo_b + 40 MoE + 2 embed/head + 8 indexer candidate merges +
  36 log-sum-exp merges) + ~30 per DSpark K=5 cycle. One-shot P2P all-reduce target 3–5 µs.
- DSpark K=5, drafter sharded TP=2.
- Decode step under CUDA graphs.
- Goal: first a successful small-token run (coherent text, no errors) on this design; then ≥150 tok/s (≤6.67 ms),
  stretch ~200 tok/s.

## Memory budget per GPU (GiB, from checkpoint headers)
weights ≈ 89.6 | KV 2M fp8 seq-sharded ≈ 2.0 | runtime ≈ 2–2.5 | total ≈ 93.6–94.1 of 95.59.

## Worktrees / containers (P7 GPU0+GPU1 only)
| Workstream | Worktree (branch) | Container |
|---|---|---|
| shard tool + runner (main) | `/home/treadall/recipe-2xrtx-tp2` (`recipe/2x-rtxpro6000-tp2`) | `exl3-tp2` |
| MoE kernel | `/home/treadall/recipe-2xrtx-tp2-moe` (`tp2/moe-kernel`) | `exl3-tp2-moe` |
| P2P all-reduce | `/home/treadall/recipe-2xrtx-tp2-p2p` (`tp2/p2p-allreduce`) | `exl3-tp2-p2p` |
Containers: image vllm-moet:dsv41-213d5fa, `--gpus device=0,1`, CUDA_HOME=/opt/cuda, repo at /src/exllamav3
(PYTHONPATH=/src/exllamav3), artifacts at /artifacts (= /home/treadall/recipe-artifacts/2x-rtx-tp2).
Each has its own torch extension cache. GPU memory is shared between the three containers: full-model runs only
in `exl3-tp2`; kernel microbenchmarks keep < 4 GiB per GPU.

## CONVENTION (owner directive, 2026-10-27): no hardcoded values — everything is a dial
Every tunable MUST be readable from an environment variable with a documented default, and MUST live in a named
constant near the top of its module — never inline in a function, never repeated as a literal. Applies to: batch size,
cache/history sizes, chunk sizes, draft length and its bounds, acceptance targets, thresholds, row/tile caps, chunking
of collectives, retry/iteration counts, timeouts, and every A/B switch. Existing examples to copy:
`MAX_NEW`, `MAX_CHUNK`, `CACHE_TOKENS`, `MAX_HISTORY`, `NUM_DRAFT`, `DSPARK_DYNAMIC`, `DSPARK_TARGET_ACC`, `COMM`,
`HYBRID_NCCL_MIN_BYTES`, `EXL3_WHOLE_GRAPH_MAXSEQ`, `EXL3_PW_MERGE`, `EXL3_MOE_COOP_KSPLIT/_WIDE/_WIDE_A/_WIDE_B`,
`EXL3_MOE_FUSED_ROWS*`, `EXL3_HC_FOLD`, `EXL3_DSV4_B1_BATCH`, `EXL3_MOE_COOP_SINGLE`, `EXL3_DSPARK_TAP_SCALE`.
New work must add dials, not literals, and must document them in `doc/env_vars.md` (one line: name, default, why).
