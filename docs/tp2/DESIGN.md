# TP=2 design contract

How the model is laid out across two GPUs, and the invariants the runtime relies on. The numbers this
design produces are in [`RESULTS.md`](RESULTS.md); the checkpoint it consumes is in [`RECIPE.md`](RECIPE.md).

## Source
- Checkpoint (read-only): `<models>/dsv41-exl3-mixed-2.565` (177.92 GiB, 41 shards)
- External Engram tables (read-only, host DDR5 ECC RAM, ONE shared copy for both ranks):
  `<models>/dsv41-serve/engram-only-L01.safetensors`, `engram-only-L14.safetensors` (FP8 E4M3 + E8M0 scale)
- Expert bitrates and layer positions stay EXACTLY as quantized (hot 3.5 → cold 2.0). Nothing is requantized,
  moved or re-ranked. Every expert is cut down the middle.

## Output (pre-sharded offline)
`<models>/dsv41-exl3-mixed-2.565-shard2/{rank0,rank1}/`
(safetensors + `model.safetensors.index.json` + `config.json` with `tp_rank`, `tp_size=2` + `shard_manifest.json`).
Disk: ~179 GiB total; <models> has ~200 GiB free → do not write anything else large there.

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
- Collectives per decode token (bsz1): 126 (40 `wo_b` + 40 MoE + 2 embed/head + 8 indexer candidate
  merges + 36 log-sum-exp merges) + ~30 per DSpark K=5 cycle. That count is why the all-reduce path —
  one-shot P2P over PCIe, not NCCL — is the single most important piece of the runtime.
- KV cache: sharded **by sequence** across the two GPUs. The default pool is fp16; `KV_BITS=8` selects the
  quantized `CacheLayer_quant` pool when the context, not the weights, is the binding constraint.
- DSpark speculative decoding at K=5 with the drafter itself sharded TP=2.
- Whole decode step captured in piecewise CUDA graphs, with an eager fallback when a step is ineligible.

## Memory budget per GPU (GiB, from checkpoint headers and `nvidia-smi` peaks)

| Item | GiB |
|---|---|
| Weights (one rank's shards) | ≈ 89.6 |
| KV pool, 2M tokens, fp8, sequence-sharded | ≈ 2.0 |
| Runtime (graphs, workspaces, collectives, allocator slack) | ≈ 2–2.5 |
| **Total** | **≈ 93.6–94.1 of 95.59** |

The prefill chunk is not budgeted by hand: `MAX_CHUNK=auto` sizes it from a table of *measured* device
peaks (see [`env_vars.md`](env_vars.md)), because the shard size on disk (89.81 GiB) is not the resident
footprint, and a weights-plus-overhead estimate would be wrong in the unsafe direction.

## Running it

The runtime expects a container with CUDA 13, PyTorch and Triton preinstalled (see `docker/Dockerfile.tp2`),
the repository on `PYTHONPATH`, the pre-sharded checkpoint reachable through `OVERLAY`, and two ranks
started with `RANK` / `WORLD_SIZE` / `INIT_METHOD`. `scripts/bench_v1.sh` shows the exact invocation.
No GPU is used before the model is loaded, so the sharding and verification steps in
[`RECIPE.md`](RECIPE.md) run on any CPU-only host.

## Convention: no hardcoded values — everything is a dial
Every tunable MUST be readable from an environment variable with a documented default, and MUST live in a named
constant near the top of its module — never inline in a function, never repeated as a literal. Applies to: batch size,
cache/history sizes, chunk sizes, draft length and its bounds, acceptance targets, thresholds, row/tile caps, chunking
of collectives, retry/iteration counts, timeouts, and every A/B switch. Existing examples to copy:
`MAX_NEW`, `MAX_CHUNK`, `CACHE_TOKENS`, `MAX_HISTORY`, `NUM_DRAFT`, `DSPARK_DYNAMIC`, `DSPARK_TARGET_ACC`, `COMM`,
`HYBRID_NCCL_MIN_BYTES`, `EXL3_WHOLE_GRAPH_MAXSEQ`, `EXL3_PW_MERGE`, `EXL3_MOE_COOP_KSPLIT/_WIDE/_WIDE_A/_WIDE_B`,
`EXL3_MOE_FUSED_ROWS*`, `EXL3_HC_FOLD`, `EXL3_DSV4_B1_BATCH`, `EXL3_MOE_COOP_SINGLE`, `EXL3_DSPARK_TAP_SCALE`.
New work must add dials, not literals, and must document them in [`env_vars.md`](env_vars.md) and, for
engine-level switches, in `doc/env_vars.md` (one line: name, default, why).
