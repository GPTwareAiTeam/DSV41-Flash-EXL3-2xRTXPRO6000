# Changelog

## v1.0.0 — 2026-10-29

First public release of the TP=2 runtime: ExLlamaV3 (upstream, MIT) plus a tensor-parallel backend, a sharded
checkpoint format, a runner, and the measurements to go with them. Upstream's README is preserved as
[README.exllamav3.md](README.exllamav3.md); the project overview is [README.md](README.md) and every number
below is documented with its conditions in [docs/tp2/RESULTS.md](docs/tp2/RESULTS.md).

### Added

- **True TP=2 execution** for a large MoE checkpoint (`qmesh/tp2/run_tp2.py`, `exllamav3/tp2/`). One process per
  GPU; attention heads, KV heads, vocab embedding/head and the MoE intermediate dimension are split two ways;
  each rank builds a single-device ExLlamaV3 model with `tp_size=2, tp_rank=r`, so the loader, cache and sampler
  are upstream code.
- **One-shot P2P all-reduce** (`exllamav3/exllamav3_ext/parallel/p2p_oneshot.cu`, `exllamav3/tp2/p2p.py`):
  CUDA-IPC arena, device-side sequence counters so it is graph-capture safe, optional device spin/timing stats,
  host-memory fallback when peer access is unavailable. 4.93 µs per decode-sized call against NCCL's 11.84 µs.
- **`COMM=hybrid`**: P2P for decode-sized payloads, NCCL above a byte threshold. +16.5% prefill over P2P on a
  32k prompt while keeping the decode path. `COMM=nccl` and `COMM=p2p` still selectable.
- **Piecewise CUDA graph capture and cross-block merge** for the decode step, with explicit capture-eligibility
  checks that log the fallback instead of silently degrading.
- **Batched capture path** (`EXL3_DSV4_B1_BATCH`) enabling 2–8 concurrent streams: 93.7 tok/s at B1 (same
  kernel path, one row) to 203–206 tok/s aggregate at B4.
- **Speculative decoding (DSpark)**: sharded drafter head, one block forward per round, rank-0 decisions
  broadcast so both ranks verify the same width. `DSPARK`, `NUM_DRAFT`, `DSPARK_POLICY`, `DSPARK_DYNAMIC`,
  `DSPARK_TARGET_ACC`.
- **Prefix cache** (`EXL3_PREFIX_CACHE`, default on): recurrent-state checkpoints at chunk boundaries,
  36–40× TTFT at 20k and 86–88× at 60k on a full hit, ~268 MB host RAM per rank per 60k context.
- **Auto prefill chunk sizing** (`MAX_CHUNK=auto`) from a measured device-peak table against the KV budget, so
  a 1M-token context costs one chunk step instead of an OOM.
- **CPU-only sharding tools** (`qmesh/tp2/shard_tool.py`, `verify_shards.py`) that turn a full checkpoint into
  the TP=2 overlay, with a written spec ([docs/tp2/SHARD_SPEC.md](docs/tp2/SHARD_SPEC.md)) and a bit-exact
  verifier.
- **Benchmarks and launchers**: `scripts/bench_v1.sh` (reference end-to-end run with a rank-to-rank token
  comparison), `qmesh/tp2/{launch_tp2.sh,run_prefix_bench.sh,compare_launch_merge.py}`, and standalone
  MoE/P2P/prefill benches under `qmesh/`.
- **Documentation**: [docs/tp2/DESIGN.md](docs/tp2/DESIGN.md), [RECIPE.md](docs/tp2/RECIPE.md),
  [RESULTS.md](docs/tp2/RESULTS.md), [env_vars.md](docs/tp2/env_vars.md),
  [SHARD_SPEC.md](docs/tp2/SHARD_SPEC.md), and the raw working notes in `docs/tp2/notes/`.

### Performance, relative to the pre-release tree on the same hardware

- Decode: 13.35 → **10.67 ms/token** (75 → 93.7 tok/s) from capturing the decode attention core, a MoE
  prefetch/PDL hoist, and three fused launch merges; launches per step 630.8 → 585.0 at 24k context.
- Decode MoE kernel: 67.31 → **62.54 µs/call** at bsz1 (1.076×) from removing a local-memory load and hoisting
  the trellis prefetch ring.
- Prefill: 3.9k → **4.78k tok/s** at 32k from chunk sizing; +16.5% from `COMM=hybrid`.
- Long context: `aten::nonzero` 8 → **0** and launches 585 → **557** per step at 24k from device compaction;
  16.4 → **16.1 ms/token**.
- Batching: 92 → **203–206 tok/s** aggregate at B4.

### Known issues

- The captured decode path stops at roughly 1024 positions; long context runs eager (16.1 ms/token at 24k).
- A batch containing any long row drops the whole batch to eager.
- Batching is bitwise invariant across streams and neighbours but not across batch width; near-ties can flip.
- FP8 KV breaks capture and is numerically wrong when capture is bypassed — `KV_BITS=16` only.
- 8-bit prefill all-reduce is +6% but fails the coherence gate (prompt-3 token flip at step 11); not shipped.
- Speculative decoding loses to plain decode at B4; the two are substitutes at these widths.

### Notes

- No model weights, logs, credentials or host-specific paths are distributed with this repository. Run
  directories in the docs are written as `<artifacts>/…`.
- Upstream `exllamav3/`, `doc/`, `examples/`, `tests/`, `util/` and `science/` code is unchanged except for
  additive hooks; see [CREDITS.md](CREDITS.md) for the third-party licences vendored in the tree.
