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
- **Batched capture path** (`EXL3_DSV4_B1_BATCH`) for 2–8 concurrent streams; on short prompts without
  speculation, 203–206 tok/s total at 4 streams.
- **Speculative decoding (DSpark)**: sharded drafter head, one block forward per round, rank-0 decisions
  broadcast so both ranks verify the same width. Captured drafter graphs keyed by the cache's full page table,
  so jobs of different lengths never re-capture. `DSPARK`, `NUM_DRAFT`, `DSPARK_POLICY`, `DSPARK_DYNAMIC`,
  `DSPARK_TARGET_ACC`.
- **Adaptive speculation gate** (`EXL3_DSPARK_GATE=auto`, default): the drafter-confidence cutoff drops on
  low-acceptance text and resets per job. +0.6% tok/s over 19 prompts (best +3.0% multilingual 2k, worst −0.9%),
  every repeat token-identical. The target still verifies every token; only the verify width changes.
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
- **OpenAI-compatible HTTP server** (`qmesh/tp2/serve_tp2.py` with the `serve_tp2.sh` launcher,
  [docs/tp2/SERVER.md](docs/tp2/SERVER.md)): stdlib HTTP/SSE front end that drives both ranks in lockstep from
  one scheduler, with chat/completions endpoints, tools, streaming usage, prefix-cache reuse and optional bearer
  auth. Greedy output is token-identical to `run_tp2.py`, checked by `scripts/server_smoke.sh`.

### Performance, relative to the pre-release tree on the same hardware

All single-user figures are decode speed with speculation on (default settings), 512 generated tokens.

- Decode at 24k context: **142 tok/s** agentic, 177 code, 189 math, 169 structured, 97 prose, 100 multilingual
  (pre-release: 62 tok/s at 24k without speculation).
- Long context: **128 tok/s** at 72k, **124 tok/s** at 128k (agentic).
- Decode MoE kernel: 1.076× faster per call at bsz1 from removing a local-memory load and hoisting the trellis
  prefetch ring.
- Cold prefill (no prefix cache): 2,614 tok/s at 24k, 2,782 at 72k, 2,746 at 128k.
- 4 concurrent users at 12k context: **189 tok/s** total.

### Fixed

- **Mixed-rate MoE prefill decoded experts at the wrong bitrate.** In a prefill chunk, experts with more than
  32 routed rows go through the batched reconstruct tier, which decoded every expert of a group at one K. On
  mixed-rate packs (2 / 2.5 / 3 / 3.5 / 4 bpw experts) those experts were reconstructed wrong, so long prompts
  were read inaccurately and the result depended on the prefill chunk size. Groups are now planned per rate and
  decoded at their members' own K; the per-expert DQ path also uses each expert's own K. Covered by a mixed-rate
  parity test against the per-expert path.

### Known issues

- **Multi-user throughput scales weakly.** With several users, decode attention runs one user at a time
  (per-user graph replays, `exllamav3/modules/dsv4.py` `_forward_cached`); only the MoE and dense layers are
  batched. Measured at 12k context per user: 4 users 189 tok/s total with speculation; without speculation
  (pre-fix build) 4 users 91 and 8 users 93 tok/s total. A whole-batch attention graph is the planned fix.
- **8 concurrent users with speculation fail.** At 16k KV per user they run out of memory; at 8k per user the
  drafter's graph capture fails (`Cannot copy between CPU and CUDA tensors during CUDA graph capture`).
  8 users without speculation run.
- **Little VRAM is spare.** At the default 4096-row prefill chunk (1 user, 32k context, speculation on), a 24k
  prompt fails once more than ~1 GiB per GPU is taken; `MAX_CHUNK=2048` leaves room for at least 2.25 GiB.
- Batching is bitwise invariant across streams and neighbours but not across batch width; near-ties can flip.
- FP8 KV breaks capture and is numerically wrong when capture is bypassed — `KV_BITS=16` only.
- 8-bit prefill all-reduce is faster but fails the coherence gate (prompt-3 token flip at step 11); opt-in only.

### Notes

- No model weights, logs, credentials or host-specific paths are distributed with this repository. Run
  directories in the docs are written as `<artifacts>/…`.
- Upstream `exllamav3/`, `doc/`, `examples/`, `tests/`, `util/` and `science/` code is unchanged except for
  additive hooks; see [CREDITS.md](CREDITS.md) for the third-party licences vendored in the tree.
