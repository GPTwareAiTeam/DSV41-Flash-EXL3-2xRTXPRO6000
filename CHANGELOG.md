# Changelog

## Unreleased

### Added

- **`KV_BITS=4`: the main KV cache in DeepSeek's trained FP4 format.** The compressed latent (512 values, rope
  included) is stored as packed E2M1 with one E4M3 scale per 16 values (288 bytes per entry), indexer keys as
  packed E2M1 with one E8M0 scale per 32 (68 bytes); the quantizer matches DeepSeek's `fp4_act_quant` bit for
  bit. Main cache 3,200 → ~890 bytes per token per GPU. Prefill unpacks each chunk's pages once (exact decode)
  for the fp16 kernels; decode and verify decode in-kernel. Measured against 16-bit: decode 200.2 vs 201.6 tok/s
  and time to first token 8.34 vs 8.36 s at 24k (4 ABBA pairs); 5/5 planted facts at 128k in prose and code,
  same as 16-bit; repeat- and rank-identical. Output differs from 16-bit (lossy by design).
- **`MEM_BREAKDOWN=1`** logs per-GPU resident GiB by owner after load and the peak since load.
- **Adaptive prefill chunk (`PREFILL_ADAPTIVE=1`, default).** Before every prefill chunk the runner reads free
  memory once and solves a measured peak model (activations per row; a depth term for the uncompressed pool
  layer's staging window and the indexer score slab) for the largest chunk that leaves 1 GiB free, capped at
  `MAX_CHUNK`. Only ever shortens a chunk; ~0.57 ms per decision; both ranks agree. An 876k-token prompt with
  planned 8192-row chunks completes (6,656 → 3,328 rows as depth grows); short prompts keep full chunks and
  identical output. Larger `MAX_CHUNK` is now safe to request.

With the drafter ring below, both together: 1M-token context fits (cache 0.87 GiB per GPU); a 131k-token
prompt in a 1M cache peaks at 93.23 GiB of 94.97 per GPU with a 2048-row prefill chunk and 94.34 GiB with 4096.

### Changed

- **The drafter's cache is a small per-user window instead of a context-sized copy.** The DSpark drafter only
  ever reads the last 128 positions, but it kept a 3-layer, 512-wide row for every position of the context:
  3,072 bytes per token per GPU on top of the 3,200-byte main KV cache. It now keeps a 768-row ring per user
  per layer (`exllamav3/cache/dsa.py` `DSparkLayerState` / `DSparkState`), checkpointed and rolled back with
  the target's recurrent state. Measured on the same pinned GEMM tuning: output token-identical to v1.0.0 on
  both ranks (alternating 24k/short jobs, speed set, prefix-cache turn 2, 2 and 4 users); speed unchanged within
  noise; 0.19 GiB more free per GPU at a 64k cache and 1.50 GiB at 512k. A 150k-token prompt in a 512k cache now
  runs (85.5 tok/s decode); v1.0.0 runs out of memory there.
- **Indexer prefill default: 128-row score slabs (`EXL3_DSV4_SCORE_SLAB=128`).** Smaller transient score matrix
  at depth, same selected entries, token-identical. `EXL3_DSA_IDX_FP4_DIRECT=1` (read FP4 index keys in-kernel in
  prefill, 128 MiB less at 1M) is opt-in: its scores differ in the last fp16 bit and change generated tokens.
- **Opt-in: deep-context top-k attention reads FP4 pool entries in-kernel** (`EXL3_DSA_FP4_GATHER_DIRECT_MIN`,
  e.g. 262144) instead of unpacking the whole window to fp16. Bit-identical. One 2048-row chunk at a 1M-entry pool:
  3.2 ms / 1.23 GiB transient → 2.2 ms / 64 MiB; the full 876k-token prefill measured 5% slower (375 vs 357 s) for
  0.14 GiB lower peak, so it is off by default.
- **Verify steps score every row's candidate entries in one launch.** The candidate-block indexer layers scored
  each verify row in its own Python iteration (key gather, launch, two fills per row); the few-query indexer
  kernel now reads each row's pool rows through the entry map the compaction kernel already builds. Bit-identical
  scores; scoring at 6 rows 537 → 23 µs (FP4, 8192 entries). Full model, 26.9k-token prompt, 512 new tokens,
  3 ABBA pairs: decode 4.57 → 4.44 ms per token (+2.6% tok/s), token-identical in all 12 runs.

### Fixed

- Token-identity comparisons between two builds must pin the GEMM autotuner's choices
  (`EXLLAMAV3_TUNE_CACHE` pointing at copies of one tuning file): each container tunes its own kernels, and
  different picks alone change the numerics enough to flip greedy tokens a few hundred tokens in.
- Prefill-sized staging and attention-output buffers in `DSV4Attention._forward_cached_one` were kept for the life
  of the process, one copy per distinct chunk size (~200 MiB each): varying chunk sizes ran the card out of memory.
  They now come from the caching allocator; decode-sized buffers stay kept for graph capture.
- FP4 prefill staging decodes in slices instead of a full 512-wide copy on top of the fp16 windows.

### Known issues

- With `KV_BITS=4`, output can differ between prefill chunk sizes at near-tied tokens (top-2 logit gap ~0.05);
  16-bit KV is chunk-invariant.

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
  prompt still runs with 1.42 GiB per GPU used by another process and fails at 1.92 GiB; `MAX_CHUNK=2048`
  runs with 2.92 GiB used. (Corrected after release: earlier text gave the requested ballast sizes, 0.75 /
  1.25 / 2.25 GiB, not the memory actually used.)
- Batching is bitwise invariant across streams and neighbours but not across batch width; near-ties can flip.
- FP8 KV breaks capture and is numerically wrong when capture is bypassed — `KV_BITS=16` only.
- 8-bit prefill all-reduce is faster but fails the coherence gate (prompt-3 token flip at step 11); opt-in only.

### Notes

- No model weights, logs, credentials or host-specific paths are distributed with this repository. Run
  directories in the docs are written as `<artifacts>/…`.
- Upstream `exllamav3/`, `doc/`, `examples/`, `tests/`, `util/` and `science/` code is unchanged except for
  additive hooks; see [CREDITS.md](CREDITS.md) for the third-party licences vendored in the tree.
