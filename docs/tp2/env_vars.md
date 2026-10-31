# Environment dials

Every tunable in the TP=2 runner is an environment variable with a documented default. This page lists the
ones the runtime reads itself, i.e. everything you can set when starting the two rank processes. Engine-level
switches (kernel selection, graph capture, fusion, KV quantisation) are documented one screen at a time in
[`../../doc/env_vars.md`](../../doc/env_vars.md); the subset that matters most for TP=2 is summarised at the
end of this page.

The runner is `python -m qmesh.tp2.run_tp2`, one process per GPU. `scripts/bench_v1.sh` is the reference
invocation.

## Required

| Variable | Meaning |
|---|---|
| `RANK` | 0 or 1. Which GPU (`cuda:RANK`) and which rank directory of the overlay this process loads. |
| `WORLD_SIZE` | `2`. Must match the sharding of the checkpoint. |
| `INIT_METHOD` | torch.distributed rendezvous, e.g. `file://<artifacts>/tp2logs/<tag>_rendezvous`. |
| `OVERLAY` | Directory holding `rank0/` and `rank1/` (default `/artifacts/tp2-overlay`). |

## Paths and runtime

| Variable | Default | Meaning / why |
|---|---|---|
| `TORCH_EXTENSIONS_DIR` | `/root/.cache/torch_extensions` | Where torch's JIT-built extension lands. Set it to a persistent path or every container restart recompiles ~2 min of CUDA. |
| `P2P_RDV` | — (required with `COMM=p2p`/`hybrid`) | Directory for P2P handshake files. Must be **fresh** per run; a stale file makes the ranks pair with a dead peer. |
| `P2P_CHUNK` | `ARENA_MAXB // 4` fp32 words | Arena chunk size for the all-reduce. Prefill lever only; larger chunks mean fewer, longer calls. |
| `HYBRID_NCCL_MIN_BYTES` | `65536` | `COMM=hybrid` threshold: payloads below this go P2P, larger go NCCL. Decode payloads are ~10 KB, prefill payloads are MB. |
| `COMM` | `p2p` | `p2p` \| `hybrid` \| `nccl`. Decode favours P2P (4.93 vs 11.84 µs/call), prefill favours NCCL (+12% at 2048 tokens) and `hybrid` combines both (+16.5% over P2P at 32K prefill). See [RESULTS.md](RESULTS.md#3-prefill). |
| `P2P_STATS_OUT` | unset | Prefix: dumps device spin/timing stats to `<prefix>.rank<R>.json` at exit. Costs a little device time; for diagnosis only. |
| `TP_COLLECTIVE_TRACE` | `0` | Log a per-collective trace. Diagnosis only. |

## Workload

| Variable | Default | Meaning / why |
|---|---|---|
| `MAX_NEW` | `64` | Generated tokens per prompt. |
| `PROMPT_IDS` | `0,1,2` | Which built-in prompts to run. |
| `SHORT_PROMPTS` | unset | `1` swaps in a short prompt set (useful for invariance and batching tests). |
| `LONG_PROMPT_TOKENS` | `0` | Prepend N tokens of filler to each prompt. |
| `LONG_PROMPT_STYLE` | `prose` | Filler style: `prose`, `code`, `json`, `mixed`. Content changes speculative acceptance, so always record it. |
| `IGNORE_EOS` | `0` | `1` generates exactly `MAX_NEW` tokens even past EOS. Required when comparing token streams — an early stop makes the comparison look identical. |
| `WARM` | `0` | `1` runs every prompt twice and times the second pass, so the graph/allocator warmup is excluded. Also makes ~24k TTFT an artefact of page reuse — do not quote TTFT from a `WARM=1` run. |
| `WARM_TOKENS` | `8` | Untimed warm pass length. Never 0 when graphs are on: the first step is the capture. |
| `BATCH` | `1` | Concurrent streams: number of cache slots and `max_batch_size`. B>1 requires `EXL3_DSV4_B1_BATCH=1` or it silently falls back to the eager batched path (2× slower). |
| `KV_BITS` | `4` | `4` (default) stores the compressed main pools in DeepSeek's trained FP4 format (packed E2M1; E4M3 scale per 16 latent values, E8M0 per 32 index-key values): 3200 → ~890 B/token/rank, decode and prefill within 1% of 16-bit. `8` selects `CacheLayer_quant` (3200 → 2150 B/token/rank) but breaks graph capture and is numerically wrong when the capture is bypassed. |
| `CACHE_TOKENS_PER_SEQ` | `4096` | KV pool tokens per stream when `CTX_TOKENS` is 0. |
| `CTX_TOKENS` | `0` | Total KV budget in tokens; 0 = `BATCH × CACHE_TOKENS_PER_SEQ`. Set it to the context you actually want to serve (e.g. `1048576`) and the chunk auto-sizer gets whatever VRAM is left. |
| `KV_BYTES_PER_TOKEN` | `890` with `KV_BITS=4`, `2150` with `KV_BITS=8`, else `3200` | Per-rank pool cost used by the chunk sizing arithmetic. Only change it if you change the cache type. |
| `MEM_BREAKDOWN` | `0` | `1` logs `MEMB` lines: per-GPU resident GiB by owner (weights, KV cache, recurrent state, statics, generator) after load, device/torch totals, and the peak since load at the end of the run. |
| `FORCE_EXPORT` | `0` | `1` makes the target model emit `export_state_layers` (drafter input taps) during a plain run. Needed to collect drafter training data, not for serving. |

## Prefix cache

| Variable | Default | Meaning / why |
|---|---|---|
| `EXL3_PREFIX_CACHE` | `1` | Recurrent-state checkpointing and reuse. Implies `EXL3_PREFILL_RC_SPLIT=1`. Measured 36–40× TTFT at 20k and 86–88× at 60k on a full hit; short-context decode is unchanged. |
| `EXL3_PREFIX_CACHE_CAPACITY` | `4294967296` (4 GiB) | **Host RAM** budget for checkpoints, not VRAM. A 60k context costs ~268 MB per rank. |
| `EXL3_PREFIX_CACHE_MIN_MATCH` | `256` | Shortest prefix that is worth replaying; below this the cold prefill is cheaper. |
| `EXL3_PREFIX_CACHE_INTERVAL` | `1024` | Checkpoint spacing in tokens; must be a positive multiple of 256. Larger spacing means fewer, cheaper checkpoints but coarser hits. The shipped configuration takes checkpoints at 8192-token chunk boundaries. |

## Prefill chunk

| Variable | Default | Meaning / why |
|---|---|---|
| `MAX_CHUNK` | `auto` | Prefill rows per forward. `auto` picks the largest candidate whose *measured* device peak fits the KV budget with reserve. |
| `PREFILL_CHUNK_STEPS` | `16384,8192,4096,2048` | Candidates for `auto`, largest first. |
| `PREFILL_CHUNK_MIN` | `512` | Floor if nothing fits. |
| `CHUNK_PEAK_TABLE` | `2048=87600,4096=90600,8192=94000,16384=97100` | MiB-of-device-peak samples from real prefill runs, in `rows=MiB` pairs. Interpolated linearly between entries and extrapolated past the last. Replace these if your checkpoint or KV size differs — they are the only device-specific input to the sizing. |
| `CHUNK_PEAK_REF_TOKENS` | `33536` | KV token count at which the table was sampled; the table is re-based to your KV budget. |
| `VRAM_RESERVE_MIB` | `2048` | Kept free for graphs, workspaces and allocator slack. |
| `DSPARK_RESERVE_MIB` | `2048` | Extra reserve when `DSPARK=1`: an 8192-row chunk with a long-context drafter exceeded the device by ~640 MiB. Empirical, not a guarantee. |
| `PREFILL_ADAPTIVE` | `1` | Before every prefill chunk, read free memory once and solve the peak model below for the largest chunk (whole 256-row pages, at most the planned `MAX_CHUNK`) that leaves `PREFILL_MARGIN_MIB` free on the card; both ranks take the minimum. Only ever shortens a chunk. ~0.57 ms per chunk. `0` = fixed chunks. |
| `PREFILL_MARGIN_MIB` | `1024` | Card memory left free at the predicted peak. |
| `PREFILL_ROW_KIB` | `382` | Peak model: activation memory per prefill row (measured, flat in depth). |
| `PREFILL_DEPTH_BASE_MIB`, `PREFILL_DEPTH_BYTES` | `461`, `1280` | Peak model: the depth term (uncompressed pool layer's fp16 staging window + indexer score slab) is `461 MiB + 1280 B × (chunk end position)` at 2048 rows, and dominates past ~300k tokens. The rise is `max(row term, depth term)`. Fitted to 598 chunks of an 876k-token prefill, FP4 KV. |
| `PREFILL_FRAG` | `0` | Share of torch's held-but-free memory not counted as usable (expandable segments make it reusable). |
| `PREFILL_ADAPTIVE_LOG` | `0` | `1` logs each chunk's decision, measured rise vs prediction, and rows/s. |

## Speculative decoding (DSpark)

| Variable | Default | Meaning / why |
|---|---|---|
| `DSPARK` | `0` | Enable speculative decoding. The drafter is sharded alongside the target. |
| `NUM_DRAFT` | `5` | Draft tokens per round. Round cost is near-flat in K, so K=5 is the default; the whole K choice is worth ≤ 8%. |
| `DSPARK_DYNAMIC` | `0` | Pick each round's window from calibrated acceptance instead of a fixed K. Pair with `EXL3_DSPARK_CONF=0` so the drafter's own gate does not also cut the window. |
| `DSPARK_TARGET_ACC` | `0.7` | Target conditional acceptance for `DSPARK_DYNAMIC`. |
| `DSPARK_POLICY` | `always` | `always` \| policy that can skip drafting per request when the estimate is below threshold. Requires `DSPARK=1`, `BATCH=1`, `DSPARK_DYNAMIC=0`. Measured: fixed K=5 beat `auto` on 3 of 6 matched prompts. |
| `MAX_HISTORY` | `max(5, NUM_DRAFT)` | Recurrent-cache history. The generator requires `max_history ≥ K`; do not lower it independently of `NUM_DRAFT`. |

## Measurement

| Variable | Default | Meaning / why |
|---|---|---|
| `TOKENS_OUT` | unset | JSON path for generated token ids (per rank). The only way to prove two configurations produced identical output. |
| `LOGITS_OUT` | unset | Prefix for saved logits; `LOGITS_FULL=1` saves every position, not just the top-5. |
| `PROBE` | unset | Prompt token positions at which to dump hidden states. |
| `PROFILE` | `0` | torch profiler around the timed section. Changes overlap and kernel time, so never compare a profiled ms/token with an unprofiled one. |
| `PROFILE_SEQ` | unset | Restrict profiling to specific sequence indices. |
| `PROFILE_TRACE` | `/artifacts/tp2logs/profile_trace.json` | Chrome-trace output path. |
| `DRAFT_TIMING` | `0` | Per-round draft/target/wall breakdown. |
| `DSPARK_COUNT_FORWARDS` | `0` | Count drafter and target forwards per round; the cheapest way to check a policy change did what you think. |

## Engine-level switches that matter for TP=2

Defaults shown are the engine's compiled-in defaults. The TP=2 runner and server (`qmesh/tp2/run_tp2.py`, `serve_tp2.py`) set `EXL3_WHOLE_GRAPH=1` and `EXL3_DSV4_B1_BATCH=1` unless they are already in the environment.

| Variable | Default | Effect |
|---|---|---|
| `EXL3_WHOLE_GRAPH` | `0` (runner/server: `1`) | Capture the decode step as piecewise CUDA graphs, one segment per transformer block. **Turn this on** — it is 13.15 → 10.97 ms/token, the single largest decode win. |
| `EXL3_DSV4_B1_BATCH` | `0` (runner/server: `1`) | Use the captured single-token body for batches; required for `BATCH>1` to reach the fast path. Without it B4 measures 39 instead of 17.9 ms/token and looks plausible. |
| `EXL3_DSV4_NO_XFAN` / `EXL3_DSV41_XFAN` | `0` / `0` | Attention fan-out layout. V4.1 (no q head-norm) needs `EXL3_DSV41_XFAN=1` for the fused path; leave both alone unless you are changing attention. |
| `EXL3_WHOLE_GRAPH_MAXSEQ` | `8` | Longest sequence the piecewise capture will handle; beyond it the block runs eager. |
| `EXL3_PW_MERGE` | `1` | Merge consecutive eligible decode blocks into one graph. |
| `EXL3_PW_TPCOLLECT` | `0` | Capture the TP all-reduce inside the graph (requires `EXL3_WHOLE_GRAPH=1` and the P2P backend). Measured ~0.3% at B1, 2.4% at B4 — opt-in, not default. |
| `EXL3_DSV41_FUSED_ROPE`, `EXL3_DSV41_FUSED_POOL_SCATTER`, `EXL3_DSV4_FUSED_INDEXER_ROPE` | `0` (default-on in the shipped recipe) | Three fused launch merges: −1.0% short-context, −1.4% at 24k, launches/step 630.8 → 585.0. Each `=0` disables one. |
| `EXL3_DSV4_SCORE_SLAB` | `128` | Query rows per indexer score slab in prefill. Smaller = less transient memory at depth (the slab is rows × pool entries fp16). |
| `EXL3_DSA_IDX_FP4_DIRECT` | `0` | `1` reads FP4 indexer keys in-kernel in prefill scoring instead of unpacking them to fp16 once per call: 128 MiB less at 1M, but scores differ in the last fp16 bit and generated tokens change, so off by default. |
| `EXL3_DSA_FP4_GATHER_DIRECT_MIN` | off (`2^40`) | Pool entries above which top-k FP4 attention decodes the selected entries in-kernel instead of staging the window (1.23 GiB → 64 MiB per call at 1M; bit-identical). Off by default: the full 876k prefill measured 5% slower for 0.14 GiB lower peak. `262144` turns it on for memory-bound runs. |
| `EXL3_MOE_COOP_KSPLIT` | unset | Split the coop MoE kernel's k dimension. Measured worse at 2 and 3; leave unset. |
| `EXL3_MOE_COOP_WIDE` / `_WIDE_A` / `_WIDE_B` | unset | Wider tiles. All measured slower at bsz1 (the kernel is parallelism-limited, not bandwidth-limited). |
| `EXL3_MOE_COOP_SINGLE` | `0` | Single-pass coop kernel variant. |
| `EXL3_MOE_FUSED_ROWS_WIDE` | `256` | Row cap for the wide fused MoE path. |
| `EXL3_HC_FOLD` | `1` | Fold the HC mix into fewer launches. |
| `EXL3_MOE_SPLIT_FUSED` | `1` (runner forces `0`) | Fused split path; the TP=2 runner disables it. |
| `EXL3_MOE_CPU_SWAP` | `1` (runner forces `0`) | CPU-swapped expert path; off in TP=2 because every expert is resident. |
| `EXL3_DSPARK_CONF` | `0.5` | Drafter confidence gate. Set `0` when running `DSPARK_DYNAMIC` so the two gates do not compound. |
| `EXL3_DSPARK_GATE` | `auto` | How each round's verify width is chosen. `auto`: confidence gate whose cutoff drops to `EXL3_DSPARK_AUTO_LO` on low-acceptance text (+0.6% tok/s over 19 prompts, best +3.0%, worst −0.9%). `thr`: fixed `EXL3_DSPARK_CONF`. `ev`: expected-value width, opt-in (losses up to −12% tok/s on some prompts). Details in [doc/env_vars.md](../../doc/env_vars.md). |
| `EXL3_DSPARK_AUTO_LO`, `EXL3_DSPARK_AUTO_SWITCH`, `EXL3_DSPARK_AUTO_HL` | `0.25`, `2.0`, `32` | `auto` gate: low cutoff, accepted-drafts-per-round switch point, and half-life in rounds of its running average. |
| `EXL3_DSPARK_EV_SCALE`, `_EV_GAMMA`, `_EV_COST`, `_EV_KMIN` | see doc/env_vars.md | `ev` gate calibration. |
| `EXL3_ROUTER_INT8` | `0` | `1` opts into the int8 multi-row router projection; the default fp16 router is the tested path. |
| `EXL3_DSPARK_TAP_SCALE` | `1.0` | Scale on the drafter's input taps. |
| `EXL3_PREFILL_RC_SPLIT` | `0` (implied `1` by the prefix cache) | Split prefill at recurrent-checkpoint boundaries so a partial prefill can still be cached. |
