# TP2/EP2 bring-up: DeepSeek-V4.1-Flash EXL3 2.565 mixed-K

> Raw working notes from the development runs, kept because they hold the measurements. Host paths, container names and run tags refer to that machine; the results themselves are restated with conditions in [`../RESULTS.md`](../RESULTS.md).

Checkpoint: `<models>/dsv41-serve-overlay` (symlinked shards, Engram overlay L01/L14).
Hardware: GPU0+GPU1, 2× RTX PRO 6000 Max-Q (94.97 GiB each). GPU2 untouched.
Container: `<dev-container>` (devices 0,1), image `<previous-base-image>`.
Env: `EXL3_MOE_SPLIT_FUSED=0`, `EXL3_MOE_CPU_SWAP=0`. `Model.load(tensor_p=True, tp_output_device=0, max_batch_size=1, max_chunk_size=2048)`, `Cache(max_num_tokens=4096)`.

Bench: `qmesh/tp2_bench.py` (env `TP_BACKEND`, `MOE_TENSOR_SPLIT`, `IGNORE_EOS`, `PROFILE`, `TAG`, `TRACE_DIR`, `TOKENS_OUT`, `EQUAL_TO`, `PROMPT_IDS`).
All-reduce microbench: `qmesh/tp2_allreduce_bench.py`; graph capture of the ring: `qmesh/tp2_allreduce_graph.py`.
Smoke: `qmesh/tp2_smoke.py`. Per-rank memory probe: `qmesh/tp2_probe.py`.
Logs: `<artifacts>/tp2logs/`.

## Results

| Config | Load | max_alloc rank0 / rank1 (GiB) | Decode ms/tok (mean of 3 prompts, 64 tok) |
|---|---|---|---|
| native, EP2, CPU-reduce path (`EXL3_TP_GPU_REDUCE=0`), eager | 101.3 s | 86.70 / 85.80 | **20.28** |
| native, EP2, GPU ring (default), eager | 98.1 s | 86.70 / 85.80 | **20.20** |
| native, EP2, GPU ring, `EXL3_WHOLE_GRAPH=1` | 98.7 s | 86.70 / 85.80 | **17.66** (token-exact with GPU-ring eager) |
| native, EP2, CPU-reduce path, eager (earlier run) | 106.1 s | 86.95 / 86.05 | 20.74 |
| native, tensor-split MoE, eager | 106.2 s | 87.03 / 86.37 | 20.62 |
| nccl, EP2, eager | 96.1 s | 86.70 / 85.80 | 27.25 |
| nccl, EP2, `EXL3_WHOLE_GRAPH=1` | 96.5 s | 86.70 / 85.80 | 22.47 (the process then hung at exit: NCCL watchdog) |

Token equality, 64 tokens × 3 prompts:
- Graph (GPU ring) vs eager (GPU ring): **identical** on all 3 prompts (`tokens_graph_ring.json` vs `tokens_eager_ring.json`).
- GPU ring vs CPU-reduce path, both eager: **differ** on all 3 prompts, first divergence at token 17, 11, 14. The ring sums fp32 directly, while the CPU path reduces over a bf16 wire, so greedy argmax moves on near-ties. This is a reduce-precision change, not a graph bug. The reference for graph correctness is the ring eager run.

Coherence: all 3 prompts are coherent in the graph run (`The capital of France is **Paris**.`, a Fibonacci function, three primary colors).

Smoke (`qmesh/tp2_smoke.py`, prompt "Hi", earlier CPU path): `decode_ms_per_tok=26.00`, coherent.

Memory at load: rank0 86.32 GiB, rank1 85.42 GiB. Peak in decode ~0.4 GiB higher. Both fit under 94.97 GiB.

## All-reduce latency, (1, 5120) fp32, 2 ranks (GPU0/1)

| Path | Latency per op |
|---|---|
| CPU-reduce helper (`pg_all_reduce_cpu`), eager | 14.8 µs |
| GPU P2P ring (`pg_all_reduce`), eager | 20.3 µs |
| GPU P2P ring, graph replay (200 replays) | 18.6 µs (rank0) / 21.4 µs (rank1) |

Ring vs CPU-path max abs diff on the test vector: 2.35e-2 (fp32 vs bf16 wire, as above). Ring output is identical on both ranks.

## Launches per step

Under the CPU path (eager profile, 16 tokens): about 85 collective launches per step (2 all-reduces × 40 layers), 1359 recv and 1359 send over 16 steps. Collective recv+send: 21.7 ms (10.6%) of rank-0 CUDA time.

Measured on the graph config (`tp2_graph_ring_prof2.json`, 16-token profile, 16 steps): `cudaGraphLaunch` 80 per step (one per graph segment, A and B per block, 2 × 40). All-reduce kernels 84.9 per step, matching 2 collectives × 40 layers. The eager `cudaLaunchKernel`/`cudaLaunchCooperativeKernel` total (8400) includes the profiler warm-up and non-replayed attention kernels, so it is not reported as a clean per-step figure.

## Profile (rank 0, native EP2 CPU path, 16-token decode, `PROFILE=1`)

Rank-0 CUDA time total 205.4 ms over the profiled generation. Top costs by self CUDA time:

1. `exl3_gemv_int8_sq_kernel<6,1,false,…>`: 27.1 ms (13.2%), 1936 calls
2. `exl3_moe_coop_mixed_ns::exl3_moe_coop_a_mixed_k…` (MoE mixed-K A): 26.9 ms (13.1%), 640 calls
3. `pg_all_reduce_cpu_recv_kernel<0>` (collective recv): 17.2 ms (8.4%), 1359 calls

Self CPU time 167 ms vs CUDA 205 ms, so per-step CPU launch overhead is a large share of the 20 ms/token. Graphs remove most of it, which is consistent with 20.2 → 17.7 ms/tok.

Measured on the graph config (`tp2_graph_ring_prof2.json`, 16-token profile, 16 steps): `cudaGraphLaunch` 1280 total = 80 per step (one per graph segment, A and B per block, 2 × 40). All-reduce kernels 1359 total = 84.9 per step, matching 2 collectives × 40 layers. The eager `cudaLaunchKernel`/`cudaLaunchCooperativeKernel` total (8400) includes the profiler warm-up and the non-replayed attention kernels, so it is not a clean per-step graph figure and is not reported as one. Eager CPU-path baseline: about 85 collectives per step as well.

## Piecewise graphs under TP (`EXL3_WHOLE_GRAPH=1`): fixed

Root cause: `_forward_piecewise` (`transformer.py:250-260`) runs graph A, then the attention `tp_collect` eagerly, then graph B, and graph B contains the MoE `tp_collect`. The native `all_reduce` called `ext.pg_all_reduce_cpu`, a host-side CPU helper with shared-memory R buffers and per-pass job scheduling, which is not capturable and needs `run_cpu_reduce_jobs` at pass boundaries. Captured, it hangs (`Synchronization timeout`); eager, it times out in `run_cpu_reduce_jobs`.

Fix: `TPBackendNative.all_reduce` (`exllamav3/model/model_tp_backend.py`) now routes fp32 contiguous tensors (numel % 4 == 0) to `ext.pg_all_reduce`, the on-device P2P ring (`all_reduce.cu`, cooperative launch with `grid.sync()`), which is graph-capturable. A non-contributor zeroes its tensor first, since the ring sums all ranks and has no contribution flag. `EXL3_TP_GPU_REDUCE=0` restores the CPU-reduce path. Both MoE and attention reduces are fp32 on decode and go through the ring.

Result: graph decode runs to completion on all 3 prompts with no sync timeout, token-exact against the ring eager run, at 17.66 ms/tok versus 20.20 eager.

Side finding: the ring changes the reduce numerics relative to the CPU path (fp32 vs bf16 wire). Greedy tokens diverge from the CPU-path baseline at token 11–17 on these prompts. Main should pick which reference to hold the model to.

## Failures and fixes

- Chat template mismatch: prompt lacked `</think>` (fixed in bench, see Prompt format below).
- Stop condition not wired: `Job` ignored `tokenizer.eos_token_id` (fixed with `stop_conditions`).
- Per-rank memory probe: double reply from child caused `Timed out waiting for worker`. Fixed by a single `tp_worker_dispatch_single` call.
- Token-ID capture: generator emits `token_ids` per chunk and clears them, so the bench accumulates across results (earlier version kept only the last chunk).
- Encoding import: `encoding.py` is imported from its own directory (`.../DeepSeek-V4.1-Flash-1017-orig/encoding`).
- Concurrent runs OOM: two bench processes on the same GPUs exceed 94.97 GiB each. Run one at a time.
- NCCL backend graph run: process hung at exit (watchdog), not investigated.

## Prompt format (fixed in the bench)

The checkpoint ships no `chat_template`. The bench uses `encode_messages(..., thinking_mode="chat")` from the reference encoder, which ends the prompt with `<｜Assistant｜></think>`. `stop_conditions=[tokenizer.eos_token_id]` is set when `IGNORE_EOS=0`.

## Commands

```
docker exec -d -e CUDA_VISIBLE_DEVICES=0,1 -e PYTHONPATH=<repo> -e TAG=<tag> \
  [-e EXL3_WHOLE_GRAPH=1] [-e EXL3_TP_GPU_REDUCE=0] [-e IGNORE_EOS=1] [-e TOKENS_OUT=…] [-e EQUAL_TO=…] \
  [-e PROFILE=1 -e MAX_NEW=16 -e TRACE_DIR=…] -w <repo> <dev-container> \
  bash -c 'python3 qmesh/tp2_bench.py > <artifacts>/tp2logs/bench_<tag>.log 2>&1'
docker exec -e CUDA_VISIBLE_DEVICES=0,1 -e PYTHONPATH=<repo>:<repo>/tests -w <repo> <dev-container> bash -c 'python3 qmesh/tp2_allreduce_bench.py'
docker exec -e CUDA_VISIBLE_DEVICES=0,1 -e PYTHONPATH=<repo>:<repo>/tests -w <repo> <dev-container> bash -c 'python3 qmesh/tp2_allreduce_graph.py'
```

## Open items

- Profiled graph run: the first attempt hit `Synchronization timeout` under the torch profiler in the attention collective; the second identical run completed. Cause of the first failure unexplained (possibly profiler timing).
- Reduce numerics: ring (fp32) and CPU path (bf16 wire) diverge greedily; choose the reference.
- NCCL teardown hang: not investigated.
- LPT expert balancing: deferred (touches `tp_import` routing and MixedK indexing).

## Step 2: graph-mode decode profile, rank 0 (ring, `EXL3_WHOLE_GRAPH=1`)

Source: `<artifacts>/tp2logs/tp2_graph_ring_prof2.json`, 16 profiled decode steps (`PROFILE=1 MAX_NEW=16`). Shares are of total rank-0 kernel time; per-step ms = total / 16.

| Category | ms/step (profiler) | Share of kernel time | Kernels per step |
|---|---|---|---|
| MoE (`exl3_moe_coop_*`, `routing_*`) | 5.12 | 37.4% | 242 |
| GEMV/GEMM (`exl3_gemv_int8_sq`, `exl3_mgemm`, `exl3_gemm`, cublas/cutlass, `gemvx`) | 4.27 | 31.2% | 228 |
| All-reduce / collective (`pg_all_reduce_kernel`, `pg_gather`) | 2.28 | 16.7% | 86 |
| HC (`hc_mix_*`, `hc_apply_kernel`) | 0.81 | 5.9% | 255 |
| Attention core (`_dsa_attn_*`, `rope_kernel`, `dsv4_compress_windows_kernel`) | 0.76 | 5.5% | 183 |
| Other (norm, elementwise, index, copy) | 0.46 | 3.4% | 202 |
| **Kernel-busy total** | **13.70** | 100% | |

Per-step window (profiler-inflated): span 24.3–26.4 ms, kernel busy 13.6–14.3 ms, idle 10.5–12.6 ms (about 45% of the window). These absolute times are profiler-inflated and are NOT steady state. Steady state is 17.66 ms/tok (above). Use the shares, not the ms, for the breakdown. The global consecutive-kernel gap sum mixes streams and is not reported.

## Step 3: DSpark K=5 under TP2

Result: **does not fit on GPU0 next to the TP2 rank-0 shard.** No acceptance or ms/token numbers exist because the run never reaches decode.

Measured (native TP2, `CACHE_TOKENS`/`DRAFT_CACHE_TOKENS` env in the bench):

| Run | Target / draft cache | GPU0 free after drafter load | Outcome |
|---|---|---|---|
| `dspark_k5` | 4096 / 4096 | 0.31 GiB | OOM at `Generator.__init__` → `attach_to` → `_load_own_embed_head`, 632 MiB trellis alloc, 316.5 MiB free |
| `dspark_k5_mincache` | 4096 / 256 | 0.31 GiB | Assertion: draft cache must equal target cache size (`generator.py:257`) |
| `dspark_fit256` | 256 / 256 | 0.33 GiB | OOM, same 632 MiB head alloc, 338.5 MiB free |

Memory before drafter: GPU0 free 7.43 GiB with the TP2 model loaded (`alloc 86.32 GiB`). Drafter weights, activations and cache take about 7.1 GiB. The drafter's own lm head (`own_head`, loaded by `_load_own_embed_head` because the target is TP) then needs 632 MiB on the drafter's last-module device, GPU0. Shortfall: about 0.29–0.32 GiB. The KV cache size barely changes free memory (4096 → 256 tokens moves it by 0.02 GiB), so the cause is the weights and the head, not the cache.

Options (none applied; each changes the design):
- Place the drafter head on GPU1 (~8 GiB free there) or on CPU. Requires a change in `architecture/deepseek_v4_mtp.py` (`_load_own_embed_head` device choice) and a per-step cross-device logits copy in `sample_from_state`. Not tested.
- Reuse the trunk's head (`am.modules[am.logit_layer_idx]`) instead of a private copy, which is the non-TP branch in `sample_from_state`. Would reintroduce a TP collective inside the draft loop, which the code comment says was avoided on purpose.

Acceptance, mean tokens per verify and effective ms/token: **not measured** (blocked by the memory shortfall). Prior single-GPU reference, not TP (`E2ESeam.md`): acceptance 0.74–0.82, steady decode 17.6 ms/tok best.

The bench's DSpark path (`DSPARK=1`, `NUM_DRAFT`, `DRAFT_CACHE_TOKENS`, `CACHE_TOKENS`) is in `qmesh/tp2_bench.py`. Accepted/rejected counters are read from the job object. Verify-round count is not yet exposed, so mean tokens per verify is not computed.

Logs: `<artifacts>/tp2logs/bench_dspark_k5.log`, `bench_dspark_k5_mincache.log`, `bench_dspark_fit256.log`.

