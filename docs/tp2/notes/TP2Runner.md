# TP2Runner status (milestone 1 complete; M2 next)

> Raw working notes from the development runs, kept because they hold the measurements. Host paths, container names and run tags refer to that machine; the results themselves are restated with conditions in [`../RESULTS.md`](../RESULTS.md).

## Done
- Overlay: `<artifacts>/tp2-overlay/rank{0,1}` (host) = `<artifacts>/tp2-overlay/rank{0,1}` (container). Rank shard symlinks, `engram-only-L01/L14` symlinks, rank `config.json`.
- Arch (`exllamav3/architecture/deepseek_v41.py`): local dims for tp_size=2 (num_q_heads 32, o_groups 4, moe_intermediate 1152); embed and head vocab split = `vocab_size // tp_size` (64640).
- Runner `qmesh/tp2/run_tp2.py` + `qmesh/tp2/launch_tp2.sh`: per-rank pre-sharded load, `TPBackendNCCL` all_reduce on attn/MoE partial sums, vocab-split embed (mask + all_reduce), head all_gather, greedy decode.
- M1 fix (commit `51102429`): the fused bszN MoE path produced zero routed output. Root cause in `BC_BlockSparseMLP::run_bszN`: the folded routed+shared launch writes `fold_out` (= shared `out_d_sh` scratch), but Python reads `experts_cfg.out_bszn`, which was never written. Fix copies the H-wide trim into `out_bszn`. Also `fold_prepare()` now reattaches `k2_g/u/d` (mixed-K) on every rebuild.

## M1 result (MAX_NEW=16 unless noted, bsz 1, greedy)
- Prompt 0 "The capital of France is" -> "**Paris**." Ranks identical: `[671, 6102, 294, 8760, 344, 2619, 51119, 42499]`.
- Prompt 1 "# Fibonacci function in Python" -> coherent code-doc text. Ranks identical.
- Prompt 2 "List three primary colors..." -> "The three primary colors in traditional painting (specifically in the subtractive color model". Ranks identical.
- Reference: `<artifacts>/tp2logs/bench_native_tsplit.log` (native tensor-split, coherent).

## Baseline (eager, no CUDA graphs, NCCL)
- Prompt 2, 16 tokens: decode 38.02 ms/token, first token 1291 ms.
- Prompt 0, 8 tokens (clean run): decode 42.77 / 43.91 ms/token (rank0/rank1), first token ~1120 ms.
- Target >= 150 tok/s = <= 6.67 ms/token.

## Blocker history (resolved)
- `import exllamav3` JIT-builds `exllamav3_ext`. Always `export TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions` (private per container).
- Do not run two launches at once on the same GPUs: concurrent runs OOM.

## M2 results (COMM=p2p; NCCL token output is the reference)
- P2P collectives (`TPBackendP2P`, `P2PComm` from `exllamav3/tp2/p2p.py`): prompt 0 8 tok identical to NCCL; prompts 1 and 2 16 tok identical across ranks (prompt 2 matches the NCCL prefix; full NCCL list for prompt 2 not yet captured).
- Decode ms/tok, eager: P2P 30.07/30.61 (prompt 0), 30.45/30.48 (prompt 1), 28.62/28.61 (prompt 2). NCCL eager 42.77/43.91 (prompt 0).
- P2P + `EXL3_WHOLE_GRAPH=1`: token-identical to eager P2P on all 3 prompts. Decode ms/tok: prompt 1 24.05/24.04, prompt 2 20.81/20.81, prompt 0 32.26/32.27 (8-token run).
- Long-prefill fix: arena limit is 256 KiB, so `all_reduce` chunks to 65536 fp32 per call (sizes depend only on numel, identical on both ranks).
- Head all_gather under P2P: flat gather + last-dim concat, token-identical on all prompts.

## Decode profile, P2P + `EXL3_WHOLE_GRAPH=1` (prompt 2, 16 tok, rank 0, `PROFILE=1`)
Includes prefill. Self CUDA 454 ms, Self CPU 340 ms (16 steps; ~28 ms/step with profiler overhead).
| kernel | CUDA ms | share | calls | avg µs |
|---|---|---|---|---|
| `p2p_oneshot_kernel<0>` (all-reduce) | 240.5 | 53.0% | 1456 | 165 |
| `exl3_moe_kernel<0,128,2,16>` | 47.6 | 10.5% | 39 | 1221 |
| `exl3_gemv_int8_sq<6,1,false>` | 27.2 | 6.0% | 1936 | 14 |
| `exl3_moe_coop_a_mixed_k` | 26.3 | 5.8% | 640 | 41 |
| `exl3_moe_coop_b_mixed_k` | 16.9 | 3.7% | 640 | 26 |
| `exl3_mgemm_kernel` | 16.1 | 3.6% | 731 | 22 |
| `p2p_oneshot_kernel<1>` (head gather) | 1.8 | 0.4% | 16 | 115 |
[INFERENCE] The all-reduce kernel's 165 µs average is spin-wait on the peer, not transfer: transfer for a 20 KiB payload measured ~5 µs. Ranks are CPU-launch-bound (eager attention core, CPU 340 ms vs CUDA 454 ms), so each collective absorbs rank skew. Attention-core capture is not yet done (see GraphCapture.md); that is the next lever, after which skew should drop. Not yet verified.

## Profiler fix
`PROFILE=1` previously ran the extra decode on rank 0 only, so rank 0 issued collectives rank 1 never matched (`p2p_oneshot: timeout`). It now runs on both ranks; only rank 0 prints the table.

 ## Next steps (M2 step 3, then M3)
1. Verify the skew hypothesis: time the all-reduce with both ranks pre-synchronized, then with rank-skew injected.
2. Capture the attention core in graphs (or cut its CPU launch overhead), then re-profile.
3. Target >= 150 tok/s (<= 6.67 ms/tok); current best 20.8 ms/tok (P2P + graph, prompt 2).

## Decode-only profile, P2P + EXL3_WHOLE_GRAPH=1 (prompt 2, rank 0, 15 decode steps, PROFILE=1)
Profiler starts after the first token (prefill excluded). Under profiler: wall 34.2 ms/step (overhead inflated).
Unprofiled reference wall for the same config: 21.4 to 22.4 ms/tok (s0_base run, 128 tok, 3 prompts).
GPU busy 209.3 ms / 15 steps = 13.95 ms/step. Launches: 1255 kernels/step; cudaLaunch-class runtime calls 1071/step.

| category | ms/step (profiled) | share of GPU busy | main kernels |
|---|---|---|---|
| collective (P2P all-reduce + gather) | 4.79 | 34% | `p2p_oneshot_kernel<0>` 81 calls/step, avg 58 us (spin-wait on peer, see below) |
| GEMV/GEMM (dense int8/EXL3) | 4.23 | 30% | `exl3_gemv_int8_sq<6,1,false>` 1.70, `<6,1,true>` 0.69, `exl3_mgemm` 0.94, `exl3_gemm<8>` 0.23, `gemvx` 0.40, aten::mm 0.42 |
| MoE | 2.70 | 19% | `exl3_moe_coop_a_mixed_k` 1.65, `_b_mixed_k` 1.06 (40 calls each) |
| HC (mHC mix/apply/finalize) | 0.76 | 5% | `hc_mix_finalize` 0.33, `hc_mix_partials` 0.28, `hc_apply` 0.15 |
| attn core + misc (dsa_attn, rope, router) | 0.72 | 5% | `_dsa_attn_split` 0.31, `routing_gemv` 0.25, `_dsa_attn_combine` 0.19, rope ~0.1 |
| copy / elementwise / index | 0.26 | 2% | DtoD memcpy, aten::copy_ |
| other (norms, topk, misc) | 0.49 | 3% | `rms_norm` 0.32, `routing_ds3_topk` 0.13 |
| idle / host-bound (wall - GPU busy, unprofiled) | ~7.5 to 8.5 | n/a | eager attention core + per-step host bookkeeping |

Per-step kernel counts: `p2p_oneshot` 81 (all-reduce 80 + gather 1), `exl3_gemv_int8` 161, MoE 80, HC 240, rms_norm 122, dsa_attn 40+40.
[INFERENCE] Under the profiler the GPU is busy ~14 ms of a ~22 ms step, so the GPU is idle about 35% of the wall: the step is host-launch-bound, and the all-reduce time is mostly peer skew absorbed while the faster rank waits.

## DSpark tap saturation and draft-position acceptance (measured)
- Taps (raw fp32, before the fp16 cast): layers 36-38 reach ~271k at sink tokens 0 and 2 on prefill; 8-9 values per layer are >=65504; no NaN/inf. Decode steps are small (amax 10-50).
- `EXL3_DSPARK_TAP_SCALE=1/64` removes saturation (post-scale max 4236). Clean A/B (K=5, 128 tok, COMM=p2p, WHOLE_GRAPH=1): AL 3.09/3.30/2.58 at scale 1 vs 2.84/3.49/2.68 at 1/64. Neutral, so the default stays 1.0.
- Per-position acceptance, DSpark K=5, prompts 1+2 timed (77 rounds, 148 accepted drafts; prompt 0 excluded because its histogram includes an unlogged warm pass): P(a>=j) = 0.857, 0.494, 0.286, 0.182, 0.104 for j=1..5. Conditional P(draft j accepted | draft j-1 accepted) = 0.857, 0.576, 0.579, 0.636, 0.571. Round outcomes a=0..5: 11, 28, 16, 8, 6, 8. The first draft is NOT the weak point; the drop comes at draft 2 and then stays flat near 0.6.
