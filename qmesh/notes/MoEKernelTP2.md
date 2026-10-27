# MoE kernel, TP=2 shard shape (2x RTX PRO 6000 Max-Q, sm_120)

Branch `tp2/moe-kernel`. Layer 3, mixed-K. Device: 188 SMs.

## Committed
| commit | change |
|---|---|
| 6f6998bf | 256-thread blocks (WK=8) |
| 760ec369 | shared expert folded into the fused coop launch (index E); width-sized fold scratch; MINB=3 |

## Measured bsz1 (device time per kernel)
| config | A (us) | B (us) | total (us) |
|---|---|---|---|
| WK=8, pre-fold (6f6998bf) | 34.8 | 20.1 | 54.9 (+ separate shared launch, 88.7 end-to-end) |
| folded, MINB=3 (760ec369) | 34.55 | 19.89 | 54.44 |

Folded graph total measured earlier: 59.2 us. Gap to kernel sum: ~4.8 us.
Target: total bsz1 <= 45 us. **Not met (54.44 us).**

Fold correctness: `tests/kernels/moe/test_moe_fold.py`, bsz 1/2/6/8, folded vs separate err 0.000e+00.
Full `tests/kernels/moe/`: 263 passed (259 baseline + 4 fold cases).

## Rejected (measured)
- Forced `a_global` for all bsz: worse.
- WNT=4: incorrect at bsz >= 2.
- WK=4: A worse.
- PF=8 for all widths: A 43.3, B 23.0. PF=8 only for bits=2: no change, reverted.
- Two-chain ILP inside gemv_tile (per-warp k split into two independent fp16/fp32 accumulator chains, interleaved decode/MMA, fixed-order fold): A 39.01 / B 24.42 vs HEAD A 34.23 / B 19.62 (bsz1, prof_moe_tp2, reproduced). Reverted.
- B-WIDE at bsz1: B 22.4 (worse than 19.9).
- A ksplit forced 2/3/4 (EXL3_MOE_COOP_KSPLIT): A 38.3/42.0/44.4, B 26.1/30.4/36.9. Every split slows both. Default ksplit=1 kept.
- cp.async smem trellis ring (per-warp, RING_NS slots of 128 words, PF=RING_NS-1; sm_75 sync fallback). Same-harness bsz1 / bsz6 (A/B us): HEAD 34.16/19.25, 216.75/97.05. NS=8: 49.43/23.03, 324.71/110.51. NS=4: 38.83/19.65, 206.98/98.68. Worse at bsz1; rejected. Patch kept at `recipe-artifacts/2x-rtx-tp2/ring_ns4.patch` (not committed). Cause: per-slice wait_group + __syncwarp on critical path; occupancy not limited (NS=8: A 4, B 5 blocks/SM).

## Diagnostics
- Kernel B reduction costs ~1.1 us (DBG=2 skip: 18.8 vs 19.9).
- Kernel A epilogue costs ~1.8 us (DBG=1 skip: 32.8 vs 34.6).
- Real layer-3 K histogram: g/u/d K=2 x148, 2.5 x109, 3.0 x60, 3.5 x67, shared 6.0 x1. Mixed (70.8 MB) and uniform-3 (79.6 MB) take equal bsz1 time, so bsz1 is latency/phase-bound, not bandwidth-bound.
- Stream probe: 10.32 MB graph-replayed = 0.63 TB/s; the 1.65 TB/s ceiling is not reachable for one ~10 MB launch.

## Idea 6: merge A and B into one launch (IMPLEMENTED, default off, LOST)

Implemented behind `EXL3_MOE_COOP_SINGLE=1` (mixed-K only; uniform-K stays on two launches). Commit `9158b147`.
- A/B bodies are `__device__` functions (`moe_a_body`/`moe_b_body`); the two-launch kernels are thin wrappers, so the default path is behavior-neutral.
- Publish: A's last arriver sets `ready_a[s*(I/128)+chunk]` after `store_h4` with `__threadfence`. B spins on every (slot, chunk) of its run; a stuck wait traps after 20 s.
- Queue: one persistent grid, `min(resident, nA+nB)` CTAs, `atomicAdd(queue_head)` claims A items before B items. `resident` from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SMs.
- Counters (`ctr_a`, `ctr_b`, `ready_a`, `queue_head`) are memset on the launch stream before each fused call. Rotation stays a separate launch.
- Scratch layout: `ctr` tensor is `ctr_a | ctr_b | runs | ready_a | queue_head | exit_ctr`, sized `2*slots*(I/128) + bsz*(Ho/128) + 2*slots + 5`. Production (`block_sparse_mlp.py`), tests and bench use the same size.

Measured (folded graph, µs, layer 3; two-launch / single-launch):

| bsz | two-launch | single-launch |
|---|---|---|
| 1 | 59.71 | 83.53 |
| 2 | 96.75 | 137.51 |
| 6 | 307.07 | 384.47 |
| 8 | 399.83 | 507.22 |

Two-launch matches baseline e7025c67 (59.9 / 97.3 / 309.6 / 403.8). Routed and folded rel err ~1e-3 on both paths.

Cause of the loss: not isolated. Not profiled per kernel on the fused path.

Decision: keep default `EXL3_MOE_COOP_SINGLE=0`. The flag stays as negative-evidence code. Do not chase further without a new design.

## Next
- Router GEMV/top-k overlap with the previous layer tail (runner side).
