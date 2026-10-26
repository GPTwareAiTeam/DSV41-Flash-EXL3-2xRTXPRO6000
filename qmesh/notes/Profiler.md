# Profiler: P7-local decode budget (GPU0, bsz 1, null remote)

Setup: container `exl3-dsv41` (GPU0 only). Checkpoint `/models/dsv41/exl3-serve-final` (2.8535 mix, P7 hot 4,100 experts @4.0 bpw), placement `qmesh/placement.final.json`. Remote experts replaced by `NullRemoteExpertHost` (zeros, no network, no GX1), with the CPU MoE worker spawn/commit stubbed in the harness only. Scripts: `/home/treadall/src/exllamav3-dsv41/qmesh/prof/`. Run: `docker exec exl3-dsv41 bash -lc 'cd /src/exllamav3 && PYTHONPATH=/src/exllamav3 EXL3_MOE_SPLIT_FUSED=0 python3 -u qmesh/prof/<script>.py'`.

Note: `EXL3_MOE_SPLIT_FUSED=0` must be set in the environment (read at import). With the fused path the null host raises `NotImplementedError`.

## Headline (measured, proven)
| metric | value | script |
|---|---|---|
| Generator decode, warm, 62 steps | **18.6–19.2 ms/token** (mean 18.6 with BC_DSA=1; 19.2 with BC_DSA=0 + BATCH_GRAPH=0) | `time_gen.py` |
| Generator step, synced, 16 steps | mean 18.83 ms; model.forward 18.64 ms; generator overhead 0.19 ms | `breakdown.py` [A] |
| Kernels/step | ~1,260–1,290 (no graph launches in trace: 19,872 cudaLaunchKernel + 704 cooperative over 16 steps) | `prof_gen.py`, `breakdown.py` |
| GPU kernel-busy sum/step | 12.5–12.7 ms | `prof_gen.py`, `attribute.py` |
| Load | 5.1 s, VRAM 76.21 GiB | `load.py` |

Conclusion: the local P7 path alone is ~18.6–19 ms/token. Today's ~100 ms is therefore ~80 ms TCP/remote, consistent with (a). Local path alone is ~2.8x over the 6.67 ms target, consistent with (b). Target <4 ms P7-local is not reachable with the current structure.

## Component table (ms/token, 40 layers, GPU; kernels/token)
Kernel-busy attribution from trace (`attribute.py`, 16 steps; kernels inside a range by correlation id):
| component | GPU ms/token | kernels/token | notes |
|---|---|---|---|
| hot MoE (`exl3_moe_coop_a/b`, `gemv_int8_sq`, `mgemm`, routing, topk, moe elementwise) | ~5.4 (attributed to `moe` range) | 560 | `moe_coop_a<4,2>` 2.22, `coop_a<6,2>` 0.94, `coop_b<4,2>` 0.83, `coop_b<6,2>` 0.59, routing 0.26+0.13 |
| attention (incl. projection GEMVs, rope, DSA split/combine, norms) | ~5.0 (range `attn`) | 416 | `gemv_int8_sq` 2.16 + mgemm 1.22 + gemv_int8_sq 1.11 ≈ QKV/O projections; rope 0.22; dsa 0.48 (from prof_gen) |
| HyperConnection (mix + apply, 80 each) | 0.31 + 0.31 + 0.07 + 0.07 ≈ 0.76 | 240 | `hc_mix_finalize` 0.34, `hc_mix_partials` 0.28, `hc_apply` 0.13 (prof_gen totals) |
| Engram (layers 1, 14) | 0.46 | 32 | 0.23 per layer |
| lm_head | 0.44 | 1 | `exl3_gemm_kernel<8>` 0.43 |
| block residual / misc elementwise | ~0.25 + ~0.5 unattributed | ~90 | |
| **kernel-busy total** | **~12.5–12.7** | ~1,260–1,290 | |

Event-timed operator table (`op_time.py`, no profiler, sync per op; includes launch gaps — use for relative shares only):
- attention projections q_a 0.94, q_b 1.38, wkv 0.62, wo_b 1.24 (sum 4.18 ms)
- hot MoE total 7.00 (per layer 175 µs)
- HC mix 1.75+0.49, apply 0.36+0.12 (2.72 ms)
- Engram 1.27 (2 layers, 632 µs each); norms ~1.04; lm_head 0.45; embed 0.07

Wall gap: kernel-busy 12.5–12.7 ms vs wall 18.6–19.2 ms → ~6 ms/token of launch/idle gap in the eager path (CPU enqueue of ~1,290 kernels/token). This is the dominant non-kernel cost and is removable only by graph capture of the whole step.

## Local MoE selection
Local (GPU-resident) expert picks: 2.45 per layer per step = 98.1 per token (of 240 picks = 40 layers × top-6), i.e. ~41% of routed picks are local under the final placement. Not all picks are local; the remaining ~142/token are remote (null here, TCP in production).

## Graphs
- `EXL3_BC_DSA` (default 1) and `EXL3_DSV4_BATCH_GRAPH` (default 1) are on; trace contains no `cudaGraphLaunch`. Disabling both moves 18.6 → 19.3 ms (within noise, ~0.7 ms). Whole-step graph is NOT engaged in the Generator decode path. Per-module graph slots exist but do not cover the step.

## Verdict on the dense-path ceiling
- Local path today: ~18.6–19.2 ms/token. Components above 1 ms/token each: MoE hot (~5.4 kernel / 7.0 event), attention (~5.0 kernel / 14.2 event incl. projections), HC (~0.8 kernel / 2.7 event), Engram (0.46 kernel).
- Kernel-only floor (perfect launch overlap, current kernels): ~12.5 ms → still ~1.9x over 6.67 ms, and ~3.1x over 4 ms.
- Therefore the 6.67 ms target is NOT reachable with current kernels even with perfect graph capture; the dense/attention and MoE kernels themselves must get faster.

## Items needed for <4 ms P7-local (in order of expected payoff; unverified estimates)
1. Whole-step CUDA graph capture for decode (removes ~6 ms launch gap; needs static buffers, fixed shape at bsz 1). Largest single item.
2. MoE kernel efficiency at bsz 1: `moe_coop_a<4,2>` 2.2 ms/step over 40 layers = 56 µs/layer for the 4-bit expert set; fuse `coop_a`+`coop_b`, remove routing topk/gemv launches (0.4 ms) by fusing into one kernel.
3. Attention projections: four separate GEMVs (q_a, q_b, wkv, wo_b) at 15–35 µs each; fuse q_a+wkv and fold wo_b, or use mgemm for the set.
4. HC: fuse sinkhorn 20 iters into the mix kernel (`hc_mix_finalize` 344 µs + `hc_mix_partials` 283 µs over 80 calls) — 2 kernels × 80 per token today.
5. Engram: 0.23 ms/layer GPU for 2 layers; fuse gather.
6. Remove per-step aten churn (empty/slice/copy/zeros ~5 ms CPU under profiler; the launch gap).

## Not verified
- nsys: `/opt/cuda/bin/nsys` is a wrapper pointing to a missing `/usr/local/cuda-13.0/bin/nsys`; not used.
- Chrome trace from `prof_gen.py` failed (path); `breakdown.py` trace used instead and then deleted to save space.
- Per-layer GPU time for attention vs. MoE split by layer not computed beyond the table above.
