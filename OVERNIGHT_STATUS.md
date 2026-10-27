# Overnight work log — TP=2 300 tok/s push

Target (owner): **~300 tok/s** with DSpark working well on 2× RTX PRO 6000 Max-Q (P7 GPU0+GPU1).
Constraint: TP=2 recipe only. Recipe A (GPU0 + GX1) untouched. GPU2/GX2 untouched.

## Where we started tonight
| Config | Result |
|---|---|
| Non-spec eager (first coherent run, NCCL) | ~43 ms/tok |
| + P2P one-shot all-reduce (4.9 µs/call) | ~29–30 ms/tok |
| + piecewise CUDA graphs (seq 1) | ~21 ms/tok |
| + replay copy batching / capture cleanup / regression fix | **13.50 / 13.71 / 13.58 ms/tok** (~73 tok/s) |
| **DSpark K=5** (70–79% acceptance, 2.5–3.8 tok/round), width-keyed graphs | **83 / 149 / 102 tok/s** (p0/p1/p2) |

Bandwidth floor per GPU: 1-row token ~4.8 GB → 2.9 ms; 6-row verify round ~7.5 ms → with 3.3 tok/round ~440 tok/s theoretical.

## Verify-round profile (rank0, one seq=6 forward, 22.0 ms CUDA / 13.1 ms CPU)
- MoE coop A+B at 36 slots: **9.2 ms (42%)**
- TP all-reduce (80 collectives): **3.8 ms (17%)** — mostly rank-skew spin
- Dense GEMV/GEMM: ~4.2 ms
- Host/launch gap + misc: ~4.8 ms

## Workstreams
1. **HostSlim** (main branch `/home/treadall/recipe-2xrtx-tp2`) — capture the eager attention core into the patchable-graph mechanism (graph.cu Graph::launch + GP_ params), variant-keyed for the m=2 new-pool-entry parity; eager fallback for topk-regime switch / ring shift. Removes ~8 ms/step host gap + most spin.
2. **VerifyGraph** (dspark branch `/home/treadall/recipe-2xrtx-tp2-dspark`) — verify-width graphs (done, 7e3cbdc6), verify-step profile, attack the 6-row MoE.
3. **MoE 6-row** (moe worktree `/home/treadall/recipe-2xrtx-tp2-moe`) — structural MoE work for bsz 6 (persistent launch / deeper pipeline), microbench only (<4 GiB).

## Log (append as milestones land)
- 52a89689 main: default the experimental batched B=1 attention path OFF (was a +2.3 ms/tok regression, not token-exact).
- 7e3cbdc6 dspark: width-keyed piecewise graphs for seq 1..8 (verify steps now graphed): eff 91.6/129.5/74.1 → 102.3/149.2/83.0 tok/s.

## New goals from owner (overnight)
- **300 tok/s decode** (with DSpark working well), **>= 8000 tok/s prefill**.
- Prefill math: a chunk reads the weights once, so tok/s ≈ chunk_len / (weight_read + compute). 4.8 GB/GPU at ~1.3 TB/s ≈ 3.7 ms → 2048-token chunk ceiling ~10-20k tok/s; a 16-token prompt caps at ~2000 tok/s. Measured prefill today: ~110-200 tok/s.
- Workstream 4 added: **PrefillPerf** (chunk policy, MoE prefill path, attention split kernels, all-reduce payload at 2048 rows = 40 MB per collective × 80 = 3.2 GB/chunk).

## Progress log (continued)
- **52a89689** main: experimental batched B=1 attention path defaulted OFF (it was a +2.3 ms/tok regression).
- **7e3cbdc6 / d05639b6 / 32d20817** dspark: verify-width (seq 1..8) piecewise graphs; per-width profile; P2P spin/timing stats.
  - DSpark K=5 measured: acceptance 0.70–0.79, 2.5–3.8 tokens/round, **eff 83 / 149 / 102 tok/s** (p0/p1/p2).
  - All-reduce in-situ latency ~20 µs/call (microbench 4.9 µs) — it is queue-depth/skew, not one-sided spin (spin counts 62–80 vs 15–18 µs wait).
- **e7025c67 / 9158b147 / 57a8a780** tp2/moe-kernel: per-stage tile geometry at bsz>1 (verify MoE kernel sum 315.6 → 282.6 µs, −10%); scratch layout + A/B bodies as device functions + fused single-launch path behind `EXL3_MOE_COOP_SINGLE` (measured SLOWER at every bsz: 83.5/137.5/384.5/507.2 vs 59.7/96.8/307.1/399.8 — kept default off as evidence). pytest 263 green.

## Prefill measurements (first pass, chunk 4096 unless noted)
| chunk | len 256 | 1024 | 2048 | 4096 |
|---|---|---|---|---|
| 512 | 1239 | 1693 | 1970 | 2104 |
| 2048 | 1222 | 2034 | 2925 | 3320 |
| 4096 | — | 2034 | 2925 | **3660** |

Fit: TTFT ≈ 290 ms + 0.20 ms/token. Profile of a 2048-token prefill (703 ms): MoE prefill ~236 ms (≈360 GB/s effective, **4× off** the ~58 ms byte floor), NCCL all-reduce 135 ms (≈50 GB/s = PCIe hardware limit), dsa_attn 37 ms, CUTLASS GEMMs + unaccounted ~300 ms.
→ 8000 tok/s at 2048 needs ~256 ms total: MoE must go 236 → ~60 ms and the all-reduce must be overlapped with compute.
