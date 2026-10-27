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

## Validation methodology (important, learned tonight)
Greedy 128-token identity is a **very** strict gate: any fp-order change flips argmax after ~50 tokens.
Measured: EXL3_DSV41_XFAN=1 (fused x-side projection fan, 6–8 GEMV launches → 1–2 per layer) gives
13.16/13.21/13.25 ms/tok vs 13.50/13.71/13.58 baseline (~3% faster) but changes tokens from position
55/97/68 onward (73/31/59 of 128 differ), with both ranks agreeing. So:
- **A real bug shows up in the first ~10 tokens**; late divergence (>40) is usually fp chaos, not a defect.
- Decide optimizations on coherence + speed + early-token agreement, not 128-token identity.
- x-fan stays OFF by default (3% is not worth a documented quality change); available via EXL3_DSV41_XFAN=1.

## Prefill findings
- 2048-token prompt is served as TWO chunks (1792 + 255 rows) — see generator/job.py:1503-1513: prefill stops at the
  final recurrent-checkpoint page boundary (256-token pages) when a recurrent cache is present. This doubles weight
  traffic per prompt; removing it (while preserving recurrent-checkpoint semantics) is the biggest prefill lever.
- MoE prefill: 62% of assignments on the 1792-row chunk take the **batched reconstruct** path because of the fused
  row cap (`EXL3_MOE_FUSED_ROWS_WIDE=256`); raising the cap should move them to the fused kernel.
- HC at prefill: the HC fold is disabled under TP2 and above 32 rows (hyperconnections.py: `"backend" not in params`
  and `b*s <= 32`), so every layer runs 3 unfused HC launches (712 total, 54 ms). A tiled prefill HC mix is the lever.
- Transport: prefill prefers NCCL (703 ms) over P2P (759 ms) at 40 MB payloads; decode prefers P2P (4.9 µs vs 11.8 µs
  at 20 KB). Hybrid selection by payload size is worth ~56 ms at prefill.

## More results
- **DSpark K sweep** (graph, WARM=1, 3 prompts): K=5 is best — 0.8125/0.7881/0.8272 acceptance,
  4.14/3.82/2.60 tokens/round, **150.1/154.2/99.9 tok/s**. K=4: 142.8/152.5/94.9. K=3: 131/136/95.
  K>5 is capped by `dspark_block_size=5` in the checkpoint, so only 3/4/5 are meaningful.
- **Hybrid transport** (commit 05eab78d): P2P under 64 KB, NCCL above. Prefill 2048 TTFT 698.9 ms (was 703 NCCL,
  759 P2P). Decode unchanged. Enabled via COMM=hybrid.
- **Prefill MoE row-cap sweep**: raising `EXL3_MOE_FUSED_ROWS_WIDE` moves assignments from the batched-reconstruct
  path to the fused kernel but is NET SLOWER (512 → 735 ms, 1024 → 808 ms vs 701 baseline). Keep 256.
- **HC at prefill**: fold is inactive under TP2 and above 32 rows; 712 launches / 54 ms. Tiled prefill HC is the lever.

## FINAL STATE (evening→morning)
Branch `recipe/2x-rtxpro6000-tp2` (worktree /home/treadall/recipe-2xrtx-tp2) = everything merged:
MoE per-stage tiles, DSpark (K=5) with verify-width graphs, single-chunk prefill, hybrid transport,
allocation-churn fix, docs. All other worktrees clean; nothing uncommitted.

### Decode (bsz 1, greedy, 128 tokens, both ranks identical)
| config | ms/tok | tok/s |
|---|---|---|
| plain, COMM=p2p, piecewise graphs | **13.50 / 13.71 / 13.58** | 74 / 73 / 74 |
| plain, COMM=hybrid | 13.96 / 13.93 / 13.87 | 72 |
| **DSpark K=5, COMM=p2p, width graphs** | **8.30 / 6.98 / 9.53** | **116–132** (best single run 154) |
| DSpark K=5, MoE wide-A override | 7.45 / 7.22 / 10.18 | 129 / 128 / 93 |
Acceptance 0.64–0.83 (varies per prompt and run ±0.05), 2.5–3.8 tokens per verify round.
K sweep: K=5 best; K=4 ≈ −5%; K=3 −12%; K>5 capped by `dspark_block_size=5`.

### Prefill (TTFT, single prompt, both ranks agree)
| length | chunk | TTFT | tok/s |
|---|---|---|---|
| 2048 | 2048 (single chunk, new default) | 580.6 ms | **3527** |
| 2048 | 512 | 366 ms | — |
| 4096 | 4096 (single chunk) | 989.2 ms | **4141** |
| 4096 | split (EXL3_PREFILL_RC_SPLIT=1) | 1106.6 ms | 3702 |

### Gap to the owner's targets
- decode 300 tok/s: at 116–132 (best 154). Needs ~2.3×; bottleneck is GPU work in the 6-row verify forward
  (MoE 42%, all-reduce 17%, dense 19%) plus ~4-5 ms/step of host overhead that only attention capture removes.
- prefill 8000 tok/s: at 3527–4141. Needs ~2×; kernel split at 2048: MoE 30.6%, all-reduce 22.3%,
  dense GEMM 21.3%, other 12.7%, HC 8.5%, MoE-recon 4.5%. GPU is 93% busy, so it is kernel work, not host.

### Highest-value remaining work (in priority order)
1. **Capture the decode attention core into a graph.** It is the only way to remove the remaining
   ~4–5 ms/step of host overhead: 80 `cudaGraphLaunch` + ~490 `cudaLaunchKernel` per step
   (`cudaLaunchKernel` 23.6% and `cudaGraphLaunch` 12.9% of self CPU). Design: route the eager core's
   `ext.*` calls through the engine's patchable `Graph` (graph.cu, `record_param`/`GP_*`), capture per layer
   with the position scalars patchable, and select variants by host parity (the m=2 new-pool-entry branch
   alternates with position parity; single ENTRY variant for m=1 layers). Eager fallback for the top-k
   regime switch (ec > 512) and ring shift/rebase steps. Blocked previously only by effort, not by design.
2. **MoE at the verify width (36 slots)**: 282 µs/layer per GPU, ~688 GB/s vs 1.65 TB/s peak. Split-K, tile
   geometry, register/cp.async rings and the fused single-launch all measured and rejected; what is left is
   a deeper restructure (warp-specialized producer/consumer, TMA-style bulk copies) — needs ncu counters,
   which now work in the moe container (`--cap-add SYS_ADMIN`, real binary at
   /opt/nvidia/nsight-compute/2026.1.1/ncu; also copied into exl3-tp2-prefill).
3. **Prefill MoE 30.6%**: same kernel family; the 64-row instance is 88 ms of the 168 ms.
4. **HC 8.5% at prefill**: the fold is inactive under TP and above 32 rows; a tiled prefill HC mix exists
   (hc_mix_tiled.cu) and is unused for this path.
5. **Overlap the prefill all-reduce** (22.3%): needs a 2-way row split with two streams (no independent
   work exists inside a layer, so a plain async issue has no window). Upper bound ~68 ms at 2048.
