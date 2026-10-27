# TP=2 DeepSeek-V4.1 EXL3 — master plan (ordered by expected gain)

Current: plain decode **13.15–13.18 ms/tok (76 tok/s)**, DSpark K=5 **6.85–9.39 ms/tok (118/135/101)**,
prefill **3.6–3.7k tok/s @2048**, **4.3–4.4k @4096**. Targets: 300 tok/s decode, 8000 prefill.
Reference on identical hardware (diffbot, 2×RTX PRO 6000 TP2, EXL3 3bpw): **156.9 tok/s @2K**, 172.8 @46K,
362.9 aggregate at 4 streams, **11.2k prefill @40K**, fp8 KV, DSpark k=5.

| # | Change | Source | Expected | Notes |
|---|---|---|---|---|
| 1 | **Capture the decode attention core** (kill ~490 kernel + 40 graph launches/step ≈ 4–5 ms/step) | our profile | 13.15 → ~9 ms/tok plain; DSpark → ~175 | the `q_head_norm` gate contradiction (b1_prep vs _batch_ok_b1) meant capture was never actually exercised; fix in progress |
| 2 | **Verify-width MoE occupancy** — 255 regs/thread → 1 block/SM, 16.7% warps, 8.4% DRAM | our ncu | verify round 22 → ~15 ms → DSpark ~190–200 | escape the register ceiling: split the 64-row accumulator across block-groups, or int8 decode-once (weights reused by 6 rows → 6× amortization) |
| 3 | **Continuous batching / multi-stream decode** (`max_batch_size>1`) | **owner's call (raised in review 2026-10-27)**; corroborated by diffbot 157→363 tok/s @4 streams, sybil 8 slots/713 tok/s aggregate, 1Cat `max-num-seqs 6` | up to ~3× *throughput* | biggest untapped lever; MoE weights are read once per expert-run and shared across rows, so per-token cost falls with rows. Our runner is hardcoded bsz 1. Caveat: diffbot needed "batch-invariant dense GEMMs" — naive batching perturbs per-stream tokens, so this needs a batch-invariance gate, not just a speed run |
| 4 | **Prefill MoE occupancy (replaces int8 decode-once)** | code inspection: our prefill kernel ALREADY reuses each decoded weight fragment across up to 64 rows, so int8 decode-once (diffbot's +6%) has little left to give. The real problem is 255 regs/thread -> 1 block/SM -> ~12.5% occupancy at **8.4% of DRAM** | target +20-40% prefill | with DRAM at 8.4% we can afford extra weight reads: raise threads/block with half the accumulators each and broadcast the fragment through smem. Rejected already: int8 (needs activation quantisation), K-tile 16, `FUSED_ROWS_WIDE` lower |
| 5 | **8-bit prefill all-reduce** | diffbot: AR 16K 4.04 → 1.79 ms (−11% 4-session wall) | +10% prefill (our AR = 22%) | fp32 accumulate, low-bit wire; KL gate |
| 6 | **FP8 KV — quality is fine, capture is the blocker (CLOSED as negative)** | committed `0e685633` in tp2/fp8kv: KV16 11.42/11.42/11.33, KV8-eager 15.02/14.69/14.52, experimental KV8-captured 11.15/11.20/11.15 but **numerically wrong** (first divergence at generated token 11-12; agrees with KV16 on only 12/128 positions vs 21/128 for eager KV8). Quality on the eager path: common-prefix KL 0.0081/0.0003/0.0065/0.0017/0.0013 nats over prompts 0-4, top-1 20/20 | 0% today | pool 3200 -> 2150 bytes/token/rank (**-33%**; 60 GiB/rank pool-only budget ~= 30M tokens, capped by the 1M max_position). **Prerequisite: a packed-pool-aware captured attention.** Root cause found in code: `_batch_ok_b1` gated capture on `not kl.quant`, so packed pools silently fell back to eager (+30% decode). Default stays `KV_BITS=16` |
| 7 | **Replay trim in multi-chunk prefill** (late layers only need the last rows) | diffbot 3.7–4.6% at 34K–130K | +4% prefill | needs prefix-cache/ring semantics check |
| 8 | **Prefix caching** (warm-prefix replay) | diffbot agentic 224 tok/s; sybil 16K prompt 16 s → 1 s on hit | large real-world gain | unmeasured by us |
| 9 | **Speculative policy: always speculate; K is a small dial** | matched COMM=p2p + graphs, prompt 0: one DFlash block forward per round at every K; estimated round cost 19.7/20.7/21.3/22.1 ms at K=2/3/4/5, AL 2.25/2.56/2.51/2.67, decode 8.77/8.09/8.50/8.27 ms/tok. Six-prompt matched sweep: K=5 beat plain on all six; auto on/off lost to K=5 on all six. K=2/5 event stream means: draft 1.87/1.81 ms, verify 21.47/21.09 ms, host iteration 26.13/26.06 ms (iterations include startup and non-SPEC rounds). | K choice = small | `DSPARK_POLICY=always` by default. Round cost is approximately flat in K and MoE-read bound. Earlier 'K<5 dominated' and 'prose loses' claims, based on unmatched configurations, are retracted. |
| 10 | **bsz-1 decode MoE occupancy** — 60–63 regs, 27 KB smem, 2–3 blocks/SM, 33–43% warps | our ncu | maybe +10–20% on the MoE share | lower value than #2 because 33–43% is already usable |
| 11 | **Larger prefill chunk (8K/16K)** | our 2048→4096 = +18% | unknown | cheap test; memory-limited |
### Prefill chunk: settled (2 reps each, sampled device peak)
| prompt | chunk | TTFT ms | tok/s | peak (smi) | headroom |
|---|---|---|---|---|---|
| 32768 | 8192 | 6858 / 6909 | 4778 / 4743 | 93.9 GiB | **3997 MiB** (ship) |
| 32768 | 16384 | 6777 / 6875 | 4835 / 4766 | 97.1 GiB | 751 MiB (not shippable) |
| 32768 | 32768 | OOM (36 MiB free) | — | 97.2 GiB | 663 MiB |
| 2048 | 2048 | 573.9 / 573.9 | 3569 / 3569 | 90.9 GiB | 7021 MiB |
| 2048 | 4096 | 573.8 / 570.1 | 3569 / 3592 | 94.7 GiB | 3222 MiB |
**`PREFILL_CHUNK` default = 8192.** 16384 buys +1.2% for 3.2 GiB, so it is not worth shipping. Remaining gap to the
8000 tok/s target is the MoE (30.6% of prefill) — see the tile/occupancy item.

### Decode MoE: what the profiler actually says (and what is ruled out)
A kernel at 32768-token-free decode (layer 3, uniform random routing, 50 warm iters): **A 34.93 us at 42.2% DRAM, B 20.47 us
at 32.5% DRAM; 80 regs/thread, 22.5 KB smem, 3 blocks/SM register-limited, 43.8% occupancy, long-scoreboard 44%.**
So both are **latency-bound**: plenty of DRAM headroom, warps waiting on loads.
- **Wave quantisation ruled out**: A has 504 blocks (2.68 waves of 188 SMs), but `ksplit=3` (1512 blocks, 8.04 waves) is
  **21.5% SLOWER** (A 42.43 us vs 34.93) because the split cost beats the removed partial wave. k=2 also worse (38.13).
  Default stays ksplit=1.
- **Local-memory traffic is REAL and is the top lead** (a mid-investigation retraction was itself wrong — the raw counters
  are authoritative): A **1,290,240 local load sectors vs 1,317,141 global (~49%)**, B **765,250 vs 1,188,536 (~57%)**.
  "Zero LDL in SASS" only means the compiler emitted **generic addressing** (`LD.E`), not that the traffic is absent.
  A C array indexed by a runtime value is guaranteed local; prime suspects are the codebook lookup, per-expert pointer
  tables and per-K-tile bit-offset arrays. Fix by moving those tables to `__constant__` (uniform lane access) or
  `__shared__` (divergent), or replacing the lookup with arithmetic/`__shfl_sync`/a constant-folded `switch`.
- **Two-chain ILP** was already rejected (+4.7 us). Remaining candidates: more warps per block on B (1120 blocks, 5.96
  waves, 512 threads/block), and prefetch depth/width per warp.

### Round anatomy, fully attributed (DSpark K=5, CUDA events + K sweep)
`round ~22 ms = target forward 11.0 + verify ~6.0 + draft 1.9 + host ~3.1`. Round cost is flat in K
(19.7 / 20.7 / 21.3 / 22.1 ms for K=2/3/4/5), so the verify rows are a minor term: going 6 rows -> 3 rows saves 1.4 ms.
**Every remaining decode lever therefore reduces to the single-token target forward = the bsz-1 MoE kernels.**
ncu on those: A 43.4 us at 42.2% DRAM, B 28.2 us at 32.5% DRAM, 80 regs/thread, 3 blocks/SM (register-limited),
43.8% occupancy, long-scoreboard 44% -> latency-bound, not bandwidth-bound. Levers in order: occupancy (force fewer
regs / more warps), memory-level parallelism, then layout.

### New data (2026-10-27, multi-stream measured)
- **Batching works once the B>1 core is captured**: plain per-stream 14.62 ms/tok at B2 (**126 tok/s aggregate**) and
  19.51 ms/tok at B4 (**189 tok/s aggregate**), i.e. ~2.1x wall-clock throughput at 1.8x per-stream latency. Peak VRAM
  86.6 GiB (B4 plain) — batching costs almost nothing in memory at these widths.
- **Spec + batching is a substitution, not a product**: DS K=5 at B4 gives 96.8 tok/s aggregate, *below* B1 spec and far
  below B4 plain. Reason: MoE cost is linear in distinct expert slots, and 4 streams x 6 verify rows = 24 rows ->
  up to 144 slot reads, so batching adds no MoE amortisation while spec triples the row count. Ship policy: spec for
  single/low-concurrency latency, plain batching for aggregate throughput.
- **Blocker before shipping batching**: two identical prompt streams at B4 diverge from each other at position 11, so the
  captured B>1 path is not yet numerics-safe (same class as the earlier B4-vs-B1 divergence). Also fixed: the graph
  capture failure was `torch.bincount` (a CPU sync) inside the captured MLP — replaced with a CUDA scatter-add count.

### New data (2026-10-27, later)
- **Prefill chunk size matters a lot**: 2048 → 3666 tok/s, 4096 → 4324, **16384 → 5056 (+17% over 4096)**.
  Larger chunks amortize weight reads. Testing 32768/65536 next. This is now a high-priority prefill dial.
- **8-bit (e4m3) prefill all-reduce**: +6.5% at 2048, +6.4% at 4096, +6.9% at 16384 — but a plain global cast
  FAILED the quality gate (prompt-3 token flip at step 11, KL 0.22 at the flip). Trying block-scaled fp8
  (per-128 scale) next; if it still flips, drop the item.

| 13 | ~~Cheapen the draft loop~~ **retracted** — the DFlash block path already drafts the whole block in ONE forward (`generator.py:1374`, `iterate_draftmodel_dflash_gen`); the per-position loop at 1258 is the MTP path, unused here | counted forwards per round: 1 at every K | — | the reducible cost is the block forward itself and the verify, both dominated by MoE weight reads |
| 12 | **K=7 drafter** | diffbot: "of the +43%, draft length gives nearly all" | +30–40% on code | BLOCKED: our checkpoint has `dspark_block_size=5` |

Already done / no longer candidates: mixed-K fused MoE with per-expert K tables; piecewise + block-boundary-merged CUDA
graphs; P2P one-shot all-reduce (4.9 µs vs NCCL 11.8); hybrid transport by payload size; single-chunk prefill;
fused HC partials (bit-exact); DSpark width-keyed graphs; allocation-churn fix; DSpark tap scaling (neutral here).

Measured-and-rejected (with numbers in the per-workstream notes): cp.async smem ring (slower at bsz1), two-chain ILP
(+4.7 µs), PF=8, split-K, fused A→B single launch (slower at every bsz), K-tile 16 for the 64-row prefill MoE (255 regs),
min-blocks=2 (spills), lowering the fused row threshold, fp32→bf16 prefill wire (0.6%), x-fan fusion (−3% but changes
tokens from position 55), event-based sampler sync, index-math hoisting (<0.15 ms/tok ceiling).
