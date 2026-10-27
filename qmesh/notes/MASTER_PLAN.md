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
| 4 | **Prefill MoE int8 decode-once** (decode tiles once → int8 → int8 GEMM) | diffbot +6% on a faster base; our MoE = 30.6% of prefill at 8.4% DRAM | +40–70% prefill | also fixes the occupancy problem structurally |
| 5 | **8-bit prefill all-reduce** | diffbot: AR 16K 4.04 → 1.79 ms (−11% 4-session wall) | +10% prefill (our AR = 22%) | fp32 accumulate, low-bit wire; KL gate |
| 6 | **FP8 KV cache** (DECIDED — do it) | diffbot ships fp8 KV; our engine has `k_bits=8` packed pool | ~2× KV capacity; halves KV bytes read/written | validate: attention must not get slower (packed pool adds dequant ALU) and tokens/KL within noise. Removes the earlier bogus "we're already at parity" claim |
| 7 | **Replay trim in multi-chunk prefill** (late layers only need the last rows) | diffbot 3.7–4.6% at 34K–130K | +4% prefill | needs prefix-cache/ring semantics check |
| 8 | **Prefix caching** (warm-prefix replay) | diffbot agentic 224 tok/s; sybil 16K prompt 16 s → 1 s on hit | large real-world gain | unmeasured by us |
| 9 | **Content-aware adaptive K** (prose K→3, code K→5; our cap is 5) | diffbot: k=7 vs k=3 = +33% code, −5% chat, −17% 4-stream chat | +5–15% depending on content | engine already has `dynamic_draft_tokens` + confidence calibration; it is NOT engaged for DSpark today (in progress) |
| 10 | **bsz-1 decode MoE occupancy** — 60–63 regs, 27 KB smem, 2–3 blocks/SM, 33–43% warps | our ncu | maybe +10–20% on the MoE share | lower value than #2 because 33–43% is already usable |
| 11 | **Larger prefill chunk (8K/16K)** | our 2048→4096 = +18% | unknown | cheap test; memory-limited |
| 12 | **K=7 drafter** | diffbot: "of the +43%, draft length gives nearly all" | +30–40% on code | BLOCKED: our checkpoint has `dspark_block_size=5` |

Already done / no longer candidates: mixed-K fused MoE with per-expert K tables; piecewise + block-boundary-merged CUDA
graphs; P2P one-shot all-reduce (4.9 µs vs NCCL 11.8); hybrid transport by payload size; single-chunk prefill;
fused HC partials (bit-exact); DSpark width-keyed graphs; allocation-churn fix; DSpark tap scaling (neutral here).

Measured-and-rejected (with numbers in the per-workstream notes): cp.async smem ring (slower at bsz1), two-chain ILP
(+4.7 µs), PF=8, split-K, fused A→B single launch (slower at every bsz), K-tile 16 for the 64-row prefill MoE (255 regs),
min-blocks=2 (spills), lowering the fused row threshold, fp32→bf16 prefill wire (0.6%), x-fan fusion (−3% but changes
tokens from position 55), event-based sampler sync, index-math hoisting (<0.15 ms/tok ceiling).
