# v2 backlog

Ranked by expected measured win for the owner's workload (decode target 300 tok/s single user, prefill 8,000 tok/s,
~5M shared KV tokens, 8 users). v1.2 single-user decode is ~130–160 tok/s; a verify round costs ~22 ms and yields
~3.3 tokens, so 300 tok/s needs about twice the tokens per round, half the round cost, or both.

## 1. Per-matrix bit widths inside experts (w1 / w2 / w3) — main model first, drafter second

Question: do gate (w1), up (w3) and down (w2) of a routed expert deserve different bit widths? The 2.499 checkpoint
sets one rate per expert (1,400 @ 4, 4,860 @ 3, 9,100 @ 2 bpw); the drafter's routed experts are hot 4 / cold 2.

Facts (2026-10-31): each expert matrix is 5120 × 2304 (hidden × moe_intermediate), 1.41 MiB per bpw unsharded;
w1/w3 trellis `[320, 144, 16·bpw]`, w2 `[144, 320, 16·bpw]`; TP2 halves w1/w3 on the output side and w2 on the input
side (the 72 in rank dirs). Per expert: 8.44 / 12.66 / 16.88 / 33.75 MiB at 2 / 3 / 4 / 8 bpw. The runtime requires
w1 and w3 of an expert to share a rate (load-time assert in `BlockSparseMLP`); w2 has its own rate table in the
mixed-K kernels, so a w2-only change needs no kernel work. Mixed rates keep the shared-expert fold.

Bounded plan (stop after each step with a table):
1. Per-matrix sensitivity on ~8 frequently routed experts per sampled layer: capture real expert inputs (a few
   hundred tokens), quantize w1/w3 and w2 independently at 2/3/4 bpw, measure expert-output error against BF16 with
   the other matrices held at 4 bpw. No overlay, no serving. Needs a capture + error harness (not written).
2. Model KL (main model: panel/WikiText; drafter: drafter-logit KL vs 4-bit) for 3–4 w1w3/w2 combos at equal total
   bytes, e.g. 3/4, 2/3, 4/3 vs uniform. Needs a per-suffix K map in the quantizer.
3. Acceptance / panel score only if step 2 shows a clear winner.
Expected for the drafter alone: small (uniform 3 → 4 bpw moved tokens/round ~1%; 2-bit cold experts cost nothing
measurable). For the main model it can redistribute bytes from the 9,100 2-bit experts' least sensitive matrix.

## 2. Retrain the drafter on the 2.499 checkpoint

The DSpark drafter was trained on the original weights; on 2.499 it accepts 3.2 tokens/round on code vs 4.6 on math.
Quantizing the drafter harder or softer moves acceptance ≤2% (docs/tp2/RESULTS.md section 11), so the drafter's
training target, not its precision, is the limit. A head-only adapter failed (+0.5% with the Markov correction).
Days of work; needs a training-capable copy of the drafter blocks.

## 3. Prefill

Cold prefill is ~2.6–2.8k tok/s vs the 8k target; untouched in v1.2.

## 4. Per-content draft width

Fixed K=3 vs K=5 swings decode 7–14% by content (K=3 better on code/prose/multilingual, K=5 on math/structured).
A context-aware gate regressed on single prompts because width changes alter near-tied greedy choices; any new gate
needs many-prompt evidence (the 40 held-out prompts with a paired CI), not single 24k prompts.

Shipped today: the on-GPU confidence gate (`EXL3_DSPARK_GATE=auto`), which picks the verified prefix per round
from the drafter's confidence and adapts its cutoff to the content. On 24k prompts it is within 1–9% of the best
fixed width (code: auto 139 vs K=3 152 tok/s; math: 176 vs K=5 181; prose: 113 vs K=3 115).

Written, not shipped: a host-side expected-value width chooser (`kselect.py`, 2026-10-29: per-pass acceptance per
position × measured round cost, candidates K∈{3,4,5}, hysteresis, a small probe budget, rank-identical from integer
draft stats). Its GPU evaluation was never run, and the patch is kept outside this tree. Start here: run it against
the auto gate on the 40 held-out prompts with a paired CI.

## 5. MoE kernel

Coop down projection runs at 61% of DRAM peak, gate/up 68%, long-scoreboard stalls ~46%, tensor pipe 19–23%
(Nsight, 6 rows). A cheaper bit-identical trellis decode is the remaining lever (estimate 0–5%, may regress).
Diffbot-style cp.async streaming was bit-identical but slower (B 90.7 → 104.5 µs/layer; opt-in
`EXL3_MOE_COOP_STREAM=1` on branch tp2/moeflat-1031, not merged); swapped-MMA has weak support from tensor-pipe use.

## Measurement notes for whoever picks these up

- Warm the GEMM tune cache per new bitrate before timing: a cold 8-bit shape autotunes mid-decode (one ~1.5 s round,
  false 15–20% loss). `EXL3_ROUND_PROF=1` now prints slow rounds and their worst segment.
- Single 24k prompts compare different continuations once the drafter or width changes; decide on the 40 held-out
  prompts (`dtrain/prompts_held.json`) with a paired bootstrap CI. The first run of a session can be slow on a few
  prompts with identical tokens; repeat it.
