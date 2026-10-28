# TP=2 DeepSeek-V4.1 EXL3 — master plan (status board, 2026-10-27)

**Current, all measured on this card, both ranks agreeing:**
plain decode **10.80 / 10.91 / 10.85 ms/tok (92 tok/s)** · DSpark K=5 **7.86 / 6.83 / 9.01 / 4.65 ms/tok**
(111-215 tok/s, AL 3.30/3.30/2.58/5.29) · multi-stream plain **B4 = 189 tok/s aggregate** (branch, not shippable) ·
prefill **4778 tok/s** (32K prompt, chunk 8192).
Targets: **300 tok/s decode**, **8000 tok/s prefill**.
Reference, identical hardware (diffbot, 2x RTX PRO 6000 TP2, EXL3 3bpw): 156.9 tok/s @2K, 362.9 aggregate @4 streams,
11.2k prefill @40K, fp8 KV.

### Long-context: what helps and what hurts (measured 2026-10-27)
- **Speculative decoding HURTS at long context**: at 24k tokens, DSpark K=5 gives **19.89 ms/tok vs 16.58 plain**
  (acceptance 0.33, AL 1.71). The drafter is cheap at long context but the verify rows are not, and the whole path is
  eager there. Do not enable spec for long-context serving until capture works past 1K; re-measure then.
- **Prefix caching is the big lever for agent loops**: 20k full hit **126 ms vs 6.9 s cold (55x)**; 60k **154 ms vs 21 s (~130x)**;
  two alternating 60k contexts both hit at 155/154 ms; a 60k context growing ~1k per turn keeps **98.2-98.3% cached**
  (TTFT 496-529 ms, with one unexplained 5175 ms outlier at a checkpoint boundary).
- **Its cost is real**: the cache currently forces prefill chunk 1024, so cold runs are ~35% slower than the 8192-chunk
  baseline (20k cold 6851-6904 ms vs 5081 ms at chunk 8192 for 24k tokens). Fixing that decoupling is the prerequisite
  for making it default-on.
- **Memory per long context online**: ~1.98 GB of *system RAM* for recurrent checkpoints (59 of them) plus **183 MiB of
  GPU KV** for a 60k context; the 4 GiB `EXL3_PREFIX_CACHE_CAPACITY` is the host checkpoint budget, not GPU memory.
- **The auto-chunker does not reserve the drafter**: with `DSPARK=1` at 24k the default chunk 8192 OOMs (needs ~640 MiB
  more), so long-context spec runs need `MAX_CHUNK=2048` today.

## LONG-CONTEXT BASELINE (the owner's real workload: 20k-100k prompts)
Measured in the runner (end-to-end, `COMM=hybrid`, chunk 8192, graphs on, `EXL3_DSV4_B1_BATCH=1`):
| prompt tokens | TTFT | prefill tok/s | decode ms/tok |
|---|---|---|---|
| 24167 (cold, no warmup) | 5081 ms | **4756** | **16.58** |
| 60417 | 13068 ms | **4624** | **16.88** |
| ~24k with `WARM=1` | 1843 ms | (artefact: native page reuse from the warm pass) | 16.50 |
Short-context decode for comparison: 10.77-10.84 ms/tok.
**Two hard limits for this workload:**
1. **The captured decode path stops at ~1024 positions**: `_batch_ok_b1` requires the compressed pool in its dense regime
   (`ec = (pos+1)/m <= index_topk = 512`, m=2), and the captured body implements no top-k indexer selection. Beyond that
   every decode step falls back to eager. That plus KV reads is the 10.8 -> 16.6 ms/tok penalty.
2. **The same gate breaks batching for long prompts**: a batch containing ANY row past that limit drops the whole batch to
   eager (measured 81.6 vs 31.7 ms/token at B8). So all the batching numbers above are short-prompt only.
Prefill is ~4.6-4.8k tok/s at these lengths, i.e. a 100k prompt costs ~21 s TTFT cold; prefix caching is the lever that matters
(`EXL3_PREFIX_CACHE`, merged into main at `8ec9c9c8`, default off, verified 16x at 8k).

## 0. MEASUREMENT RULES LEARNED THE HARD WAY (read before trusting any number)

1. **`COMM` depends on the workload.** Measured back-to-back on main, prefill 2048-token prompt / chunk 2048:
   **`COMM=nccl` 3399 tok/s vs `COMM=p2p` 3034 tok/s (+12% for NCCL)**. The historical 4777.7 tok/s prefill figure was
   taken with `COMM=nccl`. Our one-shot P2P kernel wins on *decode* (10 KB payloads, 4.93 vs 11.84 us per call) and
   loses on *prefill* (84 MB payloads chunked through a 256 KiB arena: 204,881 launches at 16 us each = **40.7% of
   prefill CUDA time**). So "always p2p" is wrong; **hybrid** is the intended default, and the arena size is a
   separate prefill lever under investigation.
2. **These cards are power-capped**: ~2100 MHz against a 3090 MHz ceiling under sustained load, GPU1 flagged
   `SW power cap`. Never compare against a number from earlier in the session; A/B back-to-back and record
   `clocks.sm / power / throttle mask` with every result.
3. **Capture gates silently disable the fast path.** Two separate incidents cost us 2x each: `b1_prep` vs
   `_batch_ok_b1` disagreeing on `q_head_norm`, and a merge that restored the `B==1` restriction so `BATCH>1`
   fell back to eager. Whenever `BATCH>1`, verify `EXL3_DSV4_B1_BATCH=1` **and** that replay counters are non-zero.

## A. SHIPPED AND VERIFIED (on `recipe/2x-rtxpro6000-tp2`)
| 8 | **Prefix caching** (`EXL3_PREFIX_CACHE`) -- commit `036f32da` | **8192-token prompt TTFT 3178 -> 196 ms (16x)**, 2048 -> 936 -> 195 ms (4.8x); partial hits ~1.9x (2048: 921 -> 534, 8192: 3178 -> 1655); effective throughput 2578 -> 41,732 tok/s on a full 8K hit | all 8 full/partial comparisons token-identical on BOTH ranks; KV cost 3200 B/token/GPU from a preallocated pool (no extra allocation); reuse stops at the last page-aligned recurrent checkpoint, so `k*256+1` prompt lengths maximise hits |

| # | change | measured gain | verification |
|---|---|---|---|
| 3 | **Multi-stream decode (continuous batching)** -- owner's call. `BATCH` dial through Model.load/Cache/Generator, per-width warmup, graph-safe expert counting (no `torch.bincount` sync), per-row block table, and the real bug: `_indexer_keys` built RoPE positions as `arange(n) + e0`, so at B>1 the rows rotated the compressed pool at positions 4,5,6,7 instead of 4,4,4,4 | **B1 10.84 ms/tok (92 tok/s) -> B2 13.91 (133.8 aggregate) -> B4 17.89 (202.99 aggregate)** = **2.2x throughput**, peak 86.5 GiB/GPU | four IDENTICAL B4 prompts give byte-identical 128-token streams with graphs on; layer-2 attention row diff 0 at pos 9 with pool_c/pool_r/ring byte-equal; B4-captured vs eager logits max KL 0.016-0.024, mean 0.0008-0.0015 (fp-order only) |
| 1 | **Capture the decode attention core** (piecewise segments + cross-block graph merge) | plain **13.15 -> 10.97 ms/tok** | tokens byte-identical, 4 prompts x 128 tokens x 2 ranks. Root cause of every earlier failure: `b1_prep` required `q_head_norm=True` while `_batch_ok_b1` required `False`, so the capture never ran |
| 2 | **Kill the local-memory load in `gemv_tile`** (the `r0_ok ? A2[...] : hzero` conditional took the address of a per-thread const) | MoE **67.31 -> 65.76 us/call @bsz1 (1.024x)**, 1.018x @bsz6 | bitwise identical; ncu had shown A at 1,290,240 local load sectors vs 1,317,141 global (~49%), B 765,250 vs 1,188,536 (~57%) |
| 3b | **Prefetch hoist (C1) + PDL launch of kernel B (C5)** from the external review kit | **62.54 us/call (1.076x)** @bsz1, 1.052x @bsz6; in-model 10.97/11.02/10.98 -> 10.80/10.91/10.85 | bitwise identical; mixed-K header regenerates byte-identically from the patched base header |
| 5 | **`COMM=hybrid` for prefill (the real win); arena is a P2P-only lever** | prefill 32768/chunk 8192: p2p+256 KiB 4137.2 -> **hybrid 4819.1 (+16.5%)**, verified 4704.9 on main. Arena size affects **P2P mode only**: 256 KiB 4137.2 -> 2 MiB 4357.4 -> 4 MiB 4369.0 -> 8 MiB 4358.9; launches 204,881 -> 25,681 -> 12,881 -> 6,481 with AR self-CUDA flattening at ~2.51-2.54 s from 2 MiB (transfer-bound). **Hybrid is arena-insensitive** — proven by order reversal: 4 MiB first 4808.8 then 256 KiB 4800.4, versus 256 KiB first 4819.1 then 4 MiB 4751.0, so drift dominated and there is no hybrid gain. Compiled default kept at 4 MiB for P2P-mode runs; 2 MiB would capture the same with half the memory | decode unchanged: byte-identical tokens at 256 KiB vs 4 MiB, 10.84/10.86/10.86 vs 10.88/10.91/10.84; B4 ~205 aggregate |

| 11 | **Prefill chunk and auto sizing** -- `MAX_CHUNK` 8192 (4778 tok/s @32K prompt); `MAX_CHUNK=auto` sizes the chunk from the measured peak table against the `CTX_TOKENS` KV budget, so 1M context costs one chunk step (8192 -> 4096) | prefill 3.9k -> 4.78k | 2 reps per size, sampled device peak |
| 9 | **Speculative policy: always speculate**; one DFlash block forward per round at every K; round cost ~flat in K | K choice worth <=8% | matched config; `DSPARK_POLICY=always` |

**IMPORTANT:** batching requires `EXL3_DSV4_B1_BATCH=1`. Without it, B>1 silently falls back to the eager batched
body and B4 measures 39 ms/token instead of 17.89. This is the *second* time a capture gate has cost us a 2x.


| # | change | measured gain | verification |
|---|---|---|---|
| 1 | **Capture the decode attention core** (piecewise segments + cross-block graph merge) | plain **13.15 -> 10.97 ms/tok** | tokens byte-identical, 4 prompts x 128 tokens x 2 ranks. Root cause of every earlier failure: `b1_prep` required `q_head_norm=True` while `_batch_ok_b1` required `False`, so the capture never ran and every "capture is slower" reading was the eager fallback |
| 2 | **Kill the local-memory load in `gemv_tile`** — the `r0_ok ? A2[...] : hzero` conditional took the address of a per-thread const, so the false branch became a local-memory load; replaced with a uint32 reinterpret plus a literal zero | MoE **67.31 -> 65.76 us/call @bsz1 (1.024x)**, 197.34 -> 193.88 @bsz6 (1.018x) | bitwise identical. ncu had shown A at **1,290,240 local load sectors vs 1,317,141 global (~49%)**, B 765,250 vs 1,188,536 (~57%), line-mapped to `cuda_fp16.hpp:344` |
| 3 | **Hoist the trellis prefetch ring ahead of the prologue (C1) + PDL-launch kernel B (C5)**, from the external review kit | **62.54 us/call @bsz1 (1.076x)**, 187.56 @bsz6 (1.052x); C1 alone 1.054x, C5 adds ~2% | bitwise identical; in-model **10.97/11.02/10.98 -> 10.80/10.91/10.85 ms/tok**, tokens byte-identical to the pre-patch baseline |
| 11 | **Prefill chunk and auto sizing**: `MAX_CHUNK` 8192 (4778 tok/s @32K prompt, 4.0 GiB headroom); `MAX_CHUNK=auto` sizes the chunk from the measured peak table against the `CTX_TOKENS` KV budget, so 1M context costs one chunk step (8192 -> 4096) rather than a collapse | prefill **3.9k -> 4.78k** | 2 reps per size, sampled device peak |
| 9 | **Speculative policy: always speculate.** One DFlash block forward per round at every K; round cost ~flat in K (19.7/20.7/21.3/22.1 ms for K=2/3/4/5); K=5 beat plain on all six bench prompts | K choice worth <=8% | matched config; `DSPARK_POLICY=always` default. Two earlier claims of mine ("K<5 dominated", "prose loses") came from unmatched configs and are retracted |

| **Whole-step graph / host gap (item 1b) — CLOSED, ~0.3% on main, 2.4% at B4; merged as opt-in `EXL3_PW_TPCOLLECT` (default off, commit `d774e65b`)** | main verification: 10.75/10.81/10.79 (on) vs 10.77/10.84/10.83 (off) ms/tok; earlier capturing the TP collective and merging block pairs (`EXL3_PW_TPCOLLECT`) cuts launches 43 -> 29, `cudaGraphLaunch` CPU 2.03 -> 1.78 ms, `_foreach_copy_` 2.15 -> 1.31 ms, and the instrumented inter-replay gap 3.36 -> 0.72 ms, yet decode only moves 11.45 -> 11.30 ms/tok (~1%), tokens byte-identical. The single-stream step is GPU-bound; the "3 ms host gap" was mostly profiler instrumentation. A whole-step graph is not worth the refactor |
| **Narrower tile (`MOE_COOP_WNT=1`) — impossible** | the 2-bpw path derives its load count from WNT/2 and the decoder packs two 16-column tiles per 32-lane load, so a 16-column tile needs a new load/lane layout, not a macro override |
| **More warps per block (`WK=16`, 512 threads) — LOSES** | back-to-back: B1 63.19 -> 81.00 us/call (+28%, A 39.45 -> 45.90, B 26.97 -> 38.57), B6 200.21 -> 234.81 (+17%). Outputs also differ by 4.6e-4 relative (different K-split reduction). With wide tiles also 29% slower, the 32-column / 8-warp / 256-thread geometry is locally optimal |
| **Expert grouping at B>1 — nothing to win at B4, real headroom from B8** | with distinct prompts: B4 = 24 slots / 22.48 distinct (94%, no sharing), B6 = 36 / 32.19 (89%), **B8 = 48 slots / 31.76 distinct -> 34% of slots share an expert with another slot**. So grouping saves ~nothing at B4, but from B8 up a shared read could serve 2-3 rows. Note the dispatch split: B<=8 uses the coop path which executes every row-expert slot separately, while above B8 dispatch can use the sorted, expert-grouped fused `exl3_moe` path — which may explain why B16+ scales better. **Best next experiment: B8 coop vs the grouped path, matched diverse prompts, uninstrumented timing** |
| **Expert-grouped decode path at B<=8 — LOSES** | `EXL3_MOE_GROUPED_DECODE=1` vs the coop default, short distinct prompts, two reps back-to-back: **B6 coop 247.5 / 246.4 tok/s aggregate vs grouped 138.7 / 138.7**; **B8 coop 213.3 / 216.6 vs grouped 141.3 / 139.3**. Tokens are not identical either (first divergence pos 43 at B6, pos 11 at B8). The coop path wins despite the 34% slot duplication at B8 — grouping costs more in sorting/scatter than it saves in reads. Default unchanged (opt-in) |
| **Mixed-context batches lose capture entirely — 2.6x cliff** | `_batch_ok_b1` requires EVERY slot's position below the compressed `index_topk` regime, and the captured body implements no top-k selection. A B8 batch including the 1134-token prompt measured **81.6 / 83.1 ms/token (coop) vs 31.7 with all-short prompts**, for both paths. This is why every fast B16+ number used short prompts. Supporting mixed rows needs per-row capture eligibility, not a dial |
## B. MEASURED AND REJECTED (do not retry without new information)

| change | result |
|---|---|
| **8-bit prefill all-reduce** | +6.4-6.9% prefill but **fails the coherence gate** (prompt-3 token flip at step 11, KL 0.22). Block-128 scaling fixes the tokens yet sits outside the logit-noise floor (max raw logit delta 1.4-2.0) |
| **FP8 KV cache** | quality is fine (KL 0.008/0.0003/0.0065/0.0017/0.0013, top-1 20/20) and the pool drops **3200 -> 2150 bytes/token/rank (-33%)**, but packed pools lose the capture (`_batch_ok_b1` gates on `not kl.quant`) and the bypassed capture is numerically wrong (diverges at generated token ~12; agrees 12/128 vs 21/128 for eager). Default `KV_BITS=16` |
| **Wide tile for kernel B at bsz1** (`EXL3_MOE_COOP_WIDE_B=1`) | **worse**: 67.29 vs 65.66 us/call, B 27.87 vs 25.38. Side note: the launcher comment claiming "bsz 1 keeps B wide" is wrong — `moe_coop_pick_wide(72, 6)` is false on Blackwell |
| **Wide tile for kernel A at bsz1** (`EXL3_MOE_COOP_WIDE_A=1`, never tried before) | **worse**: 72.75 vs 65.66 us/call, A 51.65 vs 39.92 (+29%); bsz6 also worse (207.64 vs 193.88). Together with the wide-B result this closes the "wider per-warp span" family at bsz1 — the tiling knobs are exhausted and only latency hiding is left |
| **ksplit 2 / 3 on kernel A** (wave alignment: 1008 = 5.36 waves, 1512 = 8.04) | **worse**: k=3 A 42.43 us vs k=1 34.93 (-21.5%), k=2 38.13. Wave quantisation is not the limiter |
| **Issue-bound hypothesis (H1)** | refuted: issue_active **45.8% A / 42.7% B** against DRAM 47.8% / 33.3% — neither engine saturated. It is **memory latency/granularity** (44% long-scoreboard), so the PRMT byte-window / drop-staging-for-6bpw branch is the wrong direction |
| **int8 decode-once for prefill MoE** | our kernel already reuses each decoded weight fragment across up to 64 rows, so diffbot's +6% lever is largely spent; the defect is occupancy/tile choice |
| **Verify-width MoE "255 registers"** | false premise: at rows=6 it is **80 regs, 3 blocks/SM, 46%/39% DRAM**, and per token verify costs 48 us against 55 us for a 1-row step — verify is already efficient |
| **Adaptive K / auto spec on-off** | `auto` lost to fixed K=5 on 3 of 6 prompts at matched flags; content-dependent K is worth <=8% |
| also rejected earlier | cp.async smem trellis ring, two-chain ILP (+4.7 us), PF=8, split-K on A, fused A->B single launch, K-tile 16 at 64 rows, `__launch_bounds__` min-blocks=2 (spills, -10.8%), lowering `EXL3_MOE_FUSED_ROWS_WIDE`, batched B=1 attention path (+2.3 ms/tok), fp32->bf16 prefill wire, event-based sampler sync, index-math hoisting |

### Decode MoE: the parallelism scaling law (new, 2026-10-27)
Measured with the standalone bench at bsz1, same build:
| tile | cols | A blocks | A us | call us |
|---|---|---|---|---|
| narrow (default) | 32 | 504 | **39.92** | **65.66** |
| wide `WIDE_A=1` | 128 | 252 | 51.65 | 72.75 (+11%) |
| wide `WIDE_B=1` | 128 | - | B 27.87 | 67.29 (+2.5%) |
Wide reads 128 B per row (a full sector, ideal) yet is **29% slower on A**. So the kernel is NOT sector/coalescing
limited -- it is limited by **parallelism**: wide cuts the block count 2x and occupancy falls with it, while ncu reports
3 blocks/SM resident against only 2.68 blocks/SM *available*. That inverts the usual advice and says the next experiment
is a **narrower** tile (COLS 32 -> 16, 1008 blocks, and no reduction because the split is over output columns, unlike the
ksplit variants that lost). Needs a rebuild and must keep 263 kernel tests green; the 8 bpw staging assumes COLS=32 so
check `tile_wn`/`STAGE_WORDS` assumptions first.

## C. LEFT TO DO — reordered for the owner's real workload (20k-100k prompts)

| # | change | why | evidence / next step |
|---|---|---|---|
| 16 | **Captured decode past 1K tokens** | decode at 24k-60k is **16.58-16.88 ms/tok vs 10.77-10.84 short**; the whole gap is the eager fallback because the captured body has no top-k indexer selection | `_batch_ok_b1` requires the dense regime (`ec <= index_topk=512`, m=2, so pos <= 1023). Implement device-side top-k selection in the captured body, or make the gate per-row. IN PROGRESS |
| 17 | **Prefix caching at 20k-100k** | the owner's agent loops resend a long context every turn; verified 16x at 8k but unmeasured at their sizes | measure cold vs full vs partial hit at 20k/60k, and the growing-suffix case (turn N -> N+1); report the hit ratio and the memory a 60k entry pins (~195 MB of pages). IN PROGRESS |
| 18 | **Batching for long prompts** | blocked by the same gate: a batch with any long row drops to eager (81.6 vs 31.7 ms/token at B8), so all batching numbers are short-prompt only | follows from #16; then re-measure B4/B8 at 20k+ prompts |
| 4 | **Prefill rate** | 4624-4756 tok/s at 24k-60k, i.e. ~21 s TTFT for 100k cold | AR is transfer-bound (194 us/call, 34% of prefill CUDA) so only faster transfer helps; the remaining levers are chunk size at long lengths and #17 |
| 19 | **MoE per-slot cost** | geometry exhausted (wide -29%, narrow unbuildable, WK=16 -28%, grouped -44%) | needs a new kernel design, not a parameter |
| 12 | **K=7 drafter** | +30-40% on code per the reference | BLOCKED on weights (non-causal 5-slot block) |
| 6 | **FP8 KV** | -33% pool, quality fine | blocked on packed-pool capture correctness |
| 5 | **8-bit prefill AR** | +6.5% prefill | fails the coherence gate |

Batching is DONE and shipped; these are what remains.

| # | change | why it is worth it | evidence / next step |
|---|---|---|---|
| 14 | **Batch scaling** — per-stream 10.84 / 13.91 / 17.89 / 31.67 / 49.56 / 68.71 / 81.70 ms and decode-only aggregate **92 / 133.8 / 203.0 / 231.5 / 266.6 / 349.3 / 391.7 tok/s** at B1/2/4/8/16/24/32 (peak VRAM 86.3-87.1 GiB) | aggregate keeps climbing past the `MAX_BSZN=8` fused cap, but each doubling now buys only ~+15% while per-user latency degrades sharply (B16 = 20 tok/s per user). So the throughput/latency frontier is: B2-B4 for interactive, B8-B16 for batch |  cause: MoE cost is linear in *distinct expert slots*, and rows do not amortise it (8 rows x top-6 = 48 slot reads vs 24 at B4). Next: (a) confirm where the aggregate curve actually turns (B32?), (b) reduce per-slot cost (the `MOE_COOP_WNT=1` experiment below), (c) check whether rows routed to the *same* expert are already grouped so a slot can serve several rows |
| 15 | **Dynamic admission for real concurrency** — our runner enqueues all slots at once and they finish together | real serving has sequences arriving/leaving at different times; a fixed BATCH means idle slots when fewer than B are active, and the batch width should track the number of live streams | needs the generator's admit/evict loop exercised with staggered job lengths; measure aggregate tok/s with 8 slots and 3/5/8 live streams |
| 1b | **Whole-step graph** — collective in-graph, 1-4 replays/step instead of 40 | 10.84 ms/tok with ~9 ms CUDA busy: 1.5-2 ms/step of host work, of which `cudaGraphLaunch` ~0.67 ms and `_foreach_copy_` into static buffers ~0.65 ms (PROFILE=1) | external plan steps 3-4; P2P is graph-capturable, so put `tp_collect` inside the segment and merge block pairs. ~10-12% |
| 1c | **Narrower MoE tile** (`MOE_COOP_WNT=1`, COLS 32->16 -> 1008 blocks, no reduction) | the kernel is **parallelism-limited, not coalescing-limited**: the wide tile reads perfect 128 B sectors and is 29% SLOWER (A 51.65 vs 39.92 us) because it halves the block count, and ncu shows 3 blocks/SM resident against only 2.68 available | branch `tp2/moe-wnt1` prepared (macro made overridable); needs one build + 263 tests. Attacks both B1 latency and the B8 wall |
| 4 | **Prefill** — 4778 tok/s at chunk 8192 against an 8000 target | MoE is 30.6% of prefill at 19.5% DRAM, 33% warps, 0.98 waves, 128 regs; the tile sweep (M16/32/48/64) is FLAT within 1.8%, so tile choice is not the lever | **first resolve the 15% discrepancy**: the sweep measured 4045-4070 tok/s under the lock against the earlier 4778 — needs one GPU-idle repeat. Then profile the non-MoE 70% (AR is 22%, never profiled at prefill). Note the arithmetic: making the MoE FREE gives 6883 tok/s, so 8000 needs the other 70% to improve ~15% |
| 3b | **Spec + batching do not compose — measured negative** | B2 K5: 30.31/24.79 ms per stream, 63.27 tok/s; B2 K3: 27.80/23.88 ms, 68.78 tok/s; B4 K2: 49.91/41.79/47.86/49.91 ms, 76.45 tok/s. No clock capture, so absolute values carry a thermal/power-state caveat. Historical plain B4: 203 tok/s; B1 K5: 111–215 tok/s. | B4 + DSpark K=3 hit an illegal memory access reported at the `dsa_attn` combine launch (`dsa_triton.py:1105`); CUDA errors are asynchronous, so the originating kernel is not established. Reproduction was not attempted, so intermittent versus deterministic is unknown; shape logging is inconclusive. Treat verify width S=4 at B=4 as unsafe until buffer sizes are instrumented. Full traceback: `/home/treadall/recipe-artifacts/2x-rtx-tp2/tp2logs/specgrid_1027_1793136692_b4k3_rank0.log`. |
| 8 | **Prefix caching** | unmeasured by us; diffbot's agentic 224 tok/s rests on it (sybil: 16K prompt 16 s -> 1 s on a hit) | implement warm-prefix replay, measure TTFT on a repeated 8K/32K prefix |
| 7 | **Replay trim in multi-chunk prefill** | diffbot 3.7-4.6% at 34K-130K | late layers only need the last rows; check prefix-cache/ring semantics first |
| 6 | **FP8 KV, retry** | -33% pool for -2% decode once a packed-aware captured attention exists | blocked on packed-pool capture correctness; the quality gate already passed |
| 12 | **K=7 drafter — TRIED, blocked by training, not code** | the block is `[seed] + (block_size-1)` noise tokens (`dspark.py:200`), so no weight has dim 5 — but `dspark.py:20` says block queries attend **NON-CAUSALLY**, so block length is baked into *training*. Measured prompt 0: block5/K5 AL 3.086 acc 0.716 7.86 ms/tok; block7/K5 AL 2.357 acc 0.606 14.76 (a longer block alone degrades the first 5 slots); block7/K7 AL 4.103 acc 0.738 9.37 (draft 1.87 -> 6.3-10.1 ms, ~quadratic in block) -> net -19% | needs a 7-slot-trained drafter pack |

### Appendix: prefill chunk measurements (2 reps each, sampled device peak)
| prompt | chunk | TTFT ms | tok/s | peak (smi) | headroom |
|---|---|---|---|---|---|
| 32768 | 8192 | 6858 / 6909 | 4778 / 4743 | 93.9 GiB | **3997 MiB** (ship) |
| 32768 | 16384 | 6777 / 6875 | 4835 / 4766 | 97.1 GiB | 751 MiB (not shippable) |
| 32768 | 32768 | OOM (36 MiB free) | — | 97.2 GiB | 663 MiB |
| 2048 | 2048 | 573.9 / 573.9 | 3569 / 3569 | 90.9 GiB | 7021 MiB |
| 2048 | 4096 | 573.8 / 570.1 | 3569 / 3592 | 94.7 GiB | 3222 MiB |
**`PREFILL_CHUNK` default = 8192.** 16384 buys +1.2% for 3.2 GiB, so it is not worth shipping. Remaining gap to the
8000 tok/s target is the MoE (30.6% of prefill) — see the tile/occupancy item.

### Prefill diagnostic: collective modes and rank-0 kernel profile

The 4777.7 / 4742.8 tok/s historical 32K/8192 baseline was **NCCL**, not P2P: `tp2logs/final32768_8192_rank0.log` records `COMM=nccl`, `CACHE_TOKENS=33536`. It is not a matched baseline for the P2P tile sweep or current P2P measurements (`CACHE_TOKENS=33280`); neither the collective nor the cache-size contribution has been isolated by a matched comparison. Do not attribute the difference to a regression or thermal state without that comparison. NCCL's earlier ~22% share refers to 2048-token prefill, not the present P2P workload.

Two idle-start launches per tree, each with two timed reps after warmup, `PROMPT_LENS=32768 CHUNKS=8192 COMM=p2p TORCH_CUDA_ARCH_LIST=12.0a`, unique rendezvous:

| Tree | launch | rank 0 rep 0 / rep 1 tok/s |
|---|---|---|
| clean main d0483826 | pd_main_20261027a | 3896.0 / 3862.3 |
| clean main d0483826 | pd_main_20261027b | 3900.4 / 3879.9 |
| modified prefill tree 10512231 | pd_idle_20261027a | 3890.8 / 3884.9 |
| modified prefill tree 10512231 | pd_idle_20261027b | 3825.4 / 3778.2 |

The trees are not a controlled source-only comparison: the prefill tree has uncommitted edits. An idle check before the first run showed GPUs 0/1 at 0% and 3/2 MiB. One active prefill snapshot showed GPU0 2100/3090 MHz, 77 C, 261 W, throttle mask 0; GPU1 2077/3090 MHz, 80 C, 272.85 W, mask 0x4. This snapshot does not establish a thermal cause. The added M48 dispatch is not selected by the default Python policy, which passes 16, 32, or 64 only.

Rank-0 torch profiler, clean main, 32K prompt in 8192-token chunks, P2P: unprofiled TTFT 8405.5 ms, profiled TTFT 8566.6 ms. A 32768-token single chunk OOMed before profiling: the MoE scratch requested 2.96 GiB with 1.65 GiB free. Profile below is the achievable 8192-chunk workload, not single-chunk. Percentages use total kernel-event time 7982.31 ms, not wall time; attribution of generic kernels to MoE can be ambiguous, so generic GEMM/index kernels are labelled rather than claimed non-MoE. DRAM utilization was not measured.

| Kernel | time ms | kernel-time % | calls |
|---|---:|---:|---:|
| P2P oneshot all-reduce | 3282.04 | 41.12 | 204881 |
| CUTLASS s16816 f16 256x128 | 600.10 | 7.52 | 1596 |
| DSA attention | 587.14 | 7.36 | 160 |
| HC apply | 313.73 | 3.93 | 392 |
| HC mix partials | 213.83 | 2.68 | 316 |
| DSA indexer | 174.92 | 2.19 | 1024 |
| HC mix finalize | 134.53 | 1.69 | 396 |
| vectorized add | 57.92 | 0.73 | 166 |
| index select small | 52.12 | 0.65 | 14814 |
| routing GEMM i8 (MoE routing) | 44.10 | 0.55 | 156 |

The 40.70% P2P share reported by torch's self-CUDA table uses its 8065 ms denominator, which includes non-kernel events. P2P issues 204881 launches, consistent with its 256 KiB payload limit; reducing large-prefill collective launch/chunk overhead is the clearest measured opportunity, subject to correctness and matched A/B. Treat the old 30.6% MoE share as a different workload, not a measured share here. As a purely hypothetical bound, removing all 3282 ms of measured P2P kernel time from the profiled 8566.6 ms TTFT gives ~6201 tok/s; this is not an achieved throughput or a physical hardware ceiling and assumes no overlap or secondary effects. A hardware ceiling cannot be determined from this profile alone.


### H1 (issue-bound) vs H2 (memory-bound): settled as H2
Their decision rule was issue-active >= 80% => issue-bound. Measured on our card with the standalone bench
(`ncu --graph-profiling node`, one call each):
| kernel | grid | issue_active | DRAM | us |
|---|---|---|---|---|
| `exl3_moe_coop_a_mixed<2,0>` | 504x256 | **45.8%** | **47.8%** | 44.6 |
| `exl3_moe_coop_b_mixed<2,0>` | 1120x256 | **42.7%** | **33.3%** | 27.6 |
Neither engine is saturated: it is **memory latency/granularity**, consistent with 44% long-scoreboard. So the
issue-cutting branch (PRMT byte windows, dropping staging for 6 bpw) is the WRONG direction; latency hiding is right.
- `EXL3_MOE_COOP_WIDE_B=1` at bsz1 (the obvious "wider per-warp span"): **worse** — 67.29 vs 65.66 us/call, B 27.87 vs
  25.38. Confirms the note in `exl3_moe_coop.cu` that bsz1 keeps B narrow, in the face of the comment saying otherwise.
  Keep narrow at bsz1.

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

### Warm prefix reuse (2026-10-27, TP2 P2P, B1)

`pcbench20261027a`, 1024-token chunks, 32 greedy output tokens, two repetitions;
rank 0 shown (rank 1 agrees within 0.11 ms and all comparisons match). Effective
tok/s counts the entire prompt, including reused tokens, divided by TTFT.

| Prompt | rep | cold ms / tok/s | full hit ms / tok/s (tokens) | partial cold ms / tok/s | partial hit ms / tok/s (tokens) |
|---:|---:|---:|---:|---:|---:|
| 2048 | 0 | 936.48 / 2186.9 | 195.27 / 10488.2 (1792) | 921.15 / 2223.3 | 533.69 / 3837.4 (1024) |
| 2048 | 1 | 890.46 / 2299.9 | 212.39 / 9642.8 (1792) | 916.03 / 2235.7 | 528.55 / 3874.7 (1024) |
| 8192 | 0 | 3177.94 / 2577.8 | 196.30 / 41731.9 (7936) | 3178.20 / 2577.6 | 1654.98 / 4949.9 (4096) |
| 8192 | 1 | 3178.50 / 2577.3 | 194.07 / 42210.9 (7936) | 3203.83 / 2556.9 | 1651.37 / 4960.7 (4096) |

All 32 generated tokens matched the corresponding cold run for every full and
partial hit on both ranks. Full reuse ends at the last page-aligned recurrent
checkpoint; partial reuse also requires the intact chained page hashes and
replays the remaining suffix. A prompt length `k*256+1` leaves its prefill end
page-aligned and can maximize reuse. This is B1 evidence, not a batch correctness
claim. SM telemetry and per-rank results are in `pcbench20261027a_rank{0,1}.log`.

The measured 8704-token preallocated paged cache was 27,852,800 bytes per GPU
according to `CacheLayer.storage_size()`: 3200 bytes/token, or 0.02594 GiB for
8704 tokens; 7936 reused tokens correspond to 25,395,200 bytes (0.02365 GiB)
within that allocation. This accounting excludes any DSA mirror pools and Python
page-table metadata; their incremental cost was not measured. Retention uses the
existing allocation, not an additional KV allocation. Host recurrent snapshots
measured 67,170,304 bytes for the 2048-token cold prompt and 268,681,216 bytes
for the 8192-token cold prompt per rank. Unreferenced pages are evicted when
space is required using the tree policy: empty pages first, then oldest leaves,
preserving referenced and protected chains as long as possible; checkpoint
storage has a separate host-memory LRU budget.

Runner dials: `EXL3_PREFIX_CACHE=1`, `EXL3_PREFIX_CACHE_CAPACITY` (default
4294967296 host bytes), `EXL3_PREFIX_CACHE_MIN_MATCH` (256 tokens), and
`EXL3_PREFIX_CACHE_INTERVAL` (1024 tokens). The interval caps prefill chunks;
shorter chunks and checkpoint copies add cold-prefill cost.
