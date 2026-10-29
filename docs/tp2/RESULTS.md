# Measured results

Every number in this file was measured on the hardware described below, with the configuration named next to
it. Nothing here is extrapolated or projected; values that were not measured are marked *expected*. Each row
cites the run tag or log it came from; the raw working notes behind them are in [`notes/`](notes/).

If a number has no condition attached, it is a bug in this document — please open an issue.

## Test system

| | |
|---|---|
| GPUs | 2 × RTX PRO 6000 Blackwell Max-Q, 96 GB each, PCIe P2P, same NUMA node |
| Clock behaviour | power-capped: ~2100 MHz against a 3090 MHz ceiling under sustained load; GPU1 flags `SW power cap` |
| Model | DeepSeek-V4.1-Flash-1017, mixed-bitrate EXL3, 2.565 bpw average ([recipe](RECIPE.md)) |
| Weights | 86.9 GiB resident per rank after load (89.6 GiB shard set on disk) |
| Transport | `COMM=p2p` unless stated (one-shot P2P all-reduce over PCIe) |
| Graphs | `EXL3_WHOLE_GRAPH=1`, `EXL3_DSV4_B1_BATCH=1` unless stated |

**How to read these tables.** These cards drift with host load and thermal state; A/B comparisons are only
valid back-to-back in one session. Several early comparisons in [`notes/`](notes/) are marked invalid for exactly
that reason and are not repeated here. Where two rows come from different sessions, the absolute values are
comparable but the delta between them is not.

## 1. Single-stream decode

Greedy, 128 generated tokens, warm pass excluded, both ranks reporting identical tokens.

| Stage | ms/token | tok/s | Conditions | Source |
|---|---|---|---|---|
| Eager decode (before the attention core was captured) | 13.35 / 13.35 / 13.17 | 75 / 75 / 76 | prompts 0/1/2 | `notes/historical-status.md` |
| Piecewise graphs, attention core captured | **10.97 / 11.02 / 10.98** | **91** | prompts 0/1/2, `NUM_DRAFT=0` | `notes/historical-status.md` |
| + MoE prefetch hoist and PDL launch of kernel B | 10.80 / 10.91 / 10.85 | 92 | same prompts | `notes/master-plan.md` |
| + three fused launch merges (default on) | **10.67 / 10.74 / 10.67** | **93.7** | same prompts | `notes/master-plan.md` |
| 1134-token prompt | 14.15 | 71 | long-context KV cost | `notes/historical-status.md` |

Capturing the decode attention core was the single largest step: 13.15 → 10.97 ms/token on the tree where it
landed (`notes/master-plan.md`), verified token-exact on 4 prompts × 128 tokens × 2 ranks; the later merged
tree shows the same effect as 13.35 → 10.70 (`notes/historical-status.md`). The cause of every earlier
"capture is slower" reading was a capture-gate mismatch (`b1_prep` required `q_head_norm=True` while
`_batch_ok_b1` required `False`), so no graph was ever replayed — a class of bug this project hit twice, see §8.

## 2. Speculative decoding (DSpark)

Drafter sharded TP=2, one DFlash block forward per round, `DSPARK_POLICY=always`, rank-0 decision broadcast.

| Config | ms/token | tok/s | Acceptance / AL | Source |
|---|---|---|---|---|
| plain, prompts 0–2, three fused launch merges on | **10.67 / 10.74 / 10.67** | **93.7 / 93.1 / 93.7** | — | `notes/master-plan.md` (item 20), both ranks |
| plain, prompts 0–2, capture landed, before those merges | 10.97 / 11.02 / 10.98 | 91 | — | `notes/historical-status.md` |
| K=5, prompts 0–3 | **7.86 / 6.83 / 9.01 / 4.65** | **127 / 146 / 111 / 215** | .66 / .78 / .91 (p0–p2), AL 3.30 / 3.30 / 2.58 / 5.29 | `notes/historical-status.md` |
| K=5, prompts 0–2, long-context compaction tree | 7.87 / 6.79 / 8.95 (rank 0), 7.86 / 6.79 / 8.95 (rank 1) | — | — | `integrator_postmerge_1028` |

Round cost is close to flat in K (one drafter block forward per round at every K; 19.7 / 20.7 / 21.3 / 22.1 ms
per round for K=2/3/4/5 in that measurement), which is why the default is "always speculate at K=5" rather
than a content classifier: a fixed K=5 beat the `auto` policy on 3 of 6 matched prompts, and the whole K choice
is worth ≤ 8%.

Long context is different: with a 20k-token filler prompt, speculative decoding helps or hurts depending on
what the text is. Measured at 32 generated tokens, `COMM=p2p`, KV pool 65536, chunk 2048:

| Content, 20k | Plain ms/tok | Spec K=5 ms/tok | Acceptance | AL |
|---|---|---|---|---|
| Code | 16.84 / 17.05 / 16.91 | 18.51 / 8.99 / 11.16 | .42 / .45 / .74 | 2.10 / 2.67 / 2.55 |
| Mixed | 17.19 / 16.68 / 17.11 | 13.49 / 8.80 / 9.66 | .78 / .49 / .64 | 3.25 / 2.82 / 2.80 |
| JSON | 16.87 / 17.40 / 17.02 | 13.34 / 13.18 / 9.21 | .68 / .37 / .63 | 3.50 / 2.39 / 2.58 |

(Synthetic filler prompts; the three values per cell are prompts 0/1/2. Synthetic fillers can induce loops, so
these are not a substitute for real transcripts — see the honest caveat in
[`notes/master-plan.md`](notes/master-plan.md).)

## 3. Prefill

Throughput is the steady-state rate of a cold prefill of an N-token prompt at a given chunk size, timed
end-to-end in the runner (queueing and TTFT included).

| Prompt tokens | Chunk | TTFT | tok/s | Peak reserved / free | Source |
|---|---|---|---|---|---|
| 32,768 | 8192 | 6858 / 6909 ms | **4778 / 4743** | 90.7 GiB / ~6.0 GiB | `tp2logs/final32768_8192_rank0.log` (NCCL) |
| 32,768 | 8192 | 6799 / 6838 ms | **4819 / 4792** | 90.7 GiB / ~6.0 GiB | `tp2logs/arena_sweep_256_hybrid_1028_rank{0,1}.log` (`COMM=hybrid`) |
| 32,768 | 16384 | 6777 / 6875 ms | 4835 / 4766 | 93.9 GiB / ~1.0 GiB — not shippable | `tp2logs/chunkbig_rank0.log` |
| 32,768 | 32768 | OOM (36 MiB free at the allocator) | — | — | `notes/master-plan.md` |
| 2,048 | 2048 | 573.9 / 573.9 ms | 3569 | 87.7 GiB | `tp2logs/final2048_2048_rank0.log` |
| 2,048 | 4096 | 573.8 / 570.1 ms | 3569 / 3592 | 87.7 GiB | `tp2logs/final2048_4096_rank0.log` |
| 2,048 | 2048 | — | 3034 (`COMM=p2p`) vs **3399** (`COMM=nccl`) | — | `notes/master-plan.md` |

Chunk 16384 buys < 1% over 8192 and costs ~5 GiB of the allocator's headroom, which is why 8192 is the
practical ceiling and `MAX_CHUNK=auto` selects from the measured peak table rather than from a formula. Peaks
above are the allocator's `torch_peak_reserved_GiB` at the end of each run; the "peak (smi)" column in
[`notes/master-plan.md`](notes/master-plan.md) reports higher transient numbers (93.9 / 97.1 / 97.2 GiB) from
driver sampling and is superseded by these figures.

**The collective mode depends on the phase, and this is the most misquoted result in this project.** The
one-shot P2P kernel wins on decode (10 KB payloads: 4.93 vs 11.84 µs per call) and loses on prefill (84 MB
payloads chunked through a fixed arena): +12% for NCCL on a 2048-token prefill. `COMM=hybrid` (P2P below a byte
threshold, NCCL above, `HYBRID_NCCL_MIN_BYTES=65536`) recovers that: **p2p + 256 KiB arena 4137 → hybrid 4819
tok/s (+16.5%)** at 32K/chunk 8192, and 4704.9 tok/s re-verified on main. Arena size is a P2P-only lever
(256 KiB → 4 MiB: 4137 → 4369 tok/s). Per-call cost is only meaningful with the arena size quoted: 256 KiB
arena → 204,881 calls at 16.0 µs = 3.282 s = 40.7% of prefill CUDA time; 4 MiB arena → 12,881 calls at
194.1 µs = 2.500 s = 34.4%.

Rank-0 kernel-time split at 32K/chunk 8192/`COMM=p2p` (7982 ms of kernel events): P2P all-reduce 41.1%,
CUTLASS GEMM 7.5%, DSA attention 7.4%, HC apply 3.9%. The all-reduce share is the whole reason `hybrid` exists.

## 4. Long context

End-to-end in the runner, `COMM=hybrid`, chunk 8192, graphs on, `EXL3_DSV4_B1_BATCH=1`, cold (no warmup):

| Prompt tokens | TTFT | Prefill tok/s | Decode ms/token |
|---|---|---|---|
| 24,167 | 5081 ms | 4756 | **16.58** |
| 60,417 | 13,068 ms | 4624 | **16.88** |
| ~24,000 with `WARM=1` | 1843 ms | (page reuse from the warm pass, not a throughput figure) | 16.50 |

For comparison, short-context decode is 10.67–10.84 ms/token, so the 24k–60k regime costs ~55% per token.

The gap is the **eager fallback**: the captured decode body has no top-k indexer selection, so the capture gate
(`_batch_ok_b1`) falls back to eager past roughly 1024 positions. Profiled at 24,167 tokens, 15 decode steps:
13.68 ms of CUDA time per step, 631 kernel launches, 8 `aten::nonzero` synchronizations. (The unprofiled run is
16.55 ms/token; do not subtract these — profiling changes overlap. The two are reported separately.)

What was done about it (2026-10-28, long-context candidate compaction):

| Metric | Before | After |
|---|---|---|
| `aten::nonzero` per step | 8 | **0** |
| `cudaLaunchKernel` per step | 585 | **557** |
| 24,167 tokens, 32 decode tokens | 16.42 → 16.78 ms/tok (two runs) | **16.11 → 15.79** |
| 72,510 tokens | 16.64 ms/tok | **16.09** |

Tokens were identical on both ranks in every pair; host load 6.59 → 8.08; GPU clocks median 2272/2272 MHz.

Ties matter for this work: at 24,166–24,485 positions, the 2048th and 2049th candidate scores were **exactly
equal on 269 of 320 steps (84.06%)**, on both ranks (`topk_ties_1028b.csv.rank{0,1}`). A capture-safe selector
therefore has to reproduce eager tie order exactly; padding with `-inf` does not (`torch.topk` breaks ties by
index, so padding changes the answer).

## 5. Prefix cache

`EXL3_PREFIX_CACHE=1` is the default. Recurrent-state checkpoints are taken at 8192-token chunk boundaries;
a reused prefix replays from the last page-aligned checkpoint.

| Workload | Cold | Warm (full hit) | Speedup | Source |
|---|---|---|---|---|
| 20,000 tokens, zero-hit | 4269–4288 ms | **108–117 ms** | **36–40×** | `longprefix1027e` |
| 60,000 tokens, zero-hit | 13,253–13,436 ms | **153–154 ms** | **86–88×** | `longprefix1027e` |
| 8,192-token prompt | 3178 ms | **196 ms** | 16× | `notes/master-plan.md` |
| 2,048-token prompt | 936 ms | 195 ms | 4.8× | `notes/master-plan.md` |
| partial hits, 2k / 8k | 921 / 3178 ms | 534 / 1655 ms | ~1.9× | `notes/master-plan.md` |

Two alternating distinct 60k contexts both hit. A 60k context growing ~1k per turn stays ≥ 98% cached
(TTFT 524 / 513 / 491 ms at 61k / 62k / 63k).

Cost: **268 MB of host RAM per rank per 60k context** (8 checkpoints) plus 183 MiB of GPU KV for that context.
`EXL3_PREFIX_CACHE_CAPACITY` (4 GiB) is the host budget, not VRAM. Short-context decode is unaffected:
10.83 / 10.86 / 10.84 ms/token with the cache on vs 10.77 / 10.84 / 10.83 off, tokens byte-identical.

## 6. Concurrency

Short prompts, captured batched path, `EXL3_DSV4_B1_BATCH=1`:

| Batch | ms/token per stream | Aggregate tok/s | Peak | Source |
|---|---|---|---|---|
| B1 | 10.84 | 92 | — | `notes/master-plan.md` |
| B2 | 13.91 | 133.8 | — | `notes/master-plan.md` |
| B4 | 17.89–17.94 | **203–206** | 86.5 GiB/GPU | `notes/master-plan.md` |
| B4 + DSpark K=5 | 38.26–38.97 | 96.8 | — | `notes/master-plan.md` |
| B6 coop / grouped | — | 247.5 vs 138.7 | — | `notes/master-plan.md` |
| B8 coop / grouped | — | 213.3 vs 141.3 | — | `notes/master-plan.md` |

**Speculative decoding and batching are substitutes, not a product.** At these widths the verifier is already
wide enough that drafting costs more than it saves; B4 plain aggregates nearly twice B4+DSpark.

**Batch invariance, measured 2026-10-28** (`invariance_block_20261028_*`; greedy, 128 tokens, B2/B4/B8, same /
distinct / mixed neighbours, graphs on and off, full logits recorded):

- **Invariant across streams and neighbours, bitwise.** Every row of a batch is token- and logit-identical
  (max-abs delta exactly 0 over 128 steps), and at fixed width stream 0 is bitwise identical regardless of what
  the other streams are. A row's result does not depend on its neighbours.
- **Not invariant across batch width, and not monotone in batch size.** With graphs on, B2 ≡ B4 bitwise over
  all 128 steps; B8 is a third path (differs from step 0, max-abs 0.102, mean 0.017); B1 is a fourth. With graphs
  off, each width is its own path. Graphs on vs off is itself a numeric path: bitwise identical at B1,
  token-equal at B2, first flip at 55 (B4) and 11 (B8).
- **All flips sit on near-ties.** Decision-relevant deltas are ≤ 0.94 (top-1 ≤ 0.891, top-2 ≤ 1.219, top-2
  margin ≤ 0.9375); flips happen where the reference margin is 0.03–0.78, i.e. below the perturbation. The
  0.8–2.9 whole-vocabulary max-abs figures come from tail tokens and are not decision-relevant.
- An earlier claim that two identical rows in one B4 batch diverge at position 11 is **retracted** — it was a
  harness bug.

## 7. Kernel-level work (decode MoE)

The decode MoE kernel is the second-largest GPU consumer at short context. Measured with the standalone bench
at bsz1 unless stated, and with in-model confirmation:

| Change | Effect |
|---|---|
| Remove a local-memory load in `gemv_tile` (a ternary took the address of a per-thread `const`, making the false branch a local load) | 67.31 → **65.76 µs/call** (1.024×); ncu had shown ~49% of load sectors going to local memory |
| Hoist the trellis prefetch ring + PDL-launch kernel B | **62.54 µs/call** (1.076×), 1.052× at bsz6; in-model 10.97/11.02/10.98 → 10.80/10.91/10.85 |
| Three fused launch merges (`EXL3_DSV41_FUSED_ROPE`, `EXL3_DSV41_FUSED_POOL_SCATTER`, `EXL3_DSV4_FUSED_INDEXER_ROPE`) | short 10.78/10.86/10.79 → **10.67/10.74/10.67**; 24k 16.55 → 16.32/16.34; launches/step 630.8 → **585.0** |

Rejected after measurement — recorded so nobody repeats them: wider tiles for kernel A or B (−9% / −2.5%),
`WK=16` (+28%), k-split 2/3 on kernel A (−21%), expert-grouped decode at B ≤ 8 (−44% at B6), int8 decode-once
for prefill MoE (already spent), 8-bit prefill all-reduce (+6.9% but failed the coherence gate), FP8 KV
quality-fine but capture-breaking. Details in [`notes/MoEKernelTP2.md`](notes/MoEKernelTP2.md) and
[`notes/master-plan.md`](notes/master-plan.md).

## 8. Correctness gates actually applied

Numbers above are only reported for runs that passed these:

- **Token identity across ranks.** Both ranks sample with a rank-agreed greedy sampler; every result table
  above comes from a run where rank 0 and rank 1 emitted the same token ids for the same job.
- **Token identity across trees** for every optimization: the fused merges, the compaction, and the MoE kernel
  patches were each accepted only with byte-identical tokens against the pre-change tree on both ranks.
- **Capture actually replayed.** `cudaGraphLaunch` counters are checked, not assumed: a capture-gate mismatch
  silently halves throughput and produces plausible-looking numbers (this happened twice).
- **Batch invariance as measured in §6**, including the retraction.
- **Prefix-cache hit correctness**: 8 full-hit and 8 partial-hit comparisons, token-identical on both ranks.
- **Prefill-chunk invariance** (added for v1.0.0, after it caught the mixed-rate prefill bug): the same text
  must score the same perplexity at 256-, 2048- and 4096-row prefill chunks (8k-token WikiText-2 windows:
  2.1820 / 2.1827 / 2.1827, ±0.025). Self-consistency checks — identical repeats and identical ranks — cannot
  catch an error that is the same on every run; this one can.

## 9. v1.0.0 final measurements (fixed build)

Measured on the release tree after the mixed-rate prefill fix, default settings (speculation on, adaptive
cutoff, `COMM=p2p`, 4096-row prefill chunk), interleaved A/B blocks, logs under `<artifacts>/final/logs/`:

| Measurement | Result |
|---|---|
| Decode, 24k context, single user | agentic 142, code 177, math 189, structured 169, prose 97, multilingual 100 tok/s |
| Decode, long-file tails (72k) | agentic 127, code 118, prose 109, math 136, structured 153, multilingual 86 tok/s |
| Decode, 2k prompts | agentic 128, code 164, prose 109, math 154, structured 155, multilingual 88 tok/s |
| Decode, long context | 72k agentic 128, 72k code 128, 128k agentic 124 tok/s |
| Adaptive cutoff vs fixed (19 prompts) | +0.6% tok/s overall; best +3.0%, worst −0.9%; every repeat token-identical |
| 4 users, 12k prompt / 16k KV each | 189 tok/s total, 58 per user |
| Cold prefill (no prefix cache) | 2,614 tok/s at 24k, 2,782 at 72k, 2,746 at 128k |

Earlier sections record the development history on earlier trees; where they disagree with this table, this
table is the release build.

## 10. What this is and is not

- **Is:** a true TP=2 runtime for a 2.565 bpw mixed-bitrate EXL3 checkpoint of a 100B-class MoE model, fitting
  two 96 GB cards with speculative decoding always on, at 97–189 tok/s single-user decode at 24k context
  depending on content, 124 tok/s at 128k, and prefix caching for repeated long prompts.
- **Is not:** a claim to beat every other runtime. As a reference point published by a third party on the same
  hardware class for a *different* checkpoint and configuration (diffbot, EXL3 3 bpw, fp8 KV, their own
  drafter): 156.9 tok/s single-stream at 2k context, 362.9 tok/s aggregate with 4 streams at 46k, and
  11.2–11.5k tok/s fresh 40k prefill warm ([public-number survey](notes/EXL3_SPEED_SURVEY.md)). Different
  checkpoint, different KV precision, different batching policy. This runtime does not win on multi-user
  throughput or prefill.
- **Known limits**, all measured: decode attention runs per user when several users are active (weak
  multi-user scaling); 8 users with speculation do not run; FP8 KV breaks capture; the 8-bit prefill all-reduce
  fails the coherence gate.
