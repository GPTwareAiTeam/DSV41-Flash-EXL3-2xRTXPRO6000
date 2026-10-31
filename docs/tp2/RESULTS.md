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
  patches were each accepted only with byte-identical tokens against the pre-change tree on both ranks. Both
  trees must run on copies of the same GEMM autotune file (`EXLLAMAV3_TUNE_CACHE`): the autotuner stores its
  picks per machine/container, and different picks change results enough to flip greedy tokens.
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

### 9a. Drafter window ring (after v1.0.0)

The drafter's context-sized cache (3 layers × 512 fp16 values = 3,072 B per token per GPU) was replaced by a
768-row ring per user per layer. Release build vs ring build, both on byte-identical copies of one GEMM autotune
file (`EXLLAMAV3_TUNE_CACHE`; see 8), logs under `<artifacts>/kvc/logs/` and `<artifacts>/ideas/`:

| Check | Release | Ring |
|---|---|---|
| Alternating 24k / short jobs, both orders, 512 tokens | — | token-identical on both ranks, every job |
| Short prompts (code, summary, chat), ABBA, tok/s | 188.0 / 226.8 / 218.8, 188.3 / 232.0 / 221.7 | 187.6 / 229.4 / 221.2, 185.5 / 230.9 / 220.8 (identical tokens) |
| Prefix cache, 25.8k prompt twice, 256 tokens | TTFT 9,012 → 194 ms | TTFT 9,024 → 186 ms (turn 2 = turn 1 = release) |
| 2 users / 4 users, total tok/s | 124.5 / 126.4 | 125.4 / 129.0 (identical tokens) |
| Free after drafter load, 64k cache | 3.92 GiB | 4.12 GiB |
| Free after drafter load, 512k cache | 1.25 GiB | 2.75 GiB |
| 150k-token prompt, 512k cache, `MAX_CHUNK=2048` | out of memory | runs, 85.5 tok/s decode, peak 91.98 GiB |
| 150k-token prompt, 1M cache | — | out of memory (main KV cache alone is 3.1 GiB per 1M tokens over the 512k case) |

Without pinned tuning, two containers' independent autotune picks change the target's hidden states by ~1e-4
relative at layer 0 and ~1e-2 by the last layer; that alone flipped greedy tokens at positions 132 and 368 in the
first comparison and is not a property of either build.

### 9b. FP4 main KV cache (`KV_BITS=4`) and per-GPU memory

Same tree, fp16 vs FP4 main pools, pinned GEMM tuning, logs under `<artifacts>/fp4kv/logs/` and
`<artifacts>/comb/logs/`. FP4 output differs from 16-bit by design; FP4 runs are repeat- and rank-identical.

| Check | 16-bit cache | FP4 cache |
|---|---|---|
| Decode, 24k prompt, 256 tokens, 4 ABBA pairs (median) | 201.6 tok/s | 200.2 tok/s |
| Time to first token, same runs | 8.36 s | 8.34 s |
| Five planted facts at 10–90% depth, 128k prose / code | 5/5, 5/5 | 5/5, 5/5 (also 5/5 in a 1M cache) |
| Main cache per token per GPU | 3,200 B | ~890 B |
| 2 users, total tok/s, ABBA (first run of each kind is cold: 63.1 fp16) | 125.4 | 124.0 / 126.2 |
| 4 users, total tok/s | — | 125.3 |
| Prefix cache, 25.8k prompt twice, 256 tokens | — | TTFT 9.55 s → 0.19 s, turn 2 = turn 1 |
| Server (1M cache), chat + streaming | — | responds |

Kernel times, FP4 / fp16, one GPU at model shapes (24k / 128k pools): prefill attention 1.02 / 1.01, prefill
indexer 1.04 / 1.01, decode attention 1.04 / 1.02, decode indexer 0.90 / 1.00, verify (6 rows) attention
1.02 / 0.97, verify indexer 0.64 / 0.40.

Per-GPU memory with the drafter ring and FP4 together (`MEM_BREAKDOWN=1`, identical on both GPUs), loaded:

| Item | GiB |
|---|---|
| Target weights (routed experts 81.16 of it) | 86.31 |
| Drafter weights | 3.75 |
| Graph statics / workspaces | 2.75 |
| Recurrent state (window rings, compressor, drafter ring) | 0.03 |
| KV cache at 64k / 512k / 1M | 0.05 / 0.44 / 0.87 |
| Outside PyTorch (CUDA context, libraries) | 1.51 |

Peak during a request (reserved = PyTorch pool; card = whole device, read at end of run):

| Cache | Prompt | Prefill chunk | Peak reserved | Card used | Rise over loaded |
|---|---|---|---|---|---|
| 64k | 26.9k | 4096 | 91.77 | 93.43 | 2.56 |
| 512k | 162.8k | 2048 | 91.17 | 92.82 | 1.59 |
| 1M | 162.8k | 2048 | 91.59 | 93.25 | 1.57 |
| 1M | 131.2k | 2048 | 91.64 | 93.23 | 1.62 |
| 1M | 131.2k | 4096 | 92.76 | 94.34 | 2.73 |

Card used ≈ loaded reserved + rise + ~1.7 GiB outside PyTorch. A 4096-row chunk at 1M leaves ~0.6 GiB of the
94.97 GiB card; 2048 rows leave ~1.7 GiB at a ~12% longer time to first token (54.6 vs 47.9 s at 131k).

### 9c. Prefill chunk size and the adaptive chunk limiter

Fixed chunks (FP4 KV, 1M cache, `PREFILL_ADAPTIVE=0`), time to first token / prefill rate / card used at peak:

| Chunk | 26.9k prompt | 131k prompt | Card used |
|---|---|---|---|
| 1024 | 11.32 s, 2,375 tok/s | 66.5 s, 1,974 tok/s | 92.65–92.80 GiB |
| 2048 | 9.36 s, 2,872 tok/s | 54.9 s, 2,392 tok/s | 93.23–93.27 GiB |
| 4096 | 8.24 s, 3,262 tok/s | 47.8 s, 2,746 tok/s | 94.19–94.34 GiB |

A chunk's transient grows with its depth as well as its rows (the uncompressed pool layer's fp16 staging window
and the indexer score slab: ~1.25 KiB per context token past ~300k), so a chunk that fits a shallow prompt can run
out of memory deep in a long one. `PREFILL_ADAPTIVE=1` (default) reads free memory once per chunk and solves for
the largest chunk (whole pages, at most `MAX_CHUNK`) that leaves 1 GiB free; ~0.57 ms per decision. ABBA at host
load ≤ 3.7, 26.9k prompt, all six runs token-identical:

| Setting | Time to first token | Peak reserved |
|---|---|---|
| Fixed 4096 | 8.20 / 8.34 s | 92.36 GiB |
| Adaptive, planned 4096 (never shrank) | 8.30 / 8.36 s | 92.29 GiB |
| Adaptive, planned 8192 (5,120–6,656 rows used) | 8.14 / 8.15 s | 92.90 GiB |

876k-token prompt, adaptive, planned 8192: 369 chunks, 6,656 rows shrinking to 3,328 by the end, prediction never
more than 66 MiB under the measured rise; time to first token 390.6 s (2,243 tok/s), peak 92.97 GiB reserved, decode
6.43 ms/token, both planted facts (10% and 90% depth) answered. Odd chunk sizes cost no per-row speed (3,584-row
chunks 3,200–3,330 rows/s against 3,300–3,390 for 4096).

Two fixes came with it: per-shape staging/output buffers in `DSV4Attention._forward_cached_one` were kept for the
life of the process (one copy per distinct chunk size, ~200 MiB each); prefill-sized ones now go through the
caching allocator (`g_tensor_cache.get_scratch`), decode-sized ones stay kept for graph capture. And held prefill
memory is returned to the card when a prompt's prefill ends, since decode captures its graphs from a separate pool.

With `KV_BITS=4`, output can differ between chunk sizes where the top two candidates are nearly tied: on the
26.9k prompt the second token's top-2 logit gap is 0.047 at chunks 1024 and 2048 (different winners) against
0.89–1.34 with 16-bit KV, which is token-identical across 1024 / 2048 / 4096. The limiter itself adds no
difference: adaptive runs are token-identical to a fixed chunk of the same size.

## 10. v1.1 measurements (2026-10-30/31)

Tree `recipe/2x-rtxpro6000-tp2` at the v1.1 merge, container `exl3-tp2:clean-20261028`, one extension build,
pinned GEMM tuning per run, FP4 KV, 1M-token cache, DSpark K=5, 256 generated tokens (ignore-EOS) unless noted.
Logs: `<artifacts>/rowmap/logs/`.

### 10a. Checkpoints

2.499 bpw re-quantisation of the routed experts (ranked by routing frequency): 1,400 at 4 bpw, 4,860 at 3,
9,100 at 2. Every non-expert tensor is byte-identical to the 2.565 build (verified on all 40 layers); the 10,660
experts not re-quantised are byte-identical to their source build.

| | 2.565 | 2.499 |
|---|---|---|
| Target weights per GPU | 86.31 GiB | 84.18 GiB |
| Free per GPU after load | 3.43 GiB (4-bit drafter experts) | 6.32 GiB (3-bit drafter experts; 5.45 with 4-bit) |
| WikiText-2 ppl, 8 × 2048 chunks, 16 / FP4 KV | 2.038 / 2.044 | 2.078 / 2.079 |
| 1017 panel, 8 conversations, 16 / FP4 KV | 1.777 / 1.778 | 1.789 / 1.789 |
| sudoingx panel, 8 conversations, 16 / FP4 KV | 1.675 / 1.676 | 1.681 / 1.681 |
| 5 planted facts, 128k prose / code | 5/5, 5/5 | 5/5, 5/5 |
| 876k-token prompt, 2 facts | 2/2 | 2/2 |
| 8 users × 131k in one 1.5M cache, 5 facts each | — | 8 × 5/5, peak 91.1 GiB/GPU |

FP4 KV costs +0.3% WikiText perplexity and +0.05% on the panels against 16-bit KV.

Full panels (199 conversations each, 237,457 / 254,846 scored positions, FP4 KV): 2.499 scores 1.6998 / 1.7138
against 2.565's 1.6886 / 1.7041 (+0.66% / +0.57%). WikiText-2, all 140 chunks: 2.5241 ± 0.0129.

### 10b. Speed, release v1.0 vs v1.1

ABBA × 2, same container and extension:

| | v1.0 (b1392654, 2.565) | v1.1, 2.565 | v1.1, 2.499 |
|---|---|---|---|
| 6 short prompts, mean tok/s | 158.3 | — | 158.9 |
| 26.9k prompt, tok/s | 197.5 | 202.0 | 172 (4-bit drafter experts) / 164.8 (3-bit) |
| 131k prompt, tok/s | 157.7 | — | 170.1 |
| Verify step, 26.9k | 24.8 ms | 23.5 ms | 25.5 ms |
| Tokens per verify step, 26.9k | 4.90 | 4.72 | 4.18 / 4.20 |

Per verify step v1.1 is as fast or faster; the 2.499 checkpoint's speed differences come from drafter acceptance.

24k corpus prompts, 256 tokens, decode tok/s (2.565 / 2.499 + 3-bit drafter / 2.499 + 4-bit): code 154 / 137 / 142,
math 141 / 175 / 162, prose 112 / 109 / 112, structured 182 / 175 / 185, multilingual 97 / 94 / 91; mean 137 / 138 / 138.

8 users (code 2k prompt × 8, prefix cache on, 256 tokens, K=5): 229–240 tok/s total, first token 0.52 s (v1.0 on
2.565, same setup with K=3: 212 tok/s).

Drafter work on 2.499 that did not ship: a low-rank output adapter fit to 46,691 recorded draft rounds
(`EXL3_DSPARK_RECORD`) gained +0.5% held-out tokens/round once the markov bias is included; fixed draft widths and
a context-aware gate traded short-prompt speed for 24k speed with no net gain (and change the continuation).

### 10c. Changes and their measured effect

| Change | Measured |
|---|---|
| Runner/server default to graphs + captured single-token attention + FP4 KV | server smoke with no flags: 172 graph replays (was 0) |
| Drafter auto gate 0.35/1.5 → 0.25/2.0 (2.499) | 7 prompts: 143.3 / 144.7 → 153.4 / 154.0 tok/s |
| One-launch candidate scoring for verify rows | 26.9k, 3 ABBA pairs: 4.57 → 4.44 ms/token, token-identical |
| Multi-user graph-capture fix (MoE count readback) | 8 users with the drafter: crash → runs |
| Adaptive prefill chunk | 876k prompt with 8192 planned: completes |
| FP4 index keys in-kernel in prefill (opt-in only) | not bit-exact: changes tokens; off by default |
| FP4 direct gather at depth (opt-in only) | bit-exact; 876k prefill 5% slower; off by default |
| Batched V4.1 attention for B>1 (opt-in only) | slower (59–86 vs 36–37 ms/token per user), not token-identical |
| Drafter experts 3-bit (default) | 0.77 GiB/GPU less; 7 prompts ABBA 153.3 / 154.7 vs 4-bit 152.0 / 152.6 tok/s |

### 10d. Identity

Release v1.0 and v1.1 produce the same tokens on the same weights and drafter gate (26.9k prompt, 2.565).
Prefix cache: turn 2 = turn 1. Both ranks token-identical in every run. Batch width changes greedy output on
near-tied tokens (2/4 users vs single: 2 of 4 streams differ, first at token 2–250), as in section 6.

## 11. v1.2 investigations (2026-10-31)

Settings unless noted: 2.499 checkpoint, TP=2, FP4 KV, graphs, pinned GEMM tuning **warmed per variant before
timing** (a GEMM shape without a tune entry autotunes mid-decode: one ~1.5 s round, a false 15–20% loss on 8-bit
variants until warmed), 256 tokens, ignore EOS. Logs: `<artifacts>/{dtrain,drafterhq,dq2,hc}/`.

### 11a. Why code is slower than other content (same prompt)

Same 24k prompt file per content, three drafter/checkpoint arms, ABCCBA, repeat-identical tokens:

| code_24k | tok/s | tokens/round | wall ms/round | verify GPU ms (`EXL3_ROUND_PROF`) |
|---|---|---|---|---|
| 2.499 + 3-bit drafter experts | 137.9 / 137.9 | 3.19 | 23.7 | 17.6–17.8 |
| 2.499 + 4-bit drafter experts | 144.1 / 143.9 | 3.12 | 21.6 | 18.6 |
| 2.565 + 4-bit drafter experts | 156.5 / 155.5 | 3.46 | 22.0 | 19.0 |

The verify step is not slower on 2.499; acceptance is lower. The two checkpoints diverge at token 24 on code
(token 2 on math, where 2.499 wins 176 vs 140) and the continuations offer the drafter different positions. The
3-bit drafter's forward costs 1.42 ms vs 1.27 (4-bit): its 4-bit shared expert breaks the uniform-rate fold, so it
launches separately.

### 11b. Drafter quantisation matrix

Routed experts / attention + main_proj / shared expert, bpw. Speed block: all variants back to back, then reversed;
fixed width 5 on code_24k. Acceptance: 40 held-out prompts, auto gate, paired bootstrap over prompts.

| Variant | Drafter fwd GPU ms | Free GiB/GPU | Held tokens/round auto / fixed-5 | Held tok/s vs 3/4/4, 95% CI |
|---|---|---|---|---|
| 3/4/4 (v1.1) | 1.415 | 7.12 | 3.261 / 3.350 | — |
| 4/4/4 | 1.28 | 6.26 | 3.294 / 3.376 | +1.37% [−0.18, +2.86] |
| 3/6/6 | 1.52 | 6.98 | 3.306 / 3.359 | +0.44% [−1.61, +2.91] |
| 3/8/8 | 1.53 | 6.98 | 3.307 / 3.361 | −0.24% [−1.58, +1.01] |
| 4/8/8 | 1.54 | 6.11 | 3.319 / 3.370 | +0.98% [−0.47, +2.51] |
| 4/4/8 | 1.43 | 6.26 | (fixed width only, below) | — |
| 2/8/8, 2/6/6, 4/6/6 | 1.50–1.54 | — | 3.317 (2/8/8, fixed-5) | — |

Conditional acceptance per draft position is the same for every variant (~0.77, 0.73, 0.71, 0.72, 0.70): drafter
precision is not what limits acceptance. Fixed K=3 / K=5 on five 24k prompts, 4 variants (aggregate tok/s):
3/4/4 129.9 / 126.9, 4/4/4 130.8 / 129.3, 4/8/8 130.1 / 128.9, 4/4/8 130.2 / 126.7.

### 11c. Hot/cold drafter experts

Drafter routing picks (instrumented arm): each of the 3 drafter blocks has 128 routed experts, top 6 per token;
use concentrates with depth (block 2: 3 experts carry half of all picks, 21–33 carry 99%). Tiers from 153 training
prompts + 24 multilingual prompts (disjoint from every evaluation set):

| Variant | Tiers (bpw: experts per block 0/1/2) | Drafter ms | Free GiB | K=3 / K=5 aggregate tok/s (five 24k prompts) |
|---|---|---|---|---|
| 3/4/4 (v1.1) | 3: 128/128/128 | 1.41 | 7.12 | 127.0 / 126.2 |
| 4/4/4 | 4: all | 1.26 | 6.26 | 130.3 / 128.7 |
| **hot 4 / cold 2 (v1.2)** | 4: 101/72/49, 2: rest | 1.26 | 7.00 | 131.3 / 128.6 |
| hottest 8 / hot 4 / cold 2 | 8: 8/8/8, 4: 94/63/40, 2: rest | 1.30 | 6.75 | 131.2 / 127.9 |

40 held-out prompts, auto gate, two ABBA sessions: hot 4 / cold 2 vs 3/4/4 +1.36% [−0.20, +2.79] and +0.94%
[−1.05, +2.80]. The hot tier is 4 bpw like the drafter's shared expert, so the shared expert stays folded.

### 11d. Diffbot kernel ideas and routing prep

| Idea | Result |
|---|---|
| Decode MoE down projection with a 4-stage 16-byte cp.async shared-memory weight ring (diffbot `xmoe` flat loop) | bit-identical (rows 1–8, mixed 2/3/4/6 bpw) and token-identical; slower: 90.70 → 104.53 µs/layer (gate/up 123.57 → 124.10). Branch `tp2/moeflat-1031`, opt-in, not merged. |
| Nsight, 6 rows, baseline coop kernels | DRAM 68% (A) / 61% (B) of peak, ~46% long-scoreboard stalls, tensor pipe 19–23%: memory-latency bound; swapped MMA has weak support |
| Row top-k (diffbot `rowtopk`) | their kernel reads a padded max_model_len width; our indexer top-k already scans only the live compressed width |
| Fused routing prep (fold the norm into the carried hyper-connection mix, defer the apply under TP) | exact, token-identical, but no launch reduction: the folded norm is limited to width 4096 (model: 5120). Dropped. |
| Drafter shared-expert fold for a uniform-rate layer whose shared expert has another rate | token-identical; 136.0 → 136.1 tok/s short-prompt aggregate (no gain). Branch `tp2/dfold-1031`, not merged. |

## 12. v1.2 final measurements (2026-10-31)

Tree `tp2/adaptive-chunk` at the v1.2 commit, one extension build, container `exl3-tp2:clean-20261028`, shipped
defaults (DSpark K=5 with the auto gate, FP4 KV, graphs, CPU-pinned ranks), v1.2 drafter (hot 4 / cold 2), pinned
GEMM tuning warmed once, 256 tokens, ignore EOS, prefix cache off. Logs: `<artifacts>/final2/logs/`.
Unit tests: `tests/kernels/moe tests/generator tests/cache` 554 passed, 1 pre-existing failure
(`test_failing_allocate_pages_is_contained`, fails on v1.0 too). v1.1's drafter on the v1.2 build is token-identical
to the v1.1 build (6 prompts, both ranks); v1.2 drafter rank- and repeat-identical.

### 12a. Single user, by content (ABBA: v1.1 drafter, v1.2, v1.2, v1.1)

| Prompt set | v1.1 drafter tok/s | v1.2 tok/s | tokens/round v1.1 → v1.2 |
|---|---|---|---|
| code 24k | 139.9 / 138.0 | 129.9 / 130.4 | 3.19 → 2.88 |
| math 24k | 176.6 / 176.4 | 161.6 / 163.8 | 4.57 → 4.38 |
| prose 24k | 111.2 / 111.5 | 113.2 / 113.8 | 2.45 → 2.46 |
| structured 24k | 176.6 / 175.1 | 185.5 / 185.3 | 4.15 → 4.18 |
| multilingual 24k | 98.6 / 99.3 | 90.9 / 92.7 | 2.10 → 1.89 |
| 7 short prompts | 159.1 / 159.1 | 157.3 / 158.8 | 3.52 → 3.50 |
| 40 held-out prompts | 151.8 / 155.5 | 157.3 / 157.5 | 3.26 → 3.26 |
| all (sum of tokens / sum of decode time) | 150.7 / 153.5 | 153.9 / 154.4 | |

Held-out prompts, per-prompt best of two: v1.2 +0.94% [−1.05, +2.80]. Single 24k prompts swing ±10% either way
because each drafter changes near-tied greedy choices and the continuation differs (30 of 40 held-out outputs
differ between the arms); the held-out set is the basis for the default.

### 12b. Concurrent users (one shared cache; every user prefills; decode throughput excludes prefill)

| Users | 2k prompts: total / per user tok/s | 24k prompts: total / per user tok/s | Last first token, 2k / 24k | Peak reserved GiB, 2k / 24k |
|---|---|---|---|---|
| 1 | 156.2 / 156.2 | 131.2 / 131.2 | 0.8 s / 8.3 s | 87.8 / 88.8 |
| 2 | 146.6 / 73.3 | 146.9 / 73.4 | 1.6 s / 16.7 s | 87.9 / 89.4 |
| 4 | 176.1 / 44.0 | 162.7 / 40.7 | 3.2 s / 33.7 s | 88.2 / 89.8 |
| 8 | 255.0 / 31.9 | 230.0 / 28.8 | 6.4 s / 66.7 s | 88.7 / 90.1 |
| 16 | 215.3 / 13.5 | 253.6 / 15.9 | 12.9 s / 133.2 s | 89.9 / 90.9 |

Cache sized at 8k (2k prompts) or 32k (24k prompts) tokens per user; speculation on in every run (K=5 effective).
Decode attention runs one user at a time, so total throughput levels off at ~250 tok/s.

### 12c. Context limits

| Test | Result |
|---|---|
| Shared cache 8,388,608 tokens, 8 users | fails at load (out of memory) |
| Shared cache 6,291,456 tokens, 8 users × 2k | loads (1.61 GiB free); before the shared-page fix: recurrent-checkpoint assert; after: out of memory in decode |
| **Shared cache 5,242,880 tokens**, 8 users × 2k | runs; KV 4.35 GiB/GPU, 2.47 GiB free after load, peak 93.0 GiB reserved |
| 5,242,880 cache, 1 user × 131,233-token needle prompt | 5/5 facts; first token 49.9 s; decode 5.76 ms/token; peak 92.7 GiB reserved |
| 5,242,880 cache, 8 users × 131,233 tokens | 40/40 facts; last first token 407 s; 25.5 ms/token per user; peak 92.7 GiB reserved |
| 1,048,576 cache, 876,328-token prompt, adaptive prefill (MAX_CHUNK 8192) | both facts (10% and 90% depth) answered; first token 352.0 s (2,490 tok/s); 214 chunks; decode 9.95 ms/token |

## 13. What this is and is not

- **Is:** a true TP=2 runtime for a 2.499 bpw mixed-bitrate EXL3 checkpoint of a 100B-class MoE model, fitting
  two 96 GB cards with speculative decoding, at 91–185 tok/s single-user decode at 24k context depending on content,
  a 5M-token shared KV cache (8 users × 131k), a single 876k-token prompt, and prefix caching for repeated prompts.
- **Is not:** a claim to beat every other runtime. As a reference point published by a third party on the same
  hardware class for a *different* checkpoint and configuration (diffbot, EXL3 3 bpw, fp8 KV, their own
  drafter): 156.9 tok/s single-stream at 2k context, 362.9 tok/s aggregate with 4 streams at 46k, and
  11.2–11.5k tok/s fresh 40k prefill warm ([public-number survey](notes/EXL3_SPEED_SURVEY.md)). Different
  checkpoint, different KV precision, different batching policy. This runtime does not win on multi-user
  throughput or prefill.
- **Known limits**, all measured: decode attention runs per user when several users are active (total levels off
  at ~250 tok/s); FP8 KV breaks capture; the 8-bit prefill all-reduce fails the coherence gate; cold prefill
  2.5–3.0k tok/s.
