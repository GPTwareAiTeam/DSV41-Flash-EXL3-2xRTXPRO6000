# EXL3 speed survey (DeepSeek-V4.1-Flash-1017, 2.565 bpw, TP=2)

> Raw working notes from the development runs, kept because they hold the measurements. Host paths, container names and run tags refer to that machine; the results themselves are restated with conditions in [`../RESULTS.md`](../RESULTS.md).

Scope: published EXL3 (ExLlamaV3 trellis) inference results and the techniques behind them. Read-only. Extends `COMMUNITY_RESEARCH.md`; does not repeat it.

Reference hardware for us: 2× RTX PRO 6000 Max-Q (sm_120, 96 GB, ~1.65 TB/s), PCIe, no NVLink.
Targets: 300 tok/s decode, 8000 tok/s prefill.

Conventions: `[INFERENCE]` = not directly observed in a source. `[secondary]` = number seen only in a search/relay snippet, not in the primary post. `[date unverified]` = no date in source.

---

## 1. Data-point table

Closest match to our box is the diffbot row (same GPU class, TP2, PCIe, no NVLink). Other rows are different bandwidth regimes (Spark LPDDR5X, V100, 5090, 3090) and are listed for completeness, not as direct targets.

| # | Hardware | Model / quant | Engine / fork | Spec. method, K | AL | Decode tok/s | Prefill tok/s | Context | KV dtype | Source | Date |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2× RTX PRO 6000 (TP2, PCIe, 300 W cap, 9995WX, 8-ch DDR5) | DeepSeek-V4.1-Flash 3-bit calibrated LDLQ (~204 GB routed experts) | diffbot custom (`xmoe`, `xmoe_do`, `xmoe_do_i8`, `fp8ar`) | DSpark, k=5 (default), k=7 code head | n/r | 1-stream agentic replay 224.1 (default) / 261.8 (`CPU_EXPERTS=1`); fixed 2K 156.9 / 179.0; fixed 46K 172.8 / 210.1 | fresh 40K 11,157–11,467 (default), 11,624–12,182 (CPU); cold first 40K 7,700–8,000 | 40K–1.04M | FP8 (1.07M tokens) | github/HF `diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000` `docs/SPEED.md`, `docs/CHANGELOG.md` | 2026-10-25 (latest in CHANGELOG) |
| 2 | same | same | same | same | n/r | 4-stream per-stream 41.2 / agg 110.4 (default); 50.6 / 127.8 (CPU). Fixed 46K 4-stream agg 362.9 / 406.2 | — | — | FP8 | same `docs/SPEED.md` | 2026-10-25 |
| 3 | 8× RTX PRO 6000 (server, PCIe 5) | same family | diffbot | DSpark k=7 code head vs released k=3 | n/r | code 1-stream 396–416 (k=7) vs 278.6 (k=3) | — | — | — | `recipe/results/RESULTS.md` (same repo) | 2026-10-24 |
| 4 | RTX PRO 6000 (node count not re-verified) | DeepSeek V4.1 Flash, EXL3 K3 experts via bank | SSHdotCodes `grouped_exl3` kernels, MXFP4 baseline | DSpark, acceptance ~0.86 | ~0.86 accept | mean 49.13 vs native MXFP4 31.65 (1.55×) | — | benchmark prompts | n/r | github.com/SSHdotCodes/deepseek-v4.1-flash-pro6000 `docs/exl3.md` | 2026-10-23 |
| 5 | 2× DGX Spark (GB10, SM121a, ~273 GB/s) | DeepSeek V4.1 Flash, EXL3 (MCG, 2.0 bpw) | sfxnz vLLM fork | DSpark-3/5; L.A.I.L metric | n/r | tg32@512 20.4; tg32@64k 26.6 (baseline); after round 35 coop kernels L.A.I.L 33.95→43.17; structured c=1 83.81, c=2 156.29 (round 36) | pp@4k 683, pp@16k 695, pp@64k 703 | 4k–64k | `--kv` 8 GiB pool = 2.29M tokens | github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark `results/RESULTS.md`, `README.md` | 2026-09-18 to 09-27 |
| 6 | 2× DGX Spark (SM121a) | DeepSeek V4.1 Flash, EXL3 (MiaAI pack) | MiaAI-Lab fork of ExLlamaV3 1.4.5 + cooperative MoE (pinned `02aef45c`) | DSpark, fixed K | n/r | poetry 23.62→29.26 (+23.9%); coding 38.76→42.96 (+10.8%); incident 31.92→41.26 (+29.3%); C2 aggregate 45.87→61.06 | 32K uncached 1137.76→1135.35 (−0.2%) | 32K | n/r | github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks `docs/cooperative-moe.md` (recipe 979e68a6) | 2026-09-14 |
| 7 | 2× DGX Spark | GLM-5.3 Flash EXL3 4bpw + DFlash2 k=7 | MiaAI TensorFold v1.4 | DFlash2 k=7 | n/r | ~63 single structured; 155 at 4 sessions | 874 | 900k ctx | n/r | X @MiaAI_lab https://x.com/MiaAI_lab/status/2093270924321530052 (via relay) | 2026-08-28 |
| 8 | 2× DGX Spark | GLM-5.3 Flash EXL3 4bpw vs NVFP4 | EXL3 (Tech2Wild comparison) | none stated | n/r | EXL3 c1 61.5 (NVFP4 64.0) | cold 1.6K 684; 211K 1,752 (NVFP4 2,763 at 211K) | 1.6K–211K | n/r | X @Tech2Wild https://x.com/Tech2Wild/status/2094943002225713571 (via relay) | 2026-09-02 |
| 9 | 1× RTX 5090 (32 GB), 128 GB host RAM | DeepSeek 1017 EXL3, SSD streaming, 22 GiB GPU expert cache | acrogenesis custom | none | n/r | target >30 tok/s single seq **not reached** per 20261018 ledger; 22 GiB cache 5.71% faster than 16 GiB | n/r | n/r | n/r | github.com/acrogenesis/deepseek-v41-dev-ctr `results/20261018-*/METHOD.md` | 2026-10-18 |
| 10 | 1× RTX 3090 24 GB + offload | DeepSeek EXL3 (sybil-solutions) | dsv41-flash-offload | n/r | n/r | 19.3 / 24.0 / 26.6 at 1 / 2 / 4 streams | 690–790 at 8K–261K | 8K–261K | n/r | github.com/sybil-solutions/dsv41-flash-offload | [date unverified] |
| 11 | 1× AMD R9700 gfx1201 ×2 (not CUDA) | Qwen3.8-flash-next EXL3 experts 4.05 bpw | radiance (AMD) | MTP off, C=1 | n/a | 10.70 ms/tok (vs 11.16 ms baseline fp8/iq4) | 8K 1,734 (vs 1,542); `EXL3_PREFILL_STAGE=auto` doubled prefill on PCIe x4 | 8K | n/r | github.com/Dyluhn/qwen3.8-flash-next-exl3 | n/r |
| 12 | 8× V100 (PR #1243, philbert440 build `ds41-final-20261025`) | DeepSeek V4.1 Flash franken-v100 EXL3 | 1Cat-vLLM PR #1243 | DSpark k=5 | 4.35 (code) / 4.38 | plain 30.5; DSpark 65 (code 64.9); tap fix 43.5→62.7 (code) | solo 2,985 / 3,231 / 3,194 at 8K / 32K / 128K | 8K–128K | FP4 packed KV | philbert440 repo + PR #1243 (in COMMUNITY_RESEARCH) | 2026-10-25 |
| 13 | 2× RTX 5090 | GLM-5.3 321B EXL3 2.05 bpw | llamaperf | MTP depth 3, 8-bit KV | n/r | 64.8 | 1,973 (13.7k on 27 GB claim) | n/r | 8-bit | llamaperf (in COMMUNITY_RESEARCH) | n/r |

Notes on rows:
- Row 2 is diffbot's 4-stream measurement on the same config as row 1. Row 9 is one 5090 (not two). Row 1 quant is 3-bit calibrated, not 2.565 bpw.
- Row 6 and 7 pack/quant details: sfxnz is 2.0 bpw MCG, MiaAI pack not specified in the doc I read.
- Secondary (search snippets, not browsed): Cruz Strix Halo 40.0 (2026-09-30); CodeBuilder 124 tok/s 8×3090 Qwen 3.8 (2026-10-27) `[secondary]`; TandemLLM Flint 128–137 on one Spark (2026-10-25) `[secondary]`; Grok relay "dual Spark 2.9 bpw EXL3 TensorFold ~73 prose / 110 code" (2026-10-25) `[secondary]`; Shabarkin 6-bit EXL3 Qwen3.8-27B on A100 70–87 (2026-10-16) `[secondary]`.
- Our own measurements (task brief): plain decode 13.15 ms/tok (76 tok/s); DSpark K=5 6.85–9.39 ms/tok (118–135 tok/s), AL 2.6–3.3; prefill 3,600–3,700 tok/s @2048, 4,300–4,400 @4096. Not directly comparable to rows 1–2 (their prompts are agentic replay or a 40K prefill with different chunking).

### Highest single-stream numbers
- **Decode**: diffbot 224.1 tok/s agentic replay (default), 261.8 with CPU expert runtime. This is a replay mostly of tool-call replies with DSpark; not a like-for-like code benchmark. The apples-to-apples fixed-prompt decode is 172.8 (46K) / 156.9 (2K), default. Their 8× profile reaches 396–416 on code with k=7 (a server, not our box).
- **Prefill**: diffbot fresh 40K 11,157–12,182 tok/s (2×RTX PRO 6000). This is the highest EXL3 prefill number found on a 96 GB Blackwell pair. Their own cold first-40K number is 7,700–8,000 (kernel compile).

---

## 2. Ranked secrets

Ranked by expected value for us. "Have" column is the OUR_HAVE check (see section 3).

| Rank | Technique | Who | Attributed delta | Have? |
|---|---|---|---|---|
| 1 | **8-bit prefill all-reduce** (`kernels/fp8ar`, int8 values + bf16 scale per 32, P2P push, bit-identical across ranks; chunks ≥256 rows; decode keeps exact AR) | diffbot | 16K AR 4.0→1.8 ms (dev A/B); 4-session wall −11% (dev build, 8-bit AR); prefill fully attributed in their changelog | **No.** Our `p2p_oneshot.cu` is fp32 fixed-order exact; no 8-bit variant found in grep. |
| 2 | **Int8 tensor-core decode-once prefill MoE** (`xmoe_do_i8`: persistent blocks, one barrier per K slab, per-(row,128) activation scales; each host expert read once per chunk) | diffbot | int8 kernel: −8% 4-session wall, +6% 40K prefill (dev build) | **Partial.** We have a tiled prefill MoE in the plan (COMMUNITY_RESEARCH item 4) but not int8 tensor-core. |
| 3 | **Length-aware indexer top-k** (`DS41_ROWTOPK=1`; exact; kernel 2,054 µs→372 µs on 384 rows × 1M padded width) | diffbot | 4-stream 46K +3%; exact | **No.** No such kernel found in our tree; COMMUNITY_RESEARCH item 6 is the still-open "graph-capturable fused decode indexer". |
| 4 | **Decoder replay trim in prefill** (layers 20–39 hold 128-token SWA KV; chunks far from prompt end keep 1 placeholder row) | diffbot, philbert440 ("decoder replay in prefill") | 1.10/3.26/6.49/11.95 s off → 1.08/3.11/6.19/11.51 s on (12K/34K/70K/130K), i.e. 2–4%; replay trim 3.7–4.6% on multi-chunk prefill | **No** (COMMUNITY_RESEARCH C lists it as planned). |
| 5 | **Cooperative MoE** (fixed-K specialized kernels; not bit-exact to stock; prefill unchanged) | MiaAI (Spark) | decode +11% to +29% on 3 prompts; C2 aggregate +33%; prefill −0.2% | **Partial** — `MoEKernelTP2.md` has fixed-order fold + folded shared expert; we're bit-exact by design, so this trade-off is not directly copyable. |
| 6 | **Per-expert CUDA-graph-capturable grouped EXL3 GEMV** (`grouped_exl3` decode 1–8 rows/expert; `grouped_exl3_prefill` ≤256 rows; batch-invariant) | SSHdotCodes | decode 31.65→49.13 mean (1.55×) vs MXFP4 (different quant + kernels); cold weight timing decode 0.91–0.93× MXFP4 time | **Partial.** We have mixed-K fused MoE + per-expert K tables; no batch-invariant claim tested. |
| 7 | **Expert residency tiers**: pinned host RAM for least-used experts (5,296 of 15,360 routed) + 512 VRAM slots/GPU + second CUDA stream for cold experts; optional `CPU_EXPERTS=1` inside graphs | diffbot | CPU runtime 224→262 (1-stream agentic), 110→128 (4-stream agg) | **Partial/N/A.** We fit all experts in VRAM at 2.565 bpw (`[INFERENCE]`: our footprint is smaller than diffbot's 3-bit ~204 GB routed); the cold-tier mechanism is not needed unless VRAM is tight. Hybrid transport is in OUR_HAVE. |
| 8 | **Fixed-K specialization + draft length, not drafter head** | diffbot | k=7 vs k=3 on code: draft length gives nearly all of +43%; code head only 0–4% | **Partial**: DSpark K=5 with width-keyed graphs is ours; K=7 not tested in our tree. |
| 9 | **DSpark tap scale** (`EXL3_DSPARK_TAP_SCALE=1/64`) | philbert440 (V100): tap fix 43.5→62.7 code; DS-Spark MiaAI | On our box: AL 3.09/3.30/2.58 → 2.84/3.49/2.68 (mixed, neutral) | **Yes**, shipped default-off (commit 4a02976e). Not a win here. |
| 10 | **Engram WILLNEED read-ahead + NVMe record size 16K** | sfxnz | prefill novel 8k 184→838; 32k 231→811 | **No** (our Engram tables are replicated per rank and streamed from disk per `TP2Prep.md:66`; no read-ahead found). Matters only when Engram is cold. |
| 11 | **Quality gate**: KL + GSM8K/HumanEval + noise floor 0.005–0.008 | diffbot | KL neutral 0.052 / 0.031 / 0.011 / 0.034 | **Yes** in our notes (`MixedK.md`); keep. |
| 12 | **Calibrated LDLQ 3-bit experts** (coolbho3k method; AtomicChat calib, 320×4096 windows) | diffbot | no speed claim; quality base | `[INFERENCE]` not relevant to our 2.565 bpw pack unless we re-quantize; uncalibrated PixelML K=3 flagged as quality risk, do not copy. |
| 13 | **Packed FP4 / 8-bit KV** | philbert440 (FP4 packed), llamaperf 8-bit | n/r | **No** (we are fp16 KV, per COMMUNITY_RESEARCH). |
| 14 | **Fused HC prefill** | ours (cf6946a1) | 2048 3502.6→3542.2; 4096 4131.5→4218.5 tok/s | **Yes.** |
| 15 | **Piecewise + merged CUDA graphs; width-keyed DSpark graphs** | ours | n/r in community sources | **Yes**: this is in OUR_HAVE. Community equivalents: diffbot `kernels/xmoe` flat decode scheduler; acrogenesis per-token graph of hyperconnection mixer (optional, inconsistent). |

Lower-confidence items seen but not ranked: acrogenesis LFU expert eviction (residency only); diffbot's Engram int4 + fp16 per-32 scale (111 GB, pinned host UVA read in graphs).

---

## 3. OUR_HAVE cross-check

| Item (from brief) | Status | Evidence |
|---|---|---|
| Mixed-K fused MoE | Have | `docs/tp2/notes/MixedK.md`; `MoEKernelTP2.md` |
| Per-expert K tables | Have | `MixedK.md` (option (b) rejected, template switch chosen) |
| Piecewise + merged CUDA graphs | Have | `docs/tp2/notes/GraphCapture.md`; commit 3068d30c (`piecewise: capture B1 attention body inside the block segment`) |
| P2P one-shot all-reduce | Have (fp32 fixed-order) | `exllamav3/exllamav3_ext/parallel/p2p_oneshot.cu:171–198` (arena, cudaIpc); `docs/tp2/notes/P2PAllReduce.md` |
| 8-bit prefill all-reduce | **Missing** | No 8-bit/int8 AR found in `parallel/` (only `all_reduce.cu` generic path). |
| DSpark K=5 with width-keyed graphs | Have | `TP2Runner.md`, `GraphCapture.md` |
| Single-chunk prefill (`max_chunk_size=2048` in bringup) | Have, **but** chunk size 2048 is small vs diffbot's larger replay chunks | `docs/tp2/notes/TP2Bringup.md:6` |
| Hybrid transport | Have | `P2PAllReduce.md` / `TP2Runner.md` |
| Packed FP4 KV | Missing (fp16) | COMMUNITY_RESEARCH |
| Decoder replay in prefill | Missing | COMMUNITY_RESEARCH C |
| Tiled prefill MoE | Planned, not in tree | COMMUNITY_RESEARCH item 4 (our prefill MoE 30.6% of prefill) |
| Fused decode attention / indexer | Missing (indexer fused decode open) | COMMUNITY_RESEARCH item 6 |
| Length-aware indexer top-k | Missing | not found in tree |
| Fused HC | Have | cf6946a1 |
| DSpark tap scale | Have, default off, neutral | `TP2Runner.md:73`; commit 4a02976e |
| Engram read-ahead | Missing | `TP2Prep.md:66` (tables streamed, no WILLNEED) |

Note: grep found no DeepSeek model source under `exllamav3/models/` in this tree; the DeepSeek path is in `qmesh/` and the overlay. Indexer / top-k claims above are therefore "not found in tree" and should be double-checked by the main agent against `qmesh/tp2/` before any change.

---

## 4. Top-5 highest-expected-value changes for us

1. **8-bit prefill all-reduce (bf16 scale per 32, P2P push, chunks ≥256 rows; decode exact).** Diffbot's 16K AR 4.0→1.8 ms is the largest single prefill delta with a published A/B. Expected gain `[INFERENCE]` 1–3 ms per 16K chunk, roughly ~5–10% prefill at our 4K chunking. Lossy: gate on KL against our noise floor (0.005–0.008) and on bit-identical rank consistency. Owner: main; touches `parallel/`.
2. **Length-aware indexer top-k (exact).** Diffbot shows a 5.5× kernel speedup (2,054→372 µs) at 1M padded width, +3% 4-stream. Our fused decode indexer is still open (COMMUNITY_RESEARCH item 6). `[INFERENCE]` gain is small at our 4K contexts but grows with context. Check first whether our indexer pads to full width at each step.
3. **Decoder replay trim for prefill (SWA layers keep 128-token KV; far-from-end chunks carry one placeholder row).** 2–4% measured on diffbot at 12K–130K; `[INFERENCE]` similar for us. Low risk, exact in their A/B. Owner: prefill path; coordinate with `CaptureFullBlock` if it touches the same graphs.
4. **Tensor-core / int8 grouped prefill expert GEMM (decode-once per chunk).** sfxnz measured prefill MFU ~6.5% on GEMV-per-row kernels and called tensor-core grouped dequant GEMM the real prefill lever; diffbot's `xmoe_do_i8` gives +6% 40K prefill on top of a decode-once scheme. Our prefill MoE is 30.6% of prefill time (COMMUNITY_RESEARCH). `[INFERENCE]` this is the path from 3.6k toward 8k prefill, but it is the biggest build and must be gated on A/B and KL.
5. **Draft length K=7 with DSpark (keep K=5 graphs; add a K=7 width-keyed graph).** Diffbot shows draft length gives nearly all of their +43% code gain at k=7 vs k=3; the code head contributed 0–4%. Our K=5 AL is 2.6–3.3 vs philbert 4.35 on different content. `[INFERENCE]` K=7 is the cheapest experiment (no new kernel), but AL falls off once acceptance is below ~3, so measure AL first on our prompts before building anything.

Not in top-5 and why: tap-scale (neutral here); cooperative MoE (not bit-exact; prefill unchanged, decode gain is on 2 Sparks with different bandwidth); expert residency tiers (we fit experts in VRAM at 2.565 bpw; `[INFERENCE]` only relevant if our VRAM budget is tight); Engram read-ahead (only matters when cold).

---

## 5. Checks for the main agent (not run by me)

- Re-verify the diffbot and sfxnz numbers against their source files before quoting them externally (all read from primary docs; the diffbot CHANGELOG and RESULTS are the authority).
- Confirm our indexer top-k behavior in `qmesh/tp2/` before scheduling change 2.
- Bench the 8-bit AR change (change 1) with our own `bench_prefill.py` at 2K/4K/16K and a KL run; expect numbers only from that bench.
- Measure AL at K=7 on our prompt set before any K=7 kernel work (change 5).
- Do not claim any community number as ours until reproduced on our box.

## 6. diffbot / same-hardware reference — component-level comparison

Sources (read in full): `diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000` (HF) `docs/SPEED.md` (2026-10-25), `docs/FIDELITY.md`, `docs/CHANGELOG.md`, `recipe/results/RESULTS.md`, `recipe/serve/serve.sh`, `recipe/README.md`, `recipe/kernels/fp8ar/fp8ar_kernels.cu`, `recipe/kernels/rowtopk/rowtopk.cu`. Cross-hardware: sfxnz `results/RESULTS.md` (DGX Spark GB10).

**Identity.** diffbot 2× is a vLLM `nightly-5555a2d8` stack with custom sm_120 kernels and an EXL3 expert path. It is NOT stock ExLlamaV3. Rows 1–2 of §1 are therefore not a pure EXL3-engine comparison. Hardware matches ours (2× RTX PRO 6000, PCIe, 300 W cap, Threadripper PRO 9995WX).

**Parallelism.** TP2 for attention and dense. Routed experts are NOT EP2. 2× keeps 5,296 of 15,360 routed experts in pinned host RAM (`plan-1017-agent30-c512.json`) plus a 512-slot VRAM cache per GPU. Whether the two GPUs shard or replicate the VRAM experts is not stated in the docs read: [INFERENCE] replicated. Their 8× profile is TP4 attention + EP4 experts, all in VRAM.

**Quant.** Diffbot routed experts: EXL3 3-bit MUL1 (coolbho3k LDLQ, calibration corpora from AtomicChat, 320×4096-token windows). Drafter experts: EXL3 3-bit MCG, uncalibrated. Ours is 2.565 bpw. bpw is not comparable across the two.

### 6.1 Diffbot published numbers (2× TP2, `docs/SPEED.md`)

| Metric | Diffbot 2× (current) | Prior release | Source |
|---|---|---|---|
| Fresh 40K prefill (tok/s) | 11,157–11,467 (warm); 7,700–8,000 cold | 9,031–9,125 | SPEED.md |
| CPU-runtime prefill (tok/s) | 11,624–12,182 | — | SPEED.md |
| Decode 2K, 1 stream (GPU / CPU experts) | 156.9 / 179.0 | 129.7 | SPEED.md |
| Decode 46K, 1 stream | 172.8 / 210.1 | 114.4 | SPEED.md |
| Decode 46K, 4 streams aggregate | 362.9 / 406.2 | 215.9 | SPEED.md |
| Agentic replay 1 session | 224.1 / 261.8 | — | SPEED.md |
| Agentic replay 4 sessions, aggregate | 110.4 / 127.8 | — | SPEED.md |
| Power (RAPL) | ~815 W GPU / ~900 W CPU | — | SPEED.md |

Our measured numbers: plain decode 13.15 ms/tok (76 tok/s); DSpark K=5 118–135 tok/s; prefill 3.6–3.7k @2048, 4.3–4.4k @4096. Diffbot decode is about 2.3× ours at 2K 1-stream, and our prefill is about 3× below theirs. These are not same-day, same-engine numbers. [INFERENCE] much of the gap is architectural (their DSpark draft, our single-chunk prefill, their replay trim, their 8-bit AR), not hardware.

### 6.2 Decode components (diffbot knob → measured delta → our tree)

| # | Diffbot component | Measured delta (diffbot's own A/B, dev build) | Our tree | Status |
|---|---|---|---|---|
| D1 | Flat decode MoE loop: 128-wide K tiles, one barrier per stage, swapped MMA operands | +13% decode at 6 rows/verify; MoE kernel time −31–36%; +3% agentic | `MoEKernelTP2.md`: bsz1 54.44 µs vs ≤45 target | partial |
| D2 | VRAM expert cache 128–512 slots (`EXL3_HOST_CACHE=512`, `HOT=1`) | +7–10% at 1 stream, +15–20% at 4 | `TP2Prep.md:66` (Engram streams from disk; no expert cache) | absent |
| D3 | Fused decode routing prep (`EXL3_FUSED_PREP`, `DS41_FUSE_ROUTER/HCPREP/IDX/OPROJ`) | +7–8% (fixed bench); four bit-exact fusions +6% | `TP2Runner.md:65`: other 3%, `routing_ds3_topk` 0.13 ms | absent as a set; the topk is small |
| D4 | rowtopk: one CTA per row, radix select, valid-prefix only (`DS41_ROWTOPK=1`) | kernel 2,054 → 372 µs (384 rows × 1M width); 4-stream 46K +3.2% | ours uses a generic top-k; `TP2Runner.md:65` (0.13 ms) | absent, but small at our width |
| D5 | CPU host experts in decode (`EXL3_HOST_OVERLAP_SMS=32`) | +17% (243.3 vs 208.3) | no equivalent; our experts all in VRAM at 2.565 bpw [INFERENCE] | n/a unless VRAM-tight |
| D6 | Drafter experts all in VRAM | +5–9% at 4 sessions | our drafter placement not established in notes | unknown |
| D7 | DSpark K=5 + width-keyed graphs (cudagraph sizes 1,2,4,6,8,12,16,24,32,48) | not separately quantified in SPEED.md | we have K=5 width-keyed graphs (`GraphCapture.md`) | have |
| D8 | Hand-tuned decode loop | +3.3% | — | absent |
| D9 | FlashInfer PCIe IPC AR (`DS41_CAR_PCIE_IPC`) | +3.6% decode | our P2P one-shot AR `p2p_oneshot.cu` (fp32 fixed-order) | have a similar path |
| D10 | fp8 AR for decode (`DS41_FP8AR_MIN_TOKENS=256`; below 256 rows exact) | prefill-only; decode unchanged | not in tree | absent (prefill only) |

Decode attribution, ranked by diffbot's own A/B (they measure against their prior build, not against naive TP2):
1. Host-expert overlap and cold tier (D5): +17% decode at 1 stream (but their cold tier exists because 3-bit experts ≈204 GB exceed VRAM; [INFERENCE] not applicable to us if 2.565 bpw fits).
2. VRAM expert cache (D2): +7–10% at 1 stream, +15–20% at 4.
3. Flat MoE loop (D1): +13% decode.
4. Fused routing prep and four decode fusions (D3): +6–8%.
5. FlashInfer PCIe IPC AR (D9): +3.6%.
6. Hand-tuned loop (D8): +3.3%.

### 6.3 Prefill components

| # | Diffbot component | Measured delta | Our tree | Status |
|---|---|---|---|---|
| P1 | 8-bit prefill AR (`kernels/fp8ar`, one persistent kernel, int8 + bf16 scale per 32 elements, peer P2P push, fp32 add, bit-identical on both ranks) | AR 16K: 4.04 → 1.79 ms; fresh 40K +4–6%; agentic CPU 4-session wall −11% (197→175 s); TTFT −20% | `p2p_oneshot.cu:171–198` fp32 arena; `all_reduce.cu` generic, no 8-bit codec | absent; biggest prefill item we can copy |
| P2 | int8 decode-once MoE GEMM (`kernels/xmoe_do_i8/dof_gemm.cu`, ~1.06% GEMM error vs fp16 0.02%) | fresh 40K +6%; agentic 1-session 230.6→224.0 s; 4-session 185.8→170.5 s; decode unchanged | none (no DO kernel in tree) | absent |
| P3 | DO (decode-once) xmoe kernel: each trellis tile decoded once per 128-row job | 1.3–1.8× faster per expert at 16–1,024 rows; cold read-once: launch −42% (4K), −61% (16K) | no tiled prefill MoE in tree (only COMMUNITY_RESEARCH item 4) | absent |
| P4 | Replay trim (`patches/vllm/v41-replay-final-window.diff`): skip replay of layers 20–39 for chunks >128 rows the final window cannot read | fresh 12K/34K/70K/130K: 1.10/3.26/6.49/11.95 s → 1.08/3.11/6.19/11.51 s; DSpark acceptance on multi-chunk prompts 3.52→3.62 | our `max_chunk_size=2048` (`TP2Bringup.md:6`); no replay trim | absent |
| P5 | Chunk size `MAX_BATCHED=16384` (diffbot) vs ours 2048 | not separately measured by diffbot | `TP2Bringup.md:6` | open: test |
| P6 | CPU host experts in prefill wall (`EXL3_HOST_OVERLAP_SMS_PREFILL=64`, `CPU_EXPERTS_PREFILL=1`) | −6/−9% wall; fresh 40K 9.2K→10.5K | n/a | not applicable at 2.565 bpw [INFERENCE] |

Prefill attribution, ranked by diffbot's own A/B:
1. Their large chunk plus DO kernel (P3 + P5): the bulk of the 9.0K → 11.2K jump; the DO kernel is the one diffbot ties to tile-reuse at 16K+ rows [INFERENCE for the split between P3 and P5].
2. 8-bit prefill AR (P1): +4–6% fresh prefill, and the agentic wall-time win is larger (−11%) because it removes AR latency from TTFT.
3. int8 DO GEMM (P2): +6% fresh.
4. Replay trim (P4): 1–6% on multi-chunk prompts.

### 6.4 Per-kernel ms

Diffbot publishes no per-kernel ms table in its docs. sfxnz (cross-hardware, GB10) gives per-step composition: Round 5 device-busy per decode step: p2b MoE ~35%, dense projections ~42%, NCCL AR ~11%, eltwise/topk/norm/MLA ~12%, sparse_mla 1.2%. Round 7 after COOP: p2b 48%, dense 28%, NCCL 17.5%. Our per-kernel numbers are in `TP2Runner.md:65` and `MoEKernelTP2.md`. [INFERENCE] the AR share is our next target, since ours is a fp32 AR and diffbot's is int8.

### 6.5 Implementable items for us (ranked by expected value)

1. **8-bit prefill AR** (P1). Port `fp8ar` quantize-push-dequant-add into `p2p_oneshot.cu`, gated at ≥256 rows, with exact fp32 path retained as fallback. Expected: AR 16K −2.2 ms per AR [secondary, diffbot's measurement]; prefill +4–6%. Gate: same-day KL pair within +3% per corpus.
2. **Int8 DO prefill MoE** (P2+P3). Requires a tiled decode-once MoE path. Our `MoEKernelTP2.md` work is the decode side; the prefill DO path is new work. Expected +6% fresh prefill [secondary]; larger if the tile-reuse win (1.3–1.8×) carries to our shape [INFERENCE].
3. **Replay trim** (P4). Needs a final-window read analysis for our DSpark replay. Expected 1–6% on multi-chunk prompts [secondary]. Check our `max_chunk_size=2048` against their 16384 first (P5).
4. **Expert VRAM cache** (D2). Only if our experts do not all fit VRAM at 2.565 bpw. [INFERENCE] they fit, so this is low priority.
5. **Fused decode routing prep and rowtopk** (D3, D4). +6–8% decode [secondary] for a set of fusions; rowtopk alone +3.2% at 4 streams. Our topk is 0.13 ms, so the rowtopk gain is small for us [INFERENCE].

Not copying: CPU host experts (D5, P6), e4m3 variants (failed KL), K=7 drafter (D-note: −14..−17% on 4-stream, positive only at 1 stream).

### 6.6 Checks for main

- Confirm our VRAM footprint at 2.565 bpw before ranking item 4.
- Bench P1 at 2K/4K/16K with our `bench_prefill.py` before any kernel change. Compare against the 3.6–3.7k @2048 and 4.3–4.4k @4096 baseline.
- Use the same-day KL pair (+3% limit) for any lossy change (P1 8-bit AR and P2 int8 are both lossy).
- Diffbot numbers are dev-build, same-day, A/B against their own prior build. Do not cite them as independent.
