# Overnight run summary — TP=2 recipe (2× RTX PRO 6000 Max-Q)

Goal set by owner: **300 tok/s decode (DSpark) and 8000 tok/s prefill**.
Start of session: ~10 tok/s (handoff baseline, Python/TCP remote-expert design).

## Final measured numbers (all on branch `recipe/2x-rtxpro6000-tp2`, both GPUs, both ranks bit-identical)

| Configuration | Result |
|---|---|
| **Decode, plain** (bsz1, greedy, 128 tok) | **13.20 / 13.27 / 13.31 ms/tok → 76 / 75 / 75 tok/s** |
| **Decode, DSpark K=5** | **8.11 / 6.83 / 9.29 ms/tok → 119 / 135 / 101 tok/s** |
| — acceptance / tokens per verify round | 0.716 / 0.664 / 0.778 ; 3.09 / 3.30 / 2.58 |
| **Prefill 2048 tok** (single chunk) | 580.6 ms → **3527 tok/s** |
| **Prefill 4096 tok** (single chunk) | 989.2 ms → **4141 tok/s** |

Progress: decode 10 → 76 tok/s plain (7.6×), 10 → 101–135 tok/s with DSpark (10–13×). Prefill 2925 → 3527 (2048) and 3660 → 4141 (4096).
Targets (300 / 8000) NOT reached: decode is ~2.3× short, prefill ~2× short.

## What produced the gains (in order of impact)
1. **Native P2P one-shot all-reduce** (cudaIpc device arena, peer writes + release/acquire flags): 4.9 µs per 20 KB call
   vs NCCL 11.8 µs. Enabled by the reboot fix (Resizable BAR + `iommu=pt`; before it, GPU-to-GPU SM access returned garbage).
2. **Piecewise CUDA graphs** (everything except the attention core captured): 43 → 21 ms/tok early, later 13.2 ms/tok.
3. **DSpark K=5 with verify-width graphs** (seq 1..8 per-shape segment cache): 13.2 ms/tok → 6.8–9.3 ms/tok.
4. **Graph-merge across block boundaries** (block N's MLP graph folded into block N+1's attn graph A): 80 → 40 replays/step,
   13.6 → 13.0–13.3 ms/tok, token-identical including the DSpark verify path.
5. **Pre-sharded TP=2 checkpoint** (offline 50/50 shard of every expert/head/layer, verified byte-exact) + a dedicated
   per-rank runner: this is what makes true TP=2 (every expert split down the middle) possible at all.
6. **Mixed-K fused MoE** (per-expert bitrate tables, decode + prefill) and **per-stage tile geometry** (A narrow, B wide
   at bsz>1): verify-width MoE −10%.
7. **Single-chunk prefill** (removed the recurrent last-page split): 2048 TTFT 701 → 580.6 ms (−17%).
8. **Hybrid transport** (P2P < 64 KB, NCCL ≥ 64 KB) and **allocation-churn fix** (cached attention output buffers).

## What is blocking the remaining 2×
- **Decode**: the attention core is still eager Python (~4–5 ms/step: 490 `cudaLaunchKernel` + 40 `cudaGraphLaunch` per
  step). Capturing it is THE remaining lever. It is blocked on the existing device-driven attention path
  (`EXL3_DSV4_B1_BATCH=1`) not being token-exact (diverges at token 11 / 72–73 vs eager, and is slower). That divergence
  must be root-caused first; the last fix (roped `pool_r` write-back) only moved it later.
- **Prefill**: GPU 93% busy, split at 2048: MoE 30.6% (fused `exl3_moe`, 64-row instance), all-reduce 22.3%
  (PCIe-bound, needs a 2-way row split to overlap), dense GEMMs 21.3%, other 12.7%, HC 8.5%, MoE-recon 4.5%.
- **MoE at the verify width**: 282 µs/layer per GPU ≈ 690 GB/s vs 1.65 TB/s peak. Split-K, tile geometry, register and
  cp.async rings, and a fused single-launch A→B were all implemented and measured — every one was neutral or slower.
  What remains needs a deeper restructure (warp-specialized producer/consumer, bulk async copies) with hardware counters.

## Also measured and rejected (documented in the notes, with numbers)
x-fan fusion (−3% but changes tokens from position 55), HC fold at prefill (inactive under TP), P2P for prefill
(NCCL is better at 40 MB), `FUSED_ROWS_WIDE` > 256, `EXL3_DSV4_B1_BATCH=1` (+2.3 ms/tok, not token-exact),
event-based sampler sync, index-math hoisting (<0.15 ms/tok ceiling), fused A→B single launch (−0.3× at every bsz).

## Repo state
- Branch `recipe/2x-rtxpro6000-tp2` in `/home/treadall/recipe-2xrtx-tp2` holds everything (committed, not pushed).
- Side branches also committed: `tp2/moe-kernel`, `tp2/p2p-allreduce`, `tp2/dspark`, `tp2/prefill`, `tp2/attncap`.
- Notes: `qmesh/notes/*.md` per workstream; `OVERNIGHT_STATUS.md` in the repo root has the full log.
- Recipe A (GPU0 + GX1) and its checkpoint/branch were not touched. GPU2 untouched.

## Round 2 (after community research) — final consolidated numbers
Branch `recipe/2x-rtxpro6000-tp2` now contains everything: main + MoE tiles + DSpark + prefill + attncap.
| configuration | ms/tok | tok/s |
|---|---|---|
| plain decode (`COMM=p2p`, `EXL3_WHOLE_GRAPH=1`, WARM=1) | **13.18 / 13.16 / 13.15** | 76 |
| DSpark K=5 | **8.22 / 6.85 / 9.39** | **118 / 135 / 101** (AL 3.09/3.30/2.58, acc .716/.664/.778) |
| prefill 2048 (single chunk, fused HC) | 556–565 ms | **3624–3682** |
| prefill 4096 | 940–942 ms | **4346–4355** |

### What round 2 added
- **Community research** written up in `qmesh/notes/COMMUNITY_RESEARCH.md`: the 8×V100 reference engine for the *same*
  model (philbert440 franken-v100 / 1Cat-vLLM PR #1243) and single-5090 SSD-streaming kits. Their plain decode is
  30.5 tok/s and DSpark 65 tok/s on 8 GPUs — we are 2.5× faster than that on 2 GPUs. Their engine's techniques are
  catalogued there (packed FP4 KV, per-layer attention graphs, decoder replay in prefill, fused HC, tiled prefill MoE,
  Engram in RAM with hugepages, DSpark tap fix).
- **DSpark tap saturation**: real in our tree (taps hit ~271k vs fp16 max 65504 at sequence positions 0/2) but the
  scale fix (`EXL3_DSPARK_TAP_SCALE`, default off) is throughput-neutral here (our clamp already avoids their
  inf/NaN cascade). AL decomposition: first draft 85.7%, later drafts 57–64% conditional.
- **Graph-merge across block boundaries** (block N's MLP graph folded into block N+1's graph A): 80 → 40 replays/step,
  13.6 → 13.2 ms/tok, token-identical including the DSpark verify path.
- **Device-driven attention path fixed** (`EXL3_DSV4_B1_BATCH=1`): root cause was the batched compressor projecting
  wkv/wgate with a plain GEMM while the eager path used a 2-expert exl3_mgemm (fp16-ulp difference flipping the newest
  pool row). Now **bit-identical** on 3 prompts × 128 and on a 1134-token prompt crossing index_topk. This is the
  prerequisite for capturing the attention core. It is still slower than the scalar eager path (15.9 vs 13.2) because
  the body is still eager — capture is the remaining step.
- **Prefill**: single-chunk default (2048: 701 → 556–565 ms, −20%), fused HC partials (bit-exact), and a measured map of
  the remaining gap: the prefill MoE's dominant 64-row instance is **register-file-bound** (255 regs/thread, 1 block/SM,
  16.7% warps active, 8.4% DRAM); K-tile 16, min-blocks hints, and lowering the fused threshold all measured worse.
  The next real step there is a kernel redesign for occupancy (split the 64-row accumulator), not a knob.

### Where the remaining 2× is
- decode: the attention body still runs eagerly (host gap ~4–5 ms/step). Capture it now that the body is bit-identical.
- prefill: MoE occupancy (30.6% of prefill at 8.4% DRAM) and the PCIe-bound all-reduce (22%, at hardware limit).
