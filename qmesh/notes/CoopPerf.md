# CoopPerf: fused MoE coop kernel and dense bsz-1 GEMV

Target: ≥1.3 TB/s effective on RTX PRO 6000 Max-Q (sm_120, GPU1).

## Ceiling

Streaming `__ldcs` uint4 read, 2 GiB, grid/occupancy sweep (4–16 blocks/SM × 256–1024 thr):
**1645–1651 GB/s, flat across occupancy. Ceiling ≈ 1.65 TB/s.** 1.3 TB/s = 79% of ceiling.
Probe: `/tmp/cp/bw.cu`.

## Method

- Per-kernel device time via torch.profiler (`/tmp/cp/kb.py`, 10 warm-up, 30 timed). The
  `bench_mixedk_coop.py` wall-clock numbers include host overhead and are not kernel times.
- Bytes: distinct routed experts only; A = gate+up, B = down. EP2 = experts 0..191 local.
- Hardware counters: `ncu` on host is blocked (`ERR_NVGPUCTRPERM`); container has no perf
  counter access. Conclusions below rest on timing only.

## Baseline (mixed K, EP2=0, auto geometry)

| shape | A µs | A GB/s | B µs | B GB/s | sum µs | bytes |
|---|---|---|---|---|---|---|
| bsz 1 full layer | 60.7 (WIDE=0) / 76.7 (auto) | 777 / 620 | 33.1 | 714 | 93.8 / 107 | 70.8 MB |
| bsz 6 full layer | 310 | 790 | 185 | 661 | 500 | 367 MB |
| bsz 1 EP2-half | 24.0 | 614 | 13.8 | 535 | 37.8 | 22.1 MB |
| bsz 6 EP2-half | 142 | 746 | 89 | 597 | 236 | 159 MB |

## Sweep (WIDE × KSPLIT, sum µs)

- EP2=0 bsz 1: WIDE=0 ≈ 92.0–92.6; WIDE=1 ≈ 106–110.
- EP2=0 bsz 6: WIDE=0 ≈ 488–490; WIDE=1 ≈ 454–456.
- EP2=1 bsz 1: WIDE=0 ≈ 37.8; WIDE=1 ≈ 70.8.
- EP2=1 bsz 6: WIDE=0 ≈ 230; WIDE=1 ≈ 223.
- KSPLIT 0/1/2/4/8: within ±1% everywhere. Scratch sizing caps effective split, so this knob
  is not a valid test; not pursued.
- Finding: the auto WIDE choice is wrong at bsz 1 full layer (WIDE=1 costs +15 µs on A).

## Prefetch depth (PF) probe (WIDE=0, EP2=0 and EP2=1)

Probe builds of the fork with PF = 2 / 4 / 8 (`-DMK_PF`, separate torch extension caches).
Registers (`cuobjdump -res-usage`, mixed A `<2,false>`): PF=2 → 64, PF=4 → 64 (baseline),
PF=8 → 87. The register jump lowers resident blocks per SM at 512 threads/block.

| case | PF=2 sum µs | PF=4 (base) sum µs | PF=8 sum µs |
|---|---|---|---|
| bsz 1 full layer | 92.6 | 92.0 | 109.6 |
| bsz 6 full layer | 484.6 | 488 | 601.0 |
| bsz 1 EP2 | 38.2 | 37.8 | 43.5 |
| bsz 6 EP2 | 226.2 | 230 | 284.6 |

Conclusion: deeper prefetch is worse (occupancy loss); PF=2 ≈ PF=4. Prefetch depth is not
the bottleneck and PF=4 stays.

## Dense bsz-1 GEMV (layer-3 attention shapes, 6 bpw, `BC_LinearEXL3`, mul1)

Rotating 512 MB+ of distinct weight copies so every call reads DRAM (single-copy loops read
from L2 and gave impossible >1.65 TB/s numbers; those were discarded).

| proj | in×out | MB | µs | GB/s |
|---|---|---|---|---|
| wq_a | 5120×1280 | 4.9 | 13.5 | 364 |
| wq_b | 1280×32768 | 31.5 | 25.9 | 1215 |
| wkv | 5120×512 | 2.0 | 13.2 | 149 |
| wo_a slice | 4096×1024 | 3.1 | 12.8 | 246 |
| wo_b | 8192×5120 | 31.5 | 27.1 | 1160 |

Small projections (≤5 MB) are launch/tail-bound at ~13 µs; the large ones reach ~1.2 TB/s,
below the 1.3 TB/s target. Both large shapes are the next target after the MoE work.

## Hardware counters

`ncu` on host: `ERR_NVGPUCTRPERM` (no GPU perf-counter access). No counter data is claimed.

## Regression gate

`pytest -q tests/kernels/moe/` on the unmodified fork: **259 passed** (handoff's "82" predates
MixedK's added tests; this is the current baseline).

## Fix: per-stage, bsz-aware WIDE selection (`exl3_moe_coop.cu`, launch)

Before: `moe_coop_pick_wide` chose WIDE from k-length only (A → WIDE=1 always on Blackwell).
After: A uses WIDE only when bsz > 1; B uses WIDE when bsz > 1 (otherwise k-length rule).
`EXL3_MOE_COOP_WIDE` still overrides every stage.

| case | before sum µs | after sum µs | after A / B GB/s |
|---|---|---|---|
| bsz 1 full layer | ~107–110 | 92.6 | 787 / 723 |
| bsz 6 full layer | ~488 | 453.1 | 812 / 808 |
| bsz 1 EP2 | 37.8 | 37.7 | 614 / 538 |
| bsz 6 EP2 | 230 | 221.6 | 736 / 687 |

Regression gate after change: `pytest -q tests/kernels/moe/` → 259 passed.

## Bench scratch fix and KSPLIT sweep (bsz 1)

`qmesh/bench_mixedk_coop.py` `scratch()` used `smax = 4*slots`, which capped `max_split = slots_max/slots`
at 4, so every KSPLIT > 4 was silently ignored. Fixed to `smax = 256` (production max) for gu/act/d_out/ctr/had.
Sweep (kb.py, auto WIDE, µs sum of A+B):

| KSPLIT | bsz1 full | bsz1 EP2 |
|---|---|---|
| auto | 93.5 | 37.7 |
| 1 | 94.2 | 37.8 |
| 2 | 109.3 | 43.3 |
| 4 | 141.2 | 58.6 |
| 8 | 212.9 | 92.7 |

Auto is best or tied everywhere; no better default. Kept auto. bsz 6 KSPLIT results: auto 458 / EP2 223, KSPLIT=2 is equal (459 / 220).

## Trellis decode cost (stage A, bsz 1, uniform-K templated path)

Equal table bytes (1,132,462,080 B each): K=2 over 384 experts vs K=4 over 192 experts.

| K | A µs | A bytes read (top-6 gate+up) | A GB/s | weights/s (G) |
|---|---|---|---|---|
| 2 | 51.4 | 35.4 MB | 688 | 2754 |
| 4 | 51.8 | 70.8 MB | 1367 | 2734 |

Doubling the bytes at K=4 costs 0.7% more time, so stage A is decode-bound (ALU / per-weight work), not DRAM-bound.
At ~2.74 Tweights/s the time follows weight count, so bytes are not the limiter. The 1.65 TB/s ceiling is unreachable
by byte reduction here; the decode path (trellis/codebook arithmetic per weight) is the lever.

## Status

Landed: WIDE selection fix (above). PF change rejected (no gain, PF=8 loses).
Open: MoE A at bsz 1 is still ~790 GB/s and B ~720 GB/s (about 48% and 44% of the 1.65 TB/s ceiling);
dense wq_b/wo_b ~1.2 TB/s; the 1.3 TB/s target is not yet reached.

## Decode-path investigation (K=2/K=4 stage A, bsz 1)

Method: cubins for exl3_moe_coop_inst_k{2,4}.cu built with nvcc -cubin -arch=sm_120 -std=c++20 --expt-relaxed-constexpr (+torch includes); SASS via cuobjdump with NVDISASM_PATH. The built .so stores device code compressed, so cuobjdump on it yields no SASS.

Mul1 decode per 16-bit codeword (SASS, a_kernel<2,2,0,0> and <4,2,0,0>): SGXT/SHF field extract, IMAD by 0x83DCD12D, IDP.4A (dp4a with 0x01010101, bias 0x6400), PRMT pack of two halves, one HFMA2 affine. About 2.5 ALU ops per weight in the decode itself. The dp4a form is bit-identical to decode_mul1_product_2 (codebook.cuh). No exact integer rewrite removes the IMAD+IDP.4A pair, so there is no arithmetic change that preserves output.

No-decode probe (MK_NODECODE gate, probe copy only, /tmp/cp/ext_nodec): K=4 stage A 51.8 -> 28.4 us. K=2 probe 65.6 us is not comparable (different codeword work). Decode is roughly 45% of K=4 stage A. The 28.4 us figure is an upper bound on decode-free compute, not a DRAM floor, because it also skips table reads.

Clocks: SM 2280 MHz under trellis load, max 3090 MHz, 83 W draw. Sampled once with nvidia-smi inside exl3-worker, which exposes only the bench device (GPU1 on host). No throttle observed.

Trellis rerun this slice: K=2 50.7 us (2792 G w/s), K=4 52.2 us (2713 G w/s). Within noise of 51.4/51.8.

Conclusion: no decode change shipped. Reaching 4 T w/s needs a per-weight op-count reduction that the exact-arithmetic constraint forbids; a non-exact decode needs a numerical-equivalence decision from the user.
