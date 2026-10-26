# STATUS — Recipe A (GPU0 + GX1)

Last update: 2026-10-26. Work paused by owner. Details and commands in `qmesh/notes/`.

## Caveat on hardware used so far
Several measurements below were taken on **P7 GPU1** by mistake (same card model as GPU0, so numbers
transfer, but the recipe itself is GPU0 + GX1 only). Marked [GPU1] where so.

## Measured
| Item | Result | Source |
|---|---|---|
| Handoff baseline (TCP/pickle transport, group worker) | ~100 ms/token (10 tok/s) | EXL3_HANDOFF.md |
| P7-local decode, null remote, eager | 18.6–19.6 ms/token, ~1,290 launches/token | notes/Profiler.md (GPU0) |
| P7-local decode, piecewise graphs `EXL3_WHOLE_GRAPH=1` | 18.92 → 14.59 ms/token, bit-exact (32/32 short, 64/64 long, diff 0.0), 1225 → 445 kernels + 87 graph launches | notes/GraphCapture.md (GPU0) |
| Mixed-K fused MoE | 1 launch for mixed 2/2.5/3/3.5/4 bpw; 259 MoE tests green; real-layer rel err ≤1.3e-3 | notes/MixedK.md [GPU1] |
| MoE coop kernel | ~780 GB/s (ceiling 1.65 TB/s), latency-bound; WIDE fix 108 → 93 µs/layer bsz 1 | notes/CoopPerf.md [GPU1] |
| Native RDMA seam (stub compute) | bsz1 16–23 µs/layer, graph-capturable | notes/Transport.md (GPU0 small) |
| GX1 fused mixed-K compute (layer 0, 259 cold) | rows=1 358 µs/layer, rows=6 2.0 ms/layer | notes/Transport.md |
| Round trip with real compute | rows=1 ~525 µs/layer pipelined (~21 ms/token for 40 layers) | notes/Transport.md |
| End-to-end hybrid over seam, coherent | 28–31 ms/token; DSpark K=5 best 17.6 ms/token (acceptance rate 0.74–0.82 per drafted token) | notes/E2ESeam.md [GPU1] |
| Dense defaults | HC fold on (bit-exact, −40 launches/token); x-fan opt-in (0.5–0.8 logit drift) | notes/DenseFuse.md [GPU1] |

## Main bottleneck for this recipe
GX1 cold-expert compute: ~358 µs/layer at rows=1 vs a bandwidth floor of roughly ~100 µs/layer
(≈1.2 GB/token cold reads at 273 GB/s, spread over 40 layers) — [INFERENCE] floor estimate.

## Not done
- Re-run E2E with the P7 side on **GPU0** (was run on GPU1).
- GX1 compute speed-up; seam issue/collect inside the piecewise graphs; real Engram device path.

## Artifacts outside git
- `/home/treadall/recipe-artifacts/gpu0-gx1/densefuse_trace.json`
- Seam binaries: rebuild with `qmesh/native/build.sh` (P7 x86_64, GX1 aarch64).
