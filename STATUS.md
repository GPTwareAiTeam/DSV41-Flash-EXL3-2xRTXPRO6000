# STATUS — Recipe A (GPU0 + GX1)

Last update: 2026-11-01. Work paused by owner. Details and commands in `qmesh/notes/`.

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

## Known incompatibility (2026-11-01)
`EXL3_WHOLE_GRAPH=1` and the MTP drafter (`DSPARK=1`) cannot be combined. The piecewise decode
graphs replay the blocks, so the module-level export taps (`exllamav3/modules/transformer.py:253`,
`rmsnorm.py:169`) never run and `params["export_states"]` is never filled. The drafter then hits
`deepseek_v4_mtp.py:490 T = target_hidden[0].shape[1]` with `target_hidden = None` and raises
`TypeError: 'NoneType' object is not subscriptable` on the first drafted step (observed as the
warm-up failure of `final_prod3`). Until the graph path exports the collapsed hidden state of the
last trunk block (or drafting is skipped while the export is missing), run either
`WHOLE_GRAPH=0 DSPARK=1` or `WHOLE_GRAPH=1 DSPARK=0`. Memory is identical either way; eager vs
graph decode measured neutral on this pair (25.38 vs 25.97 ms/token) while the drafter is worth
~1.20x. `scripts/mem_budget_run.sh` takes `WHOLE_GRAPH` (default 1); the production arms run
`WHOLE_GRAPH=0 DSPARK=1`.

## Client transport switch (cost one night of failed arms)
The qes seam is only used when the client sets `REMOTE_TRANSPORT=seam` (`block_sparse_mlp_cpu.py:891`);
the default `tcp` sends the legacy pickle protocol to the control port, which is now rejected as
`SEAM_REJECT reason=bad-control-magic` without touching the server. `mem_budget_run.sh` passes
`REMOTE_TRANSPORT=seam` and `SEAM_BUCKETS` explicitly.

## Multi-session seam (2026-11-01)
One GX server process now serves any number of client lifetimes; a 150-220 s cold-pack reload per run is
no longer part of the loop. The control port (e.g. 47311) negotiates a session (`qes.Control`), every data
port is bound for the server's lifetime, and the next client either queues (`SEAM_QUEUED`, `--wait`) or
starts when the previous session's control connection closes. `SEAM_BUCKETS` on the client is a request,
not a contract: the server grants the intersection and reports it (`buckets=… ports=…`), so a decode-only
client that opens just one of three granted buckets ends with `SEAM_SESSION_PARTIAL … up=1 of 3` instead of
a failed session. An unexpected connection on a data port is validated by its hello and closed
(`SEAM_REJECT … reason=stale-data-conn`) instead of aborting the process — the old `qes: peer layout
mismatch` path killed the server and cost a reload. Any number of such connections is tolerated: they are
closed and counted, never a reason to end the session (the session ends with its control connection).
One server still runs one live session at a time (a second client queues or is refused as `busy`), so the two
pairs doing different work at once means one server per pair — prod on GX1, dev on GX2 — each with its own
`--buckets`, build and client config. See `qmesh/notes/E2ESeam.md`, `qmesh/qes.py`,
`qmesh/worker_seam_server.py`, `qmesh/seam_status.py`.

Verified on the dev pair's GX2 in stub mode (`seam-gx2-robust`, port 47351, clients in the same container, no
GPU and no queue slot): 59 checks, 0 failures, one server pid through all 10 phases - two sessions back to
back, a subset request, handles opened in reverse bucket order, a refused busy client, a queued client served
19 s later, stray control connections, 15 strangers on the data ports (7 rejected as stale, the rest drained
by the next session), a legacy pickle client, a SIGKILLed client served again after 2 s, and zero
`SEAM_SESSION_REJECT` / `peer layout mismatch` in the whole server log.
Log: `artifacts/seamrobust/battery-1101-010510/battery.log` (build=cb76845f on server and clients).
