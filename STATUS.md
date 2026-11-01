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

## Needle prompt format: the long-context arm was scoring a different prompt (2026-11-01)
The 131k arm of `qmesh/mem_budget.py` read the needle fixture as raw text with `add_bos=True`. That is
not the prompt v1.2 scored 5/5 on. The reference harness (`artifacts/2x-rtx-tp2/fp4kv/needle.sh`,
`PROMPT_IDS=8`) builds `PROMPTS[8] + "\n\n" + transcript` (`tp2/run_tp2.py:189`) and wraps it in the
checkpoint's own chat template, `encoding.encode_messages([{"role": "user", "content": prompt}],
thinking_mode="chat")`, then encodes with `add_bos=False, encode_special_tokens=True`
(`qmesh/tp2_bench.py:66-71`). Without the template the base model continues the prose instead of
answering the request block, which is what `final_prod7` produced.

`LONG_CHAT=1` now takes the reference path (`ENC_DIR` defaults to
`/models/dsv41/DeepSeek-V4.1-Flash-1017-orig/encoding`, pure python, no torch), prints
`LONG_CHAT lead=... chars=...`, `LONG_PROMPT tokens=...` and `LONG_PROMPT_TAIL`, and `IGNORE_EOS=1`
mirrors `tp2_bench.run_prompt` (`min_new_tokens=max_new`, no stop conditions). Generated ids are read
from `token_ids` (the key the generator reports, and the one `tp2_bench` reads) with `ids` as fallback;
the earlier arm printed `LONG_TEXT_IDS []` because only `ids` was read. Token counts with the template:
short fixture 3122, 131k fixture 131 233 (raw 131 222). `mem_budget_run.sh` now defaults `DEPTH_PROBES`
with `${VAR-default}`, so `DEPTH_PROBES=""` really disables the 512k/1M probes.

## Prefill chunk limiter: first-chunk term (2026-11-01)
`qmesh/prefill_limit.py` predicted 3056 MiB for an 8192-row chunk and measured it on 15 of 16 chunks;
the first chunk rose 3573 MiB. The extra ~517 MiB is one-time (uncompressed-pool staging window and
indexer score slab: `statics` 0.000 GiB after load, 0.559 GiB at 131k) and stays allocated, so it is
charged to the chunk that allocates it, in proportion to its rows: the 3121-row single-chunk needle arm
rose 1340 MiB against a row term of 1164 (+176) and the two fits give 63 and 56 KiB/row, so
`PREFILL_FIRST_ROW_KIB=64` is added per row of the first chunk only (`start == 0`). 8192 rows now predict
3568 MiB (measured 3573), 3121 rows 1359 (measured 1340); later chunks keep the 382 KiB/row term.

## Seam session setup race: server sends its hello before its QP is live (2026-11-01)
`session_handle_up` used to be *recv peer hello → send own hello → close(fd) → check_hello →
qp_connect (RTR, then RTS)*. The client proceeds as soon as it has the server's hello, so its first payload
WRITE could reach a responder QP that was still in INIT, or between RTR and RTS. The HCA answers that write
with `remote invalid request` — `qes: wc error remote invalid request error opcode=1 port=47318 rows=64
vend_err=0xdfffeca0 wr_id=0x0 off=0 kind=payload` — and the same packet moves the responder QP to ERROR, so
the server's own `ibv_modify_qp(RTS)` fails and the session is rejected. A rejected session is survivable
now, but the *client* side of it was not: the 30 s wait kernel called `__trap()`, which raised a sticky
`CUDA error: unspecified launch failure` and killed the client process (rc 134).

Proof it was ordering and not layout (`artifacts/seamrobust/trace-1101-082030/`): the server log has
`qes: seam session 1: qp rts (port=47318 rows=64)` immediately followed by
`SEAM_SESSION_REJECT … sid=1 reason=setup-failed`, a pair that only `qp_connect`'s failure path can print,
on exactly the port whose client write NAKed; the client's `SEAM_TRACE_HANDLE` (port 47318, stride 1968512,
slab 78740480, `remote_base`/`remote_rkey`) is byte-identical to the server's `SEAM_TRACE_EP`, and
`check_hello` passed. Phase 0 (deployed build, trace on) reproduced it every time; phase 1 with the reorder
was 7/7 `SEAM_CONNECT`, no `qes: wc error`, `CLIENT_RC=0`, `SEAM_SESSION_END … calls=3876`.

Fix: *recv peer hello → check_hello → qp_connect (RTR, RTS) → print `SEAM_TRACE_RTS role=srv` → send own
hello → close(fd)*. The peer cannot start WRITEs before the responder QP is live, and it costs no extra round
trip. Every failure path still closes the fd and keeps the stale/non-stale distinction the session logic
needs. A missing readiness gate on the server helper (client started ~15 s after `SEAM_SERVER_LOADED`, while
the pinned slab was still being allocated) made that window enormous on one prod arm; that helper bug is
fixed separately, and this reorder removes the residual few-hundred-µs window on a fully-ready server.

Fail-stop instead of trap: `wait_kernel` no longer traps. After `WAIT_TIMEOUT_NS` it raises a fail word in
bytes 8..11 of the slot's freq line (offset 0 is the request seq, the rest of the 64 B line is unused, so
neither the slot geometry nor the wire layout changes) and returns; `client_proxy` sees the word, prints
`qes: no response for layer slot N … the seam transport is dead`, and marks the handle dead the same way a
CQ error does; `qes_issue`/`qes_collect` then fail with −5 and `last_error()` reaches Python as a
`RuntimeError`. A lost handle now costs one session, not the process. Failed `ibv_modify_qp` calls print the
state the HCA really reports (`qes: qp rts failed errno=… state=… qpn=… peer_qpn=… port=… rows=…`).
