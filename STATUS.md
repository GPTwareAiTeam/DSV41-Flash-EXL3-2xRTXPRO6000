# STATUS

This file preserves historical measurements below, not a current runbook. For current priorities and long-context results see `qmesh/notes/MASTER_PLAN.md`. Batch correctness remains unresolved for diverse prompts; identical-prompt checks alone do not establish general correctness.

## 2026-10-27 (historical) — spec on the captured path
Measurements used `COMM=p2p`, graphs, `EXL3_DSV4_B1_BATCH=1`, WARM:
- plain: **10.97 / 11.02 / 10.98 ms/tok** (91 tok/s)
- DSpark K=5: **7.86 / 6.83 / 9.01 / 4.65 ms/tok** on prompts 0-3, or **127 / 146 / 111 / 215 tok/s**; AL 3.30/3.30/2.58/5.29.
- Round anatomy measured one DFlash block forward, not five sequential drafter forwards: draft ~1.9 ms, target+verify ~21.5 ms, host ~3 ms. Timings depend on configuration.
- In one matched six-prompt comparison K=5 beat plain and `auto` lost on 3 of 6; this does not establish that K<5 is dominated in all workloads. Long-context results are content-dependent (see master plan).
- Historical 32K prefill sweep reported 2048 -> 3856, 8192 -> **4790**, 16384 -> 4941 tok/s, and 32768 OOM. Do not attribute these prefill figures to the P2P recipe below: the documented 4777.7 tok/s 32K/8192 baseline used NCCL (`tp2logs/final32768_8192_rank0.log`); the transport for this earlier 4790 sweep is not established here.
- The short-decode profile suggested kernel-efficiency headroom; it does not establish the bottleneck of long-context decode.

## Historical loose end — resolved
The `gemv_tile` local-load fix was subsequently measured and shipped: **67.31 -> 65.76 us/call** at bsz1, bitwise identical; see the shipped section of the master plan. Earlier bootstrap hangs and the incomplete A/B are not current blockers.

## Historical measurements (both ranks agreed in these runs)
Historical decode recipe (not a reproduction of the prefill figures): `cd /home/treadall/recipe-artifacts/2x-rtx-tp2 && ./run_tag.sh <unique_tag> -e COMM=p2p -e EXL3_WHOLE_GRAPH=1 \
  -e EXL3_DSV4_B1_BATCH=1 -e MAX_NEW=128 -e PROMPT_IDS=0,1,2,3 -e IGNORE_EOS=1 -e WARM=1` (add `-e DSPARK=1 -e NUM_DRAFT=5` for spec).

| config | ms/tok | tok/s | notes |
|---|---|---|---|
| plain decode bsz1 (p0/p1/p2) | 10.97 / 11.02 / 10.98 | 91 | captured attention core |
| plain decode, 1134-token prompt | 14.15 | 71 | long-context KV cost |
| DSpark K=5 (p0/p1/p2/p3) | 7.86 / 6.83 / 9.01 / 4.65 | 127 / 146 / 111 / 215 | AL 3.30/3.30/2.58/5.29, acc 0.66/0.78/0.91 |
| prefill 2048-token prompt, chunk 2048 | 3856 tok/s | | 32K-prompt sweep, per-chunk table in PrefillPerf.md |
| prefill 32K prompt, chunk 8192 | 4790 tok/s | | 90.2 GiB peak, safe default |
| prefill 32K prompt, chunk 16384 | 4941 tok/s | | 93.7 GiB peak, too tight to ship |
| multi-stream plain B2 / B4 (branch tp2/batch) | 14.62 / 19.51 per stream | 126 / 189 aggregate | historical branch result; numerics bug was open |
| multi-stream DS K=5 B4 | 38.26-38.97 per stream | 96.8 aggregate | historical result; spec and batching performed poorly in this run |

Round anatomy (DSpark K=5, measured with CUDA events): draft **1.9 ms**, target+verify **21.5 ms**, host ~3 ms; exactly ONE drafter forward per round (DFlash block path). These measurements do not establish a DRAM floor or generalize to every workload.

## 2026-10-27 — attention-core capture lands: plain decode 13.3 -> 11.0 ms/tok
Validated on main (`b85deac0` merged from `tp2/attncap`): **10.97 / 11.02 / 10.98 ms/tok** (prompts 0-2, both ranks
identical) vs **13.35 / 13.35 / 13.17** before. 1134-token prompt 14.77 -> 14.15.
Root cause of the earlier "capture is slower" readings: `b1_prep` required `q_head_norm=True` while `_batch_ok_b1`
required `False`, so the captured body never ran and every measurement was the eager fallback. Gates removed,
capture restricted to S==1.
**Tokens byte-identical** (both ranks, 4 prompts x 128 tokens, including the 1134-token prompt).
Spec decode on the captured path: 9.46 / 6.88 / 9.09 / **4.61** ms/tok (AL 2.51 / 3.24 / 2.48 / **5.29**) — the long
prompt reaches 119.6 tok/s. Round cost stays ~22-24 ms, so the remaining spec cost is NOT host overhead.

## Historical bring-up — Recipe B (2× RTX PRO 6000 Max-Q, TP2)

The following 2026-10-26 bring-up results are retained for reference; later work supersedes the open items below. Details in `qmesh/notes/TP2Prep.md`, `qmesh/notes/TP2Bringup.md`.

## Measured (GPU0 + GPU1, native backend unless noted)
| Config | Load | Peak alloc GPU0 / GPU1 | Decode ms/token |
|---|---|---|---|
| EP2, eager | 106 s | 86.95 / 86.05 GiB | 20.74 |
| tensor-split MoE, eager | 106 s | 87.03 / 86.37 GiB | 20.62 |
| NCCL, EP2, eager | 96 s | 86.70 / 85.80 GiB | 27.25 |
| NCCL + piecewise graphs | 97 s | 86.70 / 85.80 GiB | 22.47 (hangs at exit) |
| native, on-device ring all-reduce, eager | — | — | 20.20 |
| **native, on-device ring + piecewise graphs** | — | — | **17.66** (tokens equal to eager ring, 3 prompts × 64) |

- Output coherent ("The capital of France is Paris.") after the chat-template fix.
- Graph-mode GPU time shares: MoE 37%, GEMV 31%, all-reduce 17% (~2.3 ms/token), HC 6%, attention core 5.5%.
- All-reduce (1,5120) fp32: CPU helper 14.8 µs; on-device ring ~19–21 µs.
- fp32 ring vs CPU bf16-wire reduce diverge greedily after 11–17 tokens (ring chosen as reference).

## Historical memory and open items (2026-10-26)
- At that point GPU0 free after TP2 load was 7.43 GiB; the estimated DSpark drafter requirement was ~7.74 GiB. Later DSpark runs succeeded.
- DSpark under TP2 and the one-shot P2P all-reduce were not yet completed in the bring-up; both were measured subsequently.
- The last TP2Bringup agent was killed mid-run while moving the drafter to GPU1; this is historical context, not the current state.

## Artifacts outside git
- `/home/treadall/recipe-artifacts/2x-rtx-tp2/tp2logs/` (148 MB of bench logs/traces)
