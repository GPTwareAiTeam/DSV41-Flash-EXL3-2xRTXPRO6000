# STATUS

## 2026-10-27 (later) — spec on the captured path, and where the remaining cost is
Canonical main-branch numbers (`COMM=p2p`, graphs, `EXL3_DSV4_B1_BATCH=1`, WARM):
- plain: **10.97 / 11.02 / 10.98 ms/tok** (91 tok/s)
- DSpark K=5: **7.86 / 6.83 / 9.01 / 4.65 ms/tok** on prompts 0-3; AL 3.30/3.30/2.58/**5.29**; acceptance 0.66/0.78/0.91;
  effective 104-136 tok/s on short prompts, 123 at AL 5.29.
- Round cost is flat at ~23 ms = target forward (~8) + 5 drafter forwards (~7) + verify (~8).
- Spec policy **closed**: with matched flags K=5 beats plain on all six bench prompts; K<5 is dominated; `auto` lost
  on 3 of 6. Default stays `DSPARK_POLICY=always`.
- Prefill chunk sweep (32K prompt): 2048 -> 3856, 8192 -> **4790** (90.2 GiB peak, safe), 16384 -> 4941 (93.7 GiB, too
  tight), 32768 OOM (needless `torch.cat` transient at block_sparse_mlp.py:1072).
- Decode is now **kernel-efficiency bound**, not host bound: one 11 ms step has ~9 ms CUDA busy and the bsz-1 MoE is at
  ~45% of DRAM while diffbot reaches ~92% with a bigger 3 bpw model on the same hardware.

## Consolidated best-known numbers (main branch, both ranks agree)
Reproduce: `cd /home/treadall/recipe-artifacts/2x-rtx-tp2 && ./run_tag.sh <unique_tag> -e COMM=p2p -e EXL3_WHOLE_GRAPH=1 \
  -e EXL3_DSV4_B1_BATCH=1 -e MAX_NEW=128 -e PROMPT_IDS=0,1,2,3 -e IGNORE_EOS=1 -e WARM=1` (add `-e DSPARK=1 -e NUM_DRAFT=5` for spec).

| config | ms/tok | tok/s | notes |
|---|---|---|---|
| plain decode bsz1 (p0/p1/p2) | 10.97 / 11.02 / 10.98 | 91 | captured attention core |
| plain decode, 1134-token prompt | 14.15 | 71 | long-context KV cost |
| DSpark K=5 (p0/p1/p2/p3) | 7.86 / 6.83 / 9.01 / 4.65 | 127 / 146 / 111 / 215 | AL 3.30/3.30/2.58/5.29, acc 0.66/0.78/0.91 |
| prefill 2048-token prompt, chunk 2048 | 3856 tok/s | | 32K-prompt sweep, per-chunk table in PrefillPerf.md |
| prefill 32K prompt, chunk 8192 | 4790 tok/s | | 90.2 GiB peak, safe default |
| prefill 32K prompt, chunk 16384 | 4941 tok/s | | 93.7 GiB peak, too tight to ship |
| multi-stream plain B2 / B4 (branch tp2/batch) | 14.62 / 19.51 per stream | 126 / 189 aggregate | 86.5 GiB peak; numerics bug open |
| multi-stream DS K=5 B4 | 38.26-38.97 per stream | 96.8 aggregate | spec and batching do not compose |

Round anatomy (DSpark K=5, measured with CUDA events): draft **1.9 ms**, target+verify **21.5 ms**, host ~3 ms; exactly ONE
drafter forward per round (DFlash block path). MoE cost is linear in distinct expert slots (1 row = 6 slots ~ 11 ms,
6 rows ~ 38 slots ~ 21.5 ms), so speculative decode is already at the DRAM floor per token and the remaining lever is
kernel efficiency (bsz-1 MoE ~45% of DRAM vs diffbot ~92% on the same hardware).

## 2026-10-27 — attention-core capture lands: plain decode 13.3 -> 11.0 ms/tok
Validated on main (`b85deac0` merged from `tp2/attncap`): **10.97 / 11.02 / 10.98 ms/tok** (prompts 0-2, both ranks
identical) vs **13.35 / 13.35 / 13.17** before. 1134-token prompt 14.77 -> 14.15.
Root cause of the earlier "capture is slower" readings: `b1_prep` required `q_head_norm=True` while `_batch_ok_b1`
required `False`, so the captured body never ran and every measurement was the eager fallback. Gates removed,
capture restricted to S==1.
**Tokens byte-identical** (both ranks, 4 prompts x 128 tokens, including the 1134-token prompt).
Spec decode on the captured path: 9.46 / 6.88 / 9.09 / **4.61** ms/tok (AL 2.51 / 3.24 / 2.48 / **5.29**) — the long
prompt reaches 119.6 tok/s. Round cost stays ~22-24 ms, so the remaining spec cost is NOT host overhead.

 — Recipe B (2× RTX PRO 6000 Max-Q, TP2)

Last update: 2026-10-26. Not formally started by the owner; results below came from an unapproved detour
and are kept for reference. Details in `qmesh/notes/TP2Prep.md`, `qmesh/notes/TP2Bringup.md`.

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

## Memory
- GPU0 free after TP2 load: 7.43 GiB. DSpark drafter needs ~7.74 GiB (7.12 + 0.62 private LM head) → short ~0.31 GiB on GPU0; GPU1 has ~8.3 GiB free. Not resolved.

## Not done
- DSpark under TP2 (placement), one-shot P2P all-reduce kernel (`qmesh/p2p/p2p_reduce_bench.py` written, never completed a run).
- The last TP2Bringup agent was killed mid-run while moving the drafter to GPU1. Checked: no `exllamav3/architecture/` edits exist; its only changes are in `qmesh/tp2_bench.py` (DSpark options), copied here as-is.

## Artifacts outside git
- `/home/treadall/recipe-artifacts/2x-rtx-tp2/tp2logs/` (148 MB of bench logs/traces)
