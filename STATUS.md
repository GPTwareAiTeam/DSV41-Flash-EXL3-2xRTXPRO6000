# STATUS — Recipe B (2× RTX PRO 6000 Max-Q, TP2)

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
