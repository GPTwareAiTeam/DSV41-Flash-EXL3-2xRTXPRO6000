# RECIPE B — 2× RTX PRO 6000 Max-Q (TP2, P7 GPU0 + GPU1)

> **This is one of TWO unique, separate projects.** It is NOT the GPU0 + GX1 hybrid recipe.
> Sibling project (do not mix): `recipe/gpu0-gx1-hybrid` at `/home/treadall/recipe-gpu0-gx1`.
> This recipe was never formally started by the owner; work here came from an unapproved detour and is
> kept for reference until the owner decides to pursue it.

## Hardware (only these)
| Device | Role |
|---|---|
| **P7 GPU0** — RTX PRO 6000 Blackwell Max-Q (96 GB) | TP rank 0 (+ output device) |
| **P7 GPU1** — RTX PRO 6000 Blackwell Max-Q (96 GB) | TP rank 1 |
| Link | PCIe P2P GPU0↔GPU1 (same NUMA, `NODE`) |

**Never used by this recipe:** GX1, GX2, P7 GPU2 (A4000, foreign).

## Model
- Checkpoint: `/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.565-tp2` (177.92 GiB)
- Serving overlay (with external Engram): `/models/dsv41/exl3-serve-tp2`
- Expert mix: 2,500 @ 3.5 | 2,500 @ 3.0 | 4,860 @ 2.5 | 5,500 @ 2.0 (every layer mixed)
- All experts on GPU (EP2 or tensor-split across the two cards), no remote tier.

## Layout of this branch
- `RECIPE.md` — this file
- `STATUS.md` — measured results and open items for THIS recipe only
- `qmesh/` — TP2 benches/smoke for this recipe
- `qmesh/notes/` — workstream notes for this recipe
- Shared engine changes (exllamav3/) also present in recipe A: `SHARED_ENGINE_CHANGES.md`.
