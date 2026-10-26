# RECIPE A — GPU0 + GX1 (hybrid, remote expert tier)

> **This is one of TWO unique, separate projects.** It is NOT the dual-RTX recipe.
> Sibling project (do not mix): `recipe/2x-rtxpro6000-tp2` at `/home/treadall/recipe-2xrtx-tp2`.

## Hardware (only these)
| Device | Role |
|---|---|
| **P7 GPU0** — RTX PRO 6000 Blackwell Max-Q (96 GB) | dense layers, embedding, LM head, MTP/DSpark drafter, Engram projections, KV, **4,100 hot experts @ 4.0 bpw** |
| **GX1** — ASUS Ascent GX10 / GB10 (DGX Spark class), 121.63 GiB unified, `10.0.100.2` | **11,260 cold experts** (3.0 / 2.5 / 2.0 bpw) |
| Link | 100 GbE QSFP direct / RoCE: P7 `rocep142s0f0` 10.0.100.1 ↔ GX1 `rocep1s0f0` 10.0.100.2 |

**Never used by this recipe:** P7 **GPU1** (belongs to recipe B), P7 GPU2 (A4000, foreign), GX2.
Every P7 process for this recipe must run with `CUDA_VISIBLE_DEVICES=0` (or a container created with `--gpus device=0`).

## Model
- Checkpoint: `/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.8535` (196.17 GiB)
- Serving overlay (with external Engram): `/models/dsv41/exl3-serve-final`
- Placement: `qmesh/placement.final.json` (hot 67.74 GiB on GPU0, cold 113.48 GiB on GX1)
- GX1 cold pack: `/home/treadall/exl3-cold-pack` (mounted `/cold`)
- Expert mix: 4,100 @ 4.0 (hot) | 3,560 @ 3.0 | 2,700 @ 2.5 | 5,000 @ 2.0
- DSpark: K=5, drafter on **GPU0**

## Goal
150 tok/s (6.67 ms/token) bsz 1, GPU0 + GX1 only.

## Layout of this branch
- `RECIPE.md` — this file (hardware/model contract)
- `STATUS.md` — measured results and open items for THIS recipe only
- `qmesh/` — recipe runtime (seam, workers, smoke, placement)
- `qmesh/notes/` — workstream notes for this recipe
- Shared engine changes (exllamav3/) that also exist in recipe B are listed in `SHARED_ENGINE_CHANGES.md`.

Source handoff: `/home/treadall/.local/omp-dash/EXL3_HANDOFF.md` (describes this recipe).
