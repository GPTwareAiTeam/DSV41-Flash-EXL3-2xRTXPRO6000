# Engine changes shared by both recipes

These `exllamav3/` changes exist identically in BOTH recipe branches (`recipe/gpu0-gx1-hybrid` and
`recipe/2x-rtxpro6000-tp2`). If you change one, decide explicitly whether the other recipe gets it too.

| Change | Files | Why both |
|---|---|---|
| Mixed-K fused MoE (per-expert bitrate tables, decode coop + prefill) | `exllamav3_ext/quant/exl3_moe*.cu/.cuh`, `exl3_moe_coop_mixed_*`, `libtorch/blocksparse_mlp*`, `bindings.cpp`, `modules/multilinear.py`, `modules/block_sparse_mlp.py` | A: GX1 cold tier mixes 3.0/2.5/2.0. B: every layer mixes 2.0–3.5 |
| MoE coop tile (WIDE) selection per stage | `exllamav3_ext/quant/exl3_moe_coop.cu` | kernel perf, both |
| Piecewise decode CUDA graphs (`EXL3_WHOLE_GRAPH=1`) | `util/segment.py`, `modules/transformer.py`, `model/model_ls.py` | both decode paths |
| `EXL3_HC_FOLD` default on (bit-exact) | `util/backend.py`, `doc/env_vars.md` | both |
| DSV4.1 x-fan opt-in (`EXL3_DSV41_XFAN=1`), no-head-norm rope split | `modules/dsv4.py`, `doc/env_vars.md` | both |
| Owner's original WIP (requant/copy mode) | `conversion/convert_model.py`, `modules/block_sparse_mlp.py` | owner's pre-existing work |
| Tests | `tests/kernels/moe/test_moe_*mixed*.py`, `test_moe_real_layer.py`, `test_moe_fused_det.py`, `test_moe_half_rates.py` | cover mixed-K |

Recipe-specific (NOT shared):
- Recipe A only: `modules/block_sparse_mlp_cpu.py` (REMOTE_TRANSPORT=seam switch), `qmesh/native/` (RDMA seam), GX1 workers, placement.
- Recipe B only: `model/model_tp_backend.py` (native TP all-reduce → on-device ring), `qmesh/tp2_*`, `qmesh/p2p/`.
