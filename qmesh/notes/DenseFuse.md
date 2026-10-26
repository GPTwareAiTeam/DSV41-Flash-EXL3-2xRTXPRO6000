# DenseFuse: non-MoE per-layer fusion (DSV4.1 EXL3, bsz 1, P7-local)

Measured with `qmesh/dense_layer_prof.py` + `qmesh/dense_layer_table.py` on one warm decode step, GPU1 (container device 0).

## Timing (non-MoE GPU time per layer, kernel count)

| Config | Layer 3 | Layers 2/8/14 | Layer 20 | Whole model (avg/layer, total kernels) |
|---|---|---|---|---|
| Baseline (default) | 154.7 µs / 18 | ~178 µs / 20 | ~203 µs / 38 | 158.3 µs, 748 kernels (6331.3 µs) |
| (b) `EXL3_HC_FOLD=1` | 153.6 µs / 17 | 153.0–153.8 µs / 17 | — | 156.3 µs, 708 kernels (6252.6 µs) |
| (b)+(a) x-fan, HC fold on | 149.6 µs / 16 | 151.3–153.5 µs / 17 | 199.7 µs / 37 | 151.2 µs, 666 kernels (6049.2 µs) |

Layers 2/8/14 are listed as the per-layer readings from the HC-fold run and the x-fan run.

## Correctness (`qmesh/dense_ref.py`, 8 greedy steps, two prompts, full-vocab logits)

| Config | Prompt "Hi" | Prompt "The capital of France is" |
|---|---|---|
| (b) HC fold | argmax equal, max abs logit diff 0 | argmax equal, max abs logit diff 0 |
| (a) x-fan, HC fold on | argmax equal, max abs logit diff 0.513672 | argmax equal, max abs logit diff 0.816406 |
| x-fan off (`EXL3_DSV4_NO_XFAN=1`), HC fold on | argmax equal, max abs logit diff 0 | argmax equal, max abs logit diff 0 |

The x-fan diff is numerical, not a logic error: with the fan disabled the logits are bit-identical to baseline. The fused `exl3_mgemm` for q_a/wkv uses a different reduction than two separate GEMVs. [INFERENCE] The residual is fp16 rounding; argmax is stable over 8 steps on both prompts.

## Changes

1. **(b) HC fold (no code change; default still off).** `EXL3_HC_FOLD=1` defers `attn_hc.apply` into the MLP-site mix (`exllamav3/modules/hyperconnections.py`, `_hc_fold`). Bit-identical. Default left at 0 in `exllamav3/util/backend.py:56`, pending Main's approval to flip it.
2. **(a) DSV4.1 attention x-fan (`exllamav3/modules/dsv4.py`).**
   - `_build_x_fan`: removed the `q_head_norm` gate, so the fan builds for DSV4.1 (`q_head_norm=False`). Still skipped when `EXL3_DSV4_NO_XFAN` is set. Layers whose compressor has no `wgate` (layer 20, `compress_rate == 1`) keep `x_fan = None` and the per-projection path; `MultiLinear` setup still runs.
   - Fan branch of `_forward_cached_one`: the rope now chooses between the head-norm rope (`q_head_norm=True`, unchanged) and the no-head-norm split that mirrors `_project_qkv`'s `not self.q_head_norm` branch (`_ext_rope` on the trailing q slice, kv rope with `kv_norm_w`).
   - Layer 3: q_a + wkv (13.7 + 13.4 µs) replaced by one `exl3_mgemm` (23.5 µs), 1 fewer launch.
3. **(c) layer-20 index chain: analysed, not applied.** Layer 20 is `hca`; `idx_wk` is None, so the `rows = bt_row[...]` index block at `dsv4.py` ~1889 does not run there. The ~20 µs of `index_elementwise` / `vectorized_elementwise<long>` kernels come from another unlabeled site in the hca path. Not traced to a single caller, so no change made.

## Not done / remaining

- `q_b` is still a GEMV on every layer (26 µs). It needs `q_fan`, which requires an indexer (csa layers only).
- Layer 20 is unchanged at ~200 µs; the (c) index chain is unresolved.
- `EXL3_HC_FOLD` default is still 0.
- The x-fan is numerically non-identical (max abs logit diff 0.5–0.8 over 8 steps); this needs sign-off before it becomes the default.

## Tests

- `pytest -q tests/kernels/hyperconnections tests/kernels/moe`: **430 passed**, 0 failed.
- Full `tests/kernels/` baseline before changes: 5071 passed, 478 skipped, 0 failed (1153.9 s).
- Full `tests/kernels/` was not re-run after the x-fan change; the change is limited to `dsv4.py` attention, which has no dedicated kernel tests in `tests/kernels/`.

## Files

- `exllamav3/modules/dsv4.py` (x-fan gate, gateless-compressor skip, rope split in fan branch)
- `qmesh/dense_prof.py`, `qmesh/dense_layer_prof.py`, `qmesh/dense_layer_table.py`, `qmesh/dense_ref.py`, `qmesh/dense_inspect.py` (measurement tooling, new)
- `qmesh/notes/DenseFuse.md` (this note)

## Defaults changed (Main, 2026-10-26)
- `EXL3_HC_FOLD` default 1 on CUDA (util/backend.py); bit-exact (dense_ref.py vs /tmp/densefuse_ref.pt: max abs diff 0 both prompts, 8 steps). tests/kernels/hyperconnections + dsa: 322 passed.
- DSV4.1 x-fan opt-in via `EXL3_DSV41_XFAN=1` (not bit-exact: 0.5–0.8 logit diff). doc/env_vars.md updated.

## Per-projection kernel attribution (record_function per Linear; bsz 1 decode, GPU1)
| Projection | L3 µs | L20 µs | MB | L3 GB/s |
|---|---|---|---|---|
| q_a (gemv) | 14.05 | 13.44 | 4.93 | 351 |
| q_b (gemv) | 26.08 | 26.59 | 31.53 | 1209 |
| wkv (gemv) | 13.89 | 13.86 | 1.98 | 142 |
| comp.wkv (gemv) | — | 13.31 | 1.98 | — |
| idx_wk (gemvx) | — | 2.78 | 0.13 | — |
| wo_a (mgemm, unlabelled trace) | ~29.95 | ~29.95 | 25.25 | 843 |
| wo_b (gemv) | 27.71 | 27.90 | 31.48 | 1136 |
Attention weights: L3 95.2 MB, L20 101.2 MB. Small GEMVs are latency-bound (~13–14 µs floor regardless of size); large ones run at 1.13–1.21 TB/s.
Layer-20 ~17 µs index_elementwise/long kernels: attributed to `_forward_cached_one` (frame granularity only); line not identified.
