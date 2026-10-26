# MixedK notes

## Baseline (unmodified fork, container exl3-worker, host GPU1 only)
- Container pinned to host GPU1 (DeviceIDs ["1"]); CUDA_VISIBLE_DEVICES unset; only visible device is RTX PRO 6000 Blackwell.
- TORCH_EXTENSIONS_DIR=/opt/vllm-moet/torch_extensions; compiled module cache: /opt/vllm-moet/torch_extensions/exllamav3_ext (built Oct 25).
- `cd /src/exllamav3 && python3 -c "import exllamav3.ext"` -> 1.4 s (warm cache). Ops present: exl3_moe, exl3_moe_coop, exl3_moe_gather.
- Checkpoint: mixed-2.565-tp2, codebook mul1, 40 layers, hidden 5120, moe_intermediate 2304.

## Status (MixedK, this session)
- No source edits made yet. Fork working tree untouched by this session.
- Decode coop kernels A/B are `__global__` templates `<bits,cb,WIDE,HALF>` owning `__shared__` arrays and `smem_dyn`; K is a launch-time template, so one launch = one K.
- Prefill `exl3_moe` main kernel already loops per expert and has a `t_bits==0` runtime-K switch (half-bit units), but K is passed as a launch scalar per projection.
- Options for per-expert K in decode:
  (a) Factor A/B bodies into `__device__` templates; add mixed `__global__` wrappers that switch on `k2[local]` over 11 variants. Smem = max over variants. Uniform path unchanged. Cost: ~280 lines of hot-kernel refactor, larger register/code footprint for the mixed kernel.
  (b) Group runs by K on the host and launch one kernel per K group. Rejected by the assignment (per-K launches illegal-address when concurrent; multi-launch breaks graph-once design).

## Incident (MixedK)
- A hashline range edit on exl3_moe_coop_kernel.cuh (B kernel, lines ~812-930) corrupted the file: a placeholder line and a stale duplicate of the B kernel head were inserted. Repaired the joint; the file is NOT yet compilable:
  - B body is now `__device__ exl3_moe_coop_b_body<bits,cb,WIDE,HALF>(p, smem_dyn, sh_flag, sh_n_active, sh_gate, sh_res, sh_rows)`; the epilogue still uses bare `sh_gate`/`sh_flag`/`sh_n_active` names that must become pointer derefs.
  - The `__global__ exl3_moe_coop_b_kernel` wrapper was removed and must be re-added (declares `extern __shared__`, `__shared__` arrays, calls body).
- The fork WIP backup tgz does not contain this file; the pre-edit state was not snapshotted separately.
- Remaining decode work (A/B wrappers, mixed dispatch, host launch, prefill table, plumbing, harnesses) not done.

## Result (MixedK, measured on GPU1 only)
Status of the earlier "Status" section above: superseded. The mixed kernels, prefill tables, Python plumbing and the BC binding are now in the tree (uncommitted). `exl3_moe_coop_kernel.cuh` is equal to HEAD (checked), so the Incident is resolved.

### Commands (all GPU1, container exl3-worker, CUDA_VISIBLE_DEVICES=0)
- Build ext: `cd /src/exllamav3 && CUDA_VISIBLE_DEVICES=0 python3 -c "import exllamav3.ext; print('ext ok')"` -> `ext ok` (~55 s).
- Regression: `python3 -m pytest -q tests/kernels/moe/test_moe_fused_mixed.py tests/kernels/moe/test_moe_half_rates.py tests/kernels/moe/test_moe_fused_det.py tests/kernels/moe/test_moe_coop.py tests/kernels/moe/test_moe_coop_mixed.py tests/kernels/moe/test_moe_real_layer.py` -> 82 passed.
- Real layer (item 3): `python3 -m pytest -q -s tests/kernels/moe/test_moe_real_layer.py` ->
  - bsz 1: err 1.574e-05, scale 1.236e-02, rel 1.273e-03
  - bsz 6: err 2.197e-05, scale 2.524e-02, rel 8.702e-04
  - bsz 48: err 1.529e-05, scale 2.899e-02, rel 5.272e-04
  - Reference: per-expert dense `exl3_linear_ref` on the layer's own loaded tensors, pinned routing. Tolerance 3e-3 x scale.
- Layer K table (decoded from trellis shapes, words/16 = K; all of w1, w3, w2 agree): 148 x K=2, 109 x 2.5, 60 x 3.0, 67 x 3.5; mean 2.5599 bits; no K=4.
- Per-layer timing, real layer 3 via BlockSparseMLP (`qmesh/bench_mixedk_layer.py`, BENCH_ITERS=100):
  - bsz 1 eager 117.18 us (580 GB/s); graph 114.38 us (594 GB/s)
  - bsz 6 eager 485.75 us (793 GB/s); graph 515.44 us (747 GB/s)
- Kernel timing, real geometry (H 5120, I 2304, top-6, 384 experts, mul1, silu, gated), direct `ext.exl3_moe_coop` (`qmesh/bench_mixedk_coop.py`):
  - Run A (BENCH_ITERS=100), Run B (BENCH_ITERS=300):

| bsz | variant | mode | A us | B us | B GB/s |
|---|---|---|---|---|---|
| 1 | uniform3 | eager | 119.37 | 122.16 | 652 |
| 1 | mixed | eager | 121.00 | 121.22 | 584 |
| 1 | uniform3 | graph | 116.37 | 118.59 | 671 |
| 1 | mixed | graph | 117.78 | 117.95 | 600 |
| 6 | uniform3 | eager | 488.92 | 561.36 | 780 |
| 6 | mixed | eager | 502.38 | 559.20 | 657 |
| 6 | uniform3 | graph | 574.51 | 588.87 | 744 |
| 6 | mixed | graph | 539.94 | 563.54 | 652 |

### Proven vs inferred
- Proven: mixed correctness (real layer, 3 bszs, within tolerance); mixed and uniform3 kernel times agree within about 1% at bsz 1 (both runs); the mixed path issues one launch per projection and takes CUDA-graph capture (graph timings ran).
- Measured but noisy: bsz 6 run-to-run spread is about 15% (uniform3 eager 489 vs 561 us). No bsz 6 ordering is claimed.
- Not measured: (b') sequential per-K launches (not implemented; the coop ABI has no per-K-class expert mask). The fallback comparison is therefore not in this section.
- Not measured: full-token decode rate. These are per-layer MoE numbers; the 150 tok/s target (6.67 ms/token) is not shown by this data.
- Note on bytes: mixed moves fewer bytes (70.8 MB vs 79.6 MB at bsz 1) but takes the same time, so bsz 1 is not bandwidth-bound at these sizes. GB/s is effective bytes over measured time, not a peak.
- Scope caveat: uniform3 is a synthetic layer (random trellis, K=3.0 everywhere). Layer 3 averages 2.56 bits. The comparison is therefore not bit-matched; mixed moves less data.

### Files of this session (uncommitted)
- `exllamav3/exllamav3_ext/libtorch/blocksparse_mlp_bc.h`: added `c10::optional<at::Tensor>` x3 to `py::init` after `bool, bool`, and `py::arg("gate_k2"/"up_k2"/"down_k2") = py::none()` after `sh_coop`. This was the BC constructor arity mismatch.
- `exllamav3/modules/block_sparse_mlp.py` line 136: `_buf_get` fallback now `g_tensor_cache.get(device, shape, dtype, name)` (was self-recursive). GraphCapture made the identical change in their copy.
- `tests/kernels/moe/test_moe_real_layer.py` (item 3), `qmesh/bench_mixedk_layer.py`, `qmesh/bench_mixedk_coop.py` (item 4), `qmesh/notes/mixedk_gen.py` (copy of `/tmp/mk/gen.py`).
