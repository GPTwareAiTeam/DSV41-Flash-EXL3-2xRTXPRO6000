# GraphCapture: whole decode-step CUDA graph (P7 GPU0, bsz 1, null remote)

## Status (in progress)
- Container `exl3-graph` created: image vllm-moet:dsv41-213d5fa, --gpus device=0, fork mounted at /src/exllamav3, ext cache /mnt/data12/work/exl3/graph_ext.
- Fork extension does NOT currently link: MixedK is editing exl3_moe_coop*.cu/.cuh + bindings.cpp (rebuild in progress). Not touched by GraphCapture.
- Harness: qmesh/graph/load.py (= qmesh/load.py, imports qmesh.null_remote), qmesh/graph/time_eager.py.

## Baselines (measured)
- Eager, stale container exl3-dsv41 (/src/exllamav3 = /home/treadall/src/exllamav3-dsv41), GPU0, null remote, EXL3_MOE_SPLIT_FUSED=0, 62 warm decode steps:
  mean 19.560 ms, median 19.635, min 18.375, max 21.409 (load 5.2 s, VRAM 76.21 GiB).
  Harness: qmesh/prof/time_gen.py (NTOK=64).

## Capture breakers
(to be filled after capture probe on the fork)

## Isolated copy baseline (/home/treadall/qmesh-exl3-graph, container exl3-graph, ext cache graph_ext2)
- Eager, time_eager.py, EXL3_MOE_SPLIT_FUSED=0, 62 warm steps: mean 19.765 ms, min 18.587, max 22.629 (load 5.1 s, VRAM 76.21 GiB). Matches stale 19.56 ms within noise.
- Fixes needed to run this copy: `marisa-trie==1.4.1` installed in exl3-graph (loader dependency); `BlockSparseMLP._buf_get` fallback recursed into itself; now delegates to `g_tensor_cache.get` (same cache HEAD uses).
## Whole-step capture probe (GraphCapture2)

Eager baseline (graph copy, P7 GPU0, NTOK=64, EXL3_MOE_SPLIT_FUSED=0, null remote):
EAGER steps=62 mean=19.632 ms min=18.221 max=29.230; load 5.5s; VRAM 76.21 GiB.

Breaker list (capture of Generator decode model.forward, input (1,1)):
| # | site | symptom | status |
|---|---|---|---|
| 1 | `row_table.py:_stage` `pin.event.synchronize()` (Engram host gather via `ngram_embedding.forward` -> `RowTable.lookup`) | cudaErrorStreamCaptureUnsupported | not fixed; probe-only stub |
| 2 | `exl3_gemv_int8.cu:130` `gemv_int8_get_ws` lazy `cudaMalloc`+`cudaMemset` on first use | fails inside capture if first decode step is captured | fixed by warmup (one eager decode before capture) |

With Engram stubbed (device zeros, probe-only) and the warmup, capture succeeds.

Replay correctness (validate.py, Engram stubbed in both modes, 32 greedy steps):
- steps 0-1 logits match eager exactly (max abs diff 0.0).
- steps 2-31 diverge (max abs diff 5-19), argmax collapses to a repeating token.
- Cause: a single captured `model.forward` bakes in host-side per-step work that replays skip:
  `prepare_inputs`, `m.prefetch`, `advance_recurrent_states` (position/last_history counters),
  and the `bc_dsa` `pos_mirror` branch that uploads `pos_dev`/`rb_dev` only when the host position
  changes. Replay therefore reads stale position and cache state.

Conclusion: whole-step graph is not valid as-is. Required: device-resident position/length
state and static inputs refreshed before each replay, with all host bookkeeping moved out of the
captured region. Not implemented in this pass.
- Staging `input_ids` and params into persistent buffers fails during capture: `prepare_flash_attn` (modules/attn.py:95) builds per-step `cache_seqlens`, positions, and block_table host-side; the embedding's `prepare_for_device` then copies the host tensor to device inside the capture (`RuntimeError: Cannot copy between CPU and CUDA ... not pinned`).
- Conclusion: whole-step capture needs `prepare_flash_attn` inputs (and `input_ids`) to be device-resident persistent buffers refreshed in place. This is outside slice 1. The harness was restored to its original form.

## Piecewise decode graphs (EXL3_WHOLE_GRAPH=1) — WORKING (Main, 2026-10-26)
Design: per TransformerBlock at seq 1: graph A (attn HC mix_norm + attn norm) → eager DSV4Attention core
(compressor/pool/topk/dsa_attn/ring; position-dependent) → graph B (attn HC apply, MLP HC mix_norm + norm,
MLP incl. hot MoE coop + remote issue/collect stub, MLP HC apply). `exllamav3/util/segment.py` Segment:
eager warmup, then capture on the 2nd call into a shared graph pool; tensor args → static inputs; params
dict passes through; `carry=("hc_pre",)` threads the mHC collapse weights between segments through params
(static in/out). resid_defer disabled under the flag (model_ls.py); HC_FOLD stays 0 in this copy.
Fixes needed: Segment warmup returned a bare tensor (now always a tuple); hc_pre carried across segments
(capture-time params copy otherwise left the live params' hc_pre None → next site crashed).

Results (GPU0, null remote, Engram stubbed device-only in both modes):
| check | result |
|---|---|
| G2 short ("The capital of France is", 32 steps) | tokens equal, argmax 32/32, max abs logit diff 0.0 |
| G2 long (~1000-token prompt, 64 steps; crosses ec>512 top-k regime on CSA layers) | tokens equal, argmax 64/64, max abs diff 0.0 |
| decode ms/token (time_eager.py, 62 warm steps) | eager 18.967 → piecewise 14.450 (−24%) |
| launches/step (count_launches.py) | eager 1225 kernels → 445 kernels + 87 graph launches |
Remaining eager launches (~445/step) are the 40 attention cores (~11/layer).
Not yet: real Engram (stub only), real remote seam in graph B, attention core capture.

## Piecewise decode graphs (EXL3_WHOLE_GRAPH=1) — fork port (2026-10-26)

Port of the graph-copy piecewise decode path into the fork. Enabled by `EXL3_WHOLE_GRAPH=1`.
Per TransformerBlock at decode (seq 1): graph A (attn HC mix + norm), eager attention core,
graph B (attn HC apply, MLP HC mix+norm, MLP incl. hot MoE, MLP HC apply).
`_piece_b` ends with `hc_flush(params)` so no deferred HC-fold apply leaks across the segment boundary.
`resid_defer` is disabled under the flag in `model_ls.py`.

Validation (container exl3-fork0, GPU0, EXL3_MOE_SPLIT_FUSED=0 PROBE_STUB_ENGRAM=1):
- G2 short (32 steps): tokens equal, argmax 32/32, max abs logit diff 0.0.
- G2 long (PROMPT_TOKENS=1000, 64 steps): tokens equal, argmax 64/64, max abs logit diff 0.0.

Timing (time_eager.py, NTOK=64):
- eager, HC_FOLD=1: mean 18.919 ms/token
- piecewise, HC_FOLD=1: mean 14.585 ms/token
- piecewise, HC_FOLD=0: mean 14.088 ms/token
- eager, HC_FOLD=0: mean 18.846 ms/token

Launches/step (count_launches.py): eager 1225 kernels; piecewise 445 kernels + 87 graph launches.

Kernel tests: `pytest -q tests/kernels/hyperconnections tests/kernels/moe` → 430 passed.

Note: the image's CUDA_HOME=/usr/local/cuda lacks cusparse.h; container is created with `-e CUDA_HOME=/opt/cuda` (host toolkit mount), which builds.
