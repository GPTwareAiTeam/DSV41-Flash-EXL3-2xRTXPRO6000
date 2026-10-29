# HostSlim: TP=2 decode host-cost pass

> Raw working notes from the development runs, kept because they hold the measurements. Host paths, container names and run tags refer to that machine; the results themselves are restated with conditions in [`../RESULTS.md`](../RESULTS.md).

Scope: token-identical host-side wins for TP=2 DeepSeek-V4.1 decode. Binding constraint is GPU work (MoE at verify width); host wins capped near ~2 ms/step.

## Profile (HEAD `1b82a32a`, tag `hs_prof5`, 15 decode steps, PROMPT_IDS=0, WHOLE_GRAPH=1, PROFILE=1)

Unprofiled 13.33 ms/tok. Profiled wall 20.69 ms/step (profiler overhead). Self CPU, top 20:

| Op | Self CPU (ms, 15 steps) | % | Calls note |
|---|---|---|---|
| cudaLaunchKernel | 27.97 | 23.6 | 3.8 us avg |
| cudaGraphLaunch | 15.32 | 12.9 | 80/step |
| aten::empty | 12.23 | 10.3 | allocation churn |
| cudaMemcpyAsync | 6.07 | 5.1 | |
| aten::_foreach_copy_ | 4.70 | 4.0 | 1 per segment replay |
| aten::copy_ | 4.23 | 3.6 | |
| cuLaunchKernelEx | 4.21 | 3.6 | |
| Activity Buffer Request | 3.72 | 3.1 | profiler |
| aten::empty_strided | 3.69 | 3.1 | allocation churn |
| aten::slice | 3.57 | 3.0 | |
| aten::select | 3.17 | 2.7 | |
| aten::view | 3.07 | 2.6 | |
| cudaLaunchCooperativeKernel | 3.04 | 2.6 | |
| aten::as_strided | 2.84 | 2.4 | |
| cudaDeviceSynchronize | 2.04 | 1.7 | |
| aten::bmm | 1.91 | 1.6 | |
| aten::mm | 1.75 | 1.5 | |
| aten::reshape | 1.43 | 1.2 | |
| aten::index | 0.90 | 0.8 | |
| aten::arange | 0.87 | 0.7 | |

Trace: `tp2logs/hs_prof5_trace.json`.

## Attempts and results

| Change | Result | Status |
|---|---|---|
| Event-sync replacement at `generator.py:1817` | 16.22 vs 16.17 ms/tok (no win) | reverted |
| Engram `.to("cpu")` (`deepseek_v41.py`) | no-op: input already pinned | not a target |
| Item 1: fused wo_a under TP | already active (`exl3_mgemm` single launch per layer, 40/40) | premise confirmed |
| Item 2: fused x-fan | `_build_x_fan` returns early (`q_head_norm=False`, all 40 layers); fan is not bit-identical (logit diff 0.5–0.8) | `EXL3_DSV41_XFAN=1` opt-in; tokens unstable (first divergence @55/@97/@68); 13.08/13.17/13.25 vs 13.47/13.71/13.73 ms/tok. Rejected as opt-in |
| Per-layer index hoisting (arange/index/index_put) | ceiling ~0.14 ms/tok (~2.1 ms over 15 steps) | not built |
| Allocation churn: decode attention `out` and non-kv-source `pool_c`/`pool_r` via `g_tensor_cache` | 13.17/13.45/13.42 vs 13.43/13.63/13.69 ms/tok (~0.2 ms/tok). Tokens identical to baseline, ranks 0/1, prompts 0/1/2 | **committed `9934184e`** |
| Graph launch (a): skip per-replay `_foreach_copy_` | not feasible as a pure alias: seg_b's `x` is layer N's fresh output, not seg_a(N+1)'s static buffer. Would need cross-layer buffer ownership | not built |
| Graph launch (b): merge B(N) with A(N+1) | blocked by eager `self.attn.forward` between them (`transformer.py` `_forward_piecewise`). Requires attention capture, out of scope tonight | not built |

Baseline token reference: `xbase_tokens.rank{0,1}.json` (3 prompts x 128, IGNORE_EOS=1).

## Notes
- `run_tag.sh` takes `gpu.lock` itself; do not wrap it in an outer `flock` (self-deadlock).
- Token files are named `<tag>_tokens.json.rank{0,1}.json` in some runs and `<tag>_tokens.rank{0,1}.json` in others; check with `ls`.
