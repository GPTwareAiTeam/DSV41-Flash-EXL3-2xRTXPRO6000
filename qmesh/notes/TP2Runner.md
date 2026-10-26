# TP2Runner status (milestone 1 complete; M2 next)

## Done
- Overlay: `/home/treadall/recipe-artifacts/2x-rtx-tp2/tp2-overlay/rank{0,1}` (host) = `/artifacts/tp2-overlay/rank{0,1}` (container). Rank shard symlinks, `engram-only-L01/L14` symlinks, rank `config.json`.
- Arch (`exllamav3/architecture/deepseek_v41.py`): local dims for tp_size=2 (num_q_heads 32, o_groups 4, moe_intermediate 1152); embed and head vocab split = `vocab_size // tp_size` (64640).
- Runner `qmesh/tp2/run_tp2.py` + `qmesh/tp2/launch_tp2.sh`: per-rank pre-sharded load, `TPBackendNCCL` all_reduce on attn/MoE partial sums, vocab-split embed (mask + all_reduce), head all_gather, greedy decode.
- M1 fix (commit `51102429`): the fused bszN MoE path produced zero routed output. Root cause in `BC_BlockSparseMLP::run_bszN`: the folded routed+shared launch writes `fold_out` (= shared `out_d_sh` scratch), but Python reads `experts_cfg.out_bszn`, which was never written. Fix copies the H-wide trim into `out_bszn`. Also `fold_prepare()` now reattaches `k2_g/u/d` (mixed-K) on every rebuild.

## M1 result (MAX_NEW=16 unless noted, bsz 1, greedy)
- Prompt 0 "The capital of France is" -> "**Paris**." Ranks identical: `[671, 6102, 294, 8760, 344, 2619, 51119, 42499]`.
- Prompt 1 "# Fibonacci function in Python" -> coherent code-doc text. Ranks identical.
- Prompt 2 "List three primary colors..." -> "The three primary colors in traditional painting (specifically in the subtractive color model". Ranks identical.
- Reference: `/artifacts/tp2logs/bench_native_tsplit.log` (native tensor-split, coherent).

## Baseline (eager, no CUDA graphs, NCCL)
- Prompt 2, 16 tokens: decode 38.02 ms/token, first token 1291 ms.
- Prompt 0, 8 tokens (clean run): decode 42.77 / 43.91 ms/token (rank0/rank1), first token ~1120 ms.
- Target >= 150 tok/s = <= 6.67 ms/token.

## Blocker history (resolved)
- `import exllamav3` JIT-builds `exllamav3_ext`. Always `export TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions` (private per container).
- Do not run two launches at once on the same GPUs: concurrent runs OOM.

## Next steps (M2)
1. Swap NCCL for `P2PComm` (`tp2/p2p-allreduce`): `all_reduce_` for attn/MoE/embed, `all_gather` for head. Compare tokens against NCCL.
2. `EXL3_WHOLE_GRAPH=1` piecewise CUDA graphs.
3. Profile one decode step; attack the largest item.
