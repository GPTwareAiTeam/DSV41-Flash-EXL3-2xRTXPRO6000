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

## M2 results (COMM=p2p; NCCL token output is the reference)
- P2P collectives (`TPBackendP2P`, `P2PComm` from `exllamav3/tp2/p2p.py`): prompt 0 8 tok identical to NCCL; prompts 1 and 2 16 tok identical across ranks (prompt 2 matches the NCCL prefix; full NCCL list for prompt 2 not yet captured).
- Decode ms/tok, eager: P2P 30.07/30.61 (prompt 0), 30.45/30.48 (prompt 1), 28.62/28.61 (prompt 2). NCCL eager 42.77/43.91 (prompt 0).
- P2P + `EXL3_WHOLE_GRAPH=1`: token-identical to eager P2P on all 3 prompts. Decode ms/tok: prompt 1 24.05/24.04, prompt 2 20.81/20.81, prompt 0 32.26/32.27 (8-token run).
- Long-prefill fix: arena limit is 256 KiB, so `all_reduce` chunks to 65536 fp32 per call (sizes depend only on numel, identical on both ranks).
- Head all_gather under P2P: flat gather + last-dim concat, token-identical on all prompts.

## Next steps (M2 step 3, then M3)
1. Profile one decode step (torch profiler, per-category CUDA event timing; ncu is blocked). Attack the largest item.
2. Target >= 150 tok/s (<= 6.67 ms/tok).
