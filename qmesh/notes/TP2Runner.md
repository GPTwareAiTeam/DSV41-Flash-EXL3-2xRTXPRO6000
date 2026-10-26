# TP2Runner status (milestone 1, NOT complete)

## Done
- Overlay: `/home/treadall/recipe-artifacts/2x-rtx-tp2/tp2-overlay/rank{0,1}` (host) = `/artifacts/tp2-overlay/rank{0,1}` (container). 23 rank shard symlinks, `engram-only-L01/L14` symlinks, rank `config.json`. Verified visible in `exl3-tp2`.
- Arch (`exllamav3/architecture/deepseek_v41.py`): local dims for tp_size=2 (num_q_heads 32, o_groups 4, moe_intermediate 1152); embed `vocab_size` and head `out_features` = `config.vocab_size // config.tp_size` (64640). Checkpoint `config.json` has `tp_size: 2`, `tp_rank` per rank.
- Runner `qmesh/tp2/run_tp2.py`: loads rank dir via `Model.load(device=cuda:RANK)`, `TPBackendNCCL` (all_reduce/broadcast/barrier), sets `tp_reduce=True` on attn and mlp of each TransformerBlock. Decode loop, vocab all-gather of head logits, and embed masking are NOT written yet.

## Blocker
- `import exllamav3` in `exl3-tp2` blocks on a JIT build of `exllamav3_ext`. The image sets `TORCH_EXTENSIONS_DIR=/opt/vllm-moet/torch_extensions`, which is shared across exl3-tp2 / exl3-tp2-moe / exl3-tp2-p2p (same inode). Concurrent builds collide.
- Use `TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions` (private per container) for every build and run.
- A private-cache build was in progress (~125 objects, many nvcc jobs) and had not produced `exllamav3_ext.so` when stopped. It was stopped as instructed, so the cache is partial and must be rebuilt on the next run.

## Not yet run
- No full-model load has completed. No token output exists. No numbers to report.

## Next steps
1. Finish `run_tp2.py`: vocab-split embed (mask + all_reduce), head logits all_gather to 129280, greedy decode with `Generator`-equivalent loop, using `text`/`eos` from `Generator.iterate()`.
2. Build the extension with `TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions`, then launch 2 ranks (`RANK`, `WORLD_SIZE=2`, `INIT_METHOD=file://...`, `OVERLAY=/artifacts/tp2-overlay`), logs to `/artifacts/tp2logs/`.
3. Run 3 prompts, confirm coherent text.
