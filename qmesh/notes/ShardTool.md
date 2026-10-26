# ShardTool: TP=2 pre-shard of DeepSeek-V4.1-Flash EXL3 mixed-2.565

## Artifacts
- Tool: `qmesh/tp2/shard_tool.py`
- Verifier: `qmesh/tp2/verify_shards.py`
- Spec: `qmesh/tp2/SHARD_SPEC.md`
- Source (read-only): `/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.565-tp2` (41 shards, 192,478 tensors)
- Output: `/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.565-tp2-shard2/{rank0,rank1}/`

## Commands
Dry run (plan only, ~7 s):
```
/home/treadall/.venvs/shardtool/bin/python qmesh/tp2/shard_tool.py --src <SRC> --out <OUT> --dry-run
```
Real run (final, successful):
```
/usr/bin/env time -v /home/treadall/.venvs/shardtool/bin/python qmesh/tp2/shard_tool.py --src <SRC> --out <OUT>
```
Verify:
```
/home/treadall/.venvs/shardtool/bin/python qmesh/tp2/verify_shards.py --src <SRC> --out <OUT>
```
Host venv `/home/treadall/.venvs/shardtool`: safetensors 0.8.0, numpy 2.5.4, torch 2.14.1+cpu. CPU-only, so the
`exl3-tp2` container (Exited(2), entrypoint `vllm serve`) is not used.

## Runs
1. First attempt (log `shard_run.log`): planning and writing ran through source shard 18 of 41, then failed in
   `write_rank_shards` with `TypeError: cannot unpack non-iterable NoneType`. Cause: the function had no `return`.
   Fixed by adding `return weight_map, manifest`. Output dir removed (only `*-tp2-shard2`).
2. Final run (log `shard_run2.log`): 110.92 s wall, peak RSS 16.99 GB, exit 0. `rank0 finalized`, `rank1 finalized`, `done in 110.0s`.

## Plan
- 192,478 source tensors, every one matched by exactly one rule (`exl3-in` 63,320; `exl3-out` 126,472; `plain-in` 98;
  `plain-out` 5; `rep` 1,068; `sink-split` 43; `vision-mlp-w1` 32; `vision-wqkv` 64; `wo_a-rename` 1,376).
- rank0: 191,790 tensors, 23 bins. rank1: 191,757 tensors, 23 bins.

## Sizes
- rank0: 96,433,262,704 B = 89.810 GiB (191,790 tensors); on-disk 96,587,489,328 B (`du -sb`, includes safetensors headers).
- rank1: 96,433,186,928 B = 89.810 GiB (191,757 tensors); on-disk 96,587,398,003 B.
- |rank0 - rank1| = 75,776 B (differs by the rank0-only IN biases).
- Source total 177.920 GiB (header-computed). Replicated overhead = 1.700 GiB.
- Each rank: 23 weight files + index + config.json + shard_manifest.json + copied tokenizer files.

## Disk on /models/dsv41
- Before: 215 GiB free (94% used).
- After: 35 GiB free (100% used, 3.5 T of 3.7 T). The output consumed about 180 GiB.

## Verification (`verify_shards.py`, exit 0, "VERIFY PASSED")
- Accounting: all 192,478 source tensors placed per spec on each rank; no extras; index == manifest == headers.
- Shapes and dtypes match spec and manifest.
- Per-rank tensor bytes == index `total_size`.
- Byte-exact reconstruction: 1,040 tensors compared, covering 80 routed expert bases (16 per K tier across 5 tiers:
  K=2 5500 experts, K=2.5 4860, K=3 2500, K=3.5 2500, K=4 384 MTP) plus shared experts, wq_b, wo_b, wo_a slices, head,
  embed, markov, aligner, sinks, and vision fused layers.

## Known deviations
See `qmesh/tp2/SHARD_SPEC.md` "Deviations". Key items: MTP has 128 experts per layer (K=4), `attn.indexer.wq_b`
replicated, and `wo_a` stored as 8 slices.

## Not done
- No GPU run. The `exl3-tp2` container is Exited(2) and needs a restart before any runtime test.
