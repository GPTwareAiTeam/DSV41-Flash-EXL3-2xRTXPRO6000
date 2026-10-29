# TP=2 shard spec: DeepSeek-V4.1-Flash EXL3 mixed-2.565

Produced by `qmesh/tp2/shard_tool.py`; checked by `qmesh/tp2/verify_shards.py`.

Source: `/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.565-tp2` (41 shards, 192,478 tensors, read-only).
Output: `/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.565-tp2-shard2/{rank0,rank1}/` (rank r = cuda:r).

## Principle

Every source tensor maps to exactly one rule. Split rules cut tensors so rank0 and rank1 hold disjoint
halves that concatenate back to the source. Replicated tensors are identical on both ranks.
The tool fails loudly (`ShardError`) on any module not matched by exactly one rule.

## Roles (by module name)

| Role | Modules | Split rule |
|---|---|---|
| `out` | routed + shared expert `w1`/`w3`; `head`; `embed`; `aligner.w1`; `markov_w1/w2`; `attn.wq_b` | output dim split |
| `in` | routed + shared expert `w2`; `attn.wo_b`; `aligner.w2`; `vision.*.attn.wo`; `vision.*.mlp.w2` | input dim split |
| `vfused` | `vision.*.attn.wqkv`; `vision.*.mlp.w1` | fused layout, per-part split (below) |
| `sink` | `*.attn.attn_sink` (F32 `[64]`) | dim-0 split |
| `woa` | `*.attn.wo_a.slice.k` (k=0..7) | renamed; slice k lives on rank k//4 as `slice.(k%4)` |
| `rep` | everything else (see below) | replicated |

### EXL3 part rules (`out` and `in`)

EXL3 tensors are `trellis`, `suh`, `svh`, `mul1` (plus `bias` on linear layers).

- `out`: `trellis` split dim 1 (`[K, N, 16K]` → `[K, N/2, 16K]`); `svh` and `bias` split dim 0; `suh` replicated; `mul1` copied whole.
- `in`: `trellis` split dim 0; `suh` split dim 0; `svh` replicated; `bias` on **rank0 only** (rank1 has no `bias` entry).

Trellis tiers, by last dim (16·K): 32 (K=2), 40 (K=2.5), 48 (K=3), 56 (K=3.5). Shared-expert w2 has last dim 96 (K=6).

### Replicated (`rep`)

`engram.wkv` (`layers.{1,14}`, F16 `[25600,6144]`), `mtp.N.main_proj.*`, `confidence_head`, `hc_*`,
norms (`*norm*`), `ffn.gate.*` (incl. `bias_vl`), compressor and indexer (including `attn.indexer.wq_b`),
`attn.wq_a`, `attn.wkv`, `kv_norm`, `q_norm`, vision norms and patch_embed, `head.mul1`, every `mul1`.

### Vision fused layouts

- `wqkv.weight` `[3072,1024]` and `wqkv.bias` `[3072]`: rows are q(1024) | k(1024) | v(1024). Rank r takes rows
  `[r·512:(r+1)·512]` of each part, concatenated in q,k,v order → `[1536,1024]`.
- `mlp.w1.weight` `[5632,1024]`: rows are gate(2816) | up(2816). Rank r takes `[r·1408:(r+1)·1408]` of gate and
  of up, concatenated → `[2816,1024]`.

## Deviations found in the header scan (not in the original design)

- MTP has **128 routed experts per layer** (`mtp.N.ffn.experts.0..127`, 3 layers, 384 total), trellis last dim 64 (K=4).
- MTP `ffn.gate` is `[128,5120]` with no `bias_vl`.
- `head.mul1` is an I32 scalar. No `mcg` tensors exist.
- Trunk `attn.indexer.wq_b` (`[80,256,96]` trellis, `suh [1280]`, `svh [4096]`) is absent from the design's list.
  Treated as replicated, consistent with the design's "indexer REPLICATED" rule.
- `attn.wo_a` is stored as 8 slices (`slice.0..7`, each trellis `[256,64,96]`, `suh [4096]`, `svh [1024]`).
  Slice k goes to rank k//4 as `slice.(k%4)`, with its name renamed.
- `attn.wq_b` is `trellis [80,2048,96]`, `svh [32768]`, `suh [1280]`; OUT split → 16384 per rank.
- `attn.attn_sink` F32 `[64]` → `[32]` per rank.
- Engram `wkv` is replicated (the hc streams are not head-split).

## Output layout

- Weights: `model-XXXXX-of-NNNNN.safetensors`, about 4 GiB per file (`--bin-gib 4`), 23 files per rank.
  Each file's metadata is `{"format":"pt","tp_rank":r,"tp_size":2}`.
- `model.safetensors.index.json`: `metadata.total_size` = sum of tensor bytes; `weight_map` for this rank only.
- `config.json`: source config plus `tp_rank`, `tp_size` = 2, and `tp_shard_spec` = `qmesh/tp2/SHARD_SPEC.md`.
  The tool refuses if the source already has these keys.
- `shard_manifest.json`: per dst tensor, `src_name`, `dst_shape`, `rule`, and split pieces.
- Copied as-is: `tokenizer*`, `*.jinja`, `special_tokens_map.json`, `generation_config.json`, `quantization_config.json`.

## Verification (`verify_shards.py`)

1. Accounting: every source tensor placed per the rules on each rank; no extras; index == manifest == headers.
2. Shapes and dtypes match spec-expected shapes and the manifest.
3. Per-rank tensor bytes == index `total_size`.
4. Byte-exact reconstruction (rank0 ⊕ rank1 == source) for ≥64 routed experts (16 per K tier, all layers incl. MTP)
   plus shared experts, wq_b, wo_b, wo_a slices, head, embed, markov, aligner, sinks, and vision fused layers.
   `verify_shards.py` re-derives roles and shapes itself and does not reuse `shard_tool.plan()`.

## External DSpark drafter overlay

`qmesh/tp2/drafter_overlay.py` converts diffbot's MIT-licensed `drafter-code-k7` pack into a separate TP2 overlay without modifying target weights. Pass `--drafter` for the pack directory, `--base` for the existing rank overlay, `--index-root` for the original sharded checkpoint, and `--out` for a new directory. `--block-size 7` sets `dspark_block_size` in the copied rank configs; `--verify` reconstructs mapped drafter tensors from both ranks. The converter retains non-MTP tensors in rewritten shards and symlinks other files, so the base must remain available.

For the runner, `DSPARK_HEAD=ours` (default) uses `OVERLAY`; `DSPARK_HEAD=diffbot` uses `DSPARK_DIFFBOT_OVERLAY` (default `/work/k7db_overlay`). Both rank directories and the symlinked base must remain available. Set `NUM_DRAFT=7` with a block-7 overlay; a block-5 configuration limits the effective draft length to five.

`launch_tp2.sh` pins rank 0 to CPUs `2-5` and rank 1 to `10-13` by default using `taskset`. These are distinct physical cores on NUMA node 0 on the measured host; choose appropriate cores for other hosts using the host environment variables `QMESH_CPUSET_RANK0` and `QMESH_CPUSET_RANK1`. Set either variable to an empty string to disable pinning for that rank. The container must provide `taskset` and allow the selected CPUs.
