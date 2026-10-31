# Recipe: the checkpoint

This is the recipe for the model this runtime serves: a **mixed-bitrate EXL3 quantization of
DeepSeek-V4.1-Flash-1017 (TP=2)** sized to fit two 96 GB cards at ~2.565 bits per weight, and the exact
process used to produce it. The runtime side is described in [`DESIGN.md`](DESIGN.md); the measured
behaviour in [`RESULTS.md`](RESULTS.md).

## Hardware target

| Device | Role |
|---|---|
| RTX PRO 6000 Blackwell Max-Q (96 GB) | TP rank 0 (+ output device) |
| RTX PRO 6000 Blackwell Max-Q (96 GB) | TP rank 1 |
| Link | PCIe P2P GPU0↔GPU1 (same NUMA node) |

Any two 96 GB SM120 cards on the same NUMA node will satisfy the memory and link requirements; the
measured numbers in [`RESULTS.md`](RESULTS.md) are from this pair.

## Checkpoint

| Item | Value |
|---|---|
| Base model | DeepSeek-V4.1-Flash-1017 (upstream weights, read-only) |
| Quantizer | ExLlamaV3 `convert.py` (EXL3 trellis) |
| Expert bitrate mix | 2,500 experts @ 3.5 bpw · 2,500 @ 3.0 · 4,860 @ 2.5 · 5,500 @ 2.0 |
| Average | **2.565 bpw** |
| Mixed per layer | yes — every MoE layer carries the full 3.5 → 2.0 spread |
| Single-device checkpoint | 177.92 GiB, 41 shards (EXL3 + external Engram tables) |
| Output (TP=2, pre-sharded offline) | `<models>/dsv41-exl3-mixed-2.565-shard2/{rank0,rank1}/` |
| Output size | ~179 GiB total (safetensors + index + config + `shard_manifest.json`) |

The bitrate mix is chosen by *expert hotness*: experts that fire most often get more bits, cold experts fewer.
Quantization error therefore lands where it is least likely to be read, which is why the average can sit at
2.5 bpw without the quality collapse a uniform 2.5 bpw checkpoint shows. Hotness comes from Strata's routing
profile (Niko1221/Strata, `data/dsv41/profile-2.25.bin`, commit `81dd215d`): all 15,360 trunk experts are ranked
by selection count and the ranking is cut into tiers (2.499: 1,400 @ 4, 4,860 @ 3, 9,100 @ 2 bpw). Bitrates are
assigned at conversion time; nothing is requantized or re-ranked at load time.

Engram tables (layers 1 and 14) stay **outside** the shards: FP8 E4M3 + E8M0 scale, held in host DDR5
ECC RAM as one shared read-only copy for both ranks.

## Reproducing the TP=2 checkpoint

Two CPU-only steps, no GPU required. Both derive everything from
[`SHARD_SPEC.md`](SHARD_SPEC.md) (the per-tensor split rules: which tensors are column-split,
row-split or replicated).

```bash
# 1. Plan only: scan headers, print the split plan, write nothing.
python -m qmesh.tp2.shard_tool \
    --src  <models>/dsv41-exl3-mixed-2.565 \
    --out  <models>/dsv41-exl3-mixed-2.565-shard2 \
    --bin-gib 4 --dry-run

# 2. Real run: writes rank0/ rank1/ with safetensors + index + config + shard_manifest.json.
python -m qmesh.tp2.shard_tool \
    --src  <models>/dsv41-exl3-mixed-2.565 \
    --out  <models>/dsv41-exl3-mixed-2.565-shard2 \
    --bin-gib 4

# 3. Verify: accounting, shapes, bytes, and byte-exact reconstruction on a sample of
#    >= 64 routed experts (16 per K tier, all layers mixed, including MTP) plus shared
#    experts, attention, head, embedding, vision and aligner tensors.
python -m qmesh.tp2.verify_shards \
    --src <models>/dsv41-exl3-mixed-2.565 \
    --out <models>/dsv41-exl3-mixed-2.565-shard2
```

`verify_shards.py` re-derives the expected plan for itself rather than importing `shard_tool`'s planner,
so a bug in the planner cannot pass by agreeing with itself. It exits 0 only if every check passes.

Rules the tool enforces, and the reason each cut is safe:

- Every EXL3 cut is a multiple of 128 (trellis tile granularity) so the trellis slices stay exact.
- `trellis` / `suh` / `svh` / `mul1` / `bias` are split **together** per rule; `suh` is replicated for
  column splits and `svh` for row splits, never both split.
- Any tensor not matched by an explicit rule raises `ShardError`. There is no silent default — a new
  tensor from a future checkpoint revision fails loudly instead of being shipped wrongly.

After sharding, the serving overlay (rank dirs + Engram links) is just those two directories; the runner
finds them through `OVERLAY`.

## Engram tables

`engram-only-L01.safetensors` and `engram-only-L14.safetensors` must be present in the overlay directory
(or wherever `OVERLAY` points). They are memory-mapped once per host, not per rank; the two ranks share
the same pages. See [`env_vars.md`](env_vars.md) for the host-memory sizing.
