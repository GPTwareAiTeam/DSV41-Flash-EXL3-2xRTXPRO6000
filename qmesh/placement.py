#!/usr/bin/env python3
"""Build the P7/GX1 whole-expert placement for an EXL3 checkpoint.

P7 owns the 4,100 hottest 4.0-bpw experts. GX1 owns every other expert.
Sizes are read from the checkpoint's safetensors headers, not estimated.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import re
import struct

EXPERT_KEY = re.compile(
    r"layers\.(\d+)\.ffn\.experts\.(\d+)\.w[123]\.(?:suh|svh|mul1|trellis)$"
)
GIB = 2**30


def tensor_sizes(checkpoint: str) -> dict[str, int]:
    index_path = os.path.join(checkpoint, "model.safetensors.index.json")
    index = json.load(open(index_path, encoding="utf8"))
    files = collections.defaultdict(list)
    for key, filename in index["weight_map"].items():
        files[filename].append(key)
    sizes: dict[str, int] = {}
    for filename, keys in files.items():
        path = os.path.join(checkpoint, filename)
        if not os.path.isfile(path):
            continue
        with open(path, "rb") as f:
            header_len = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_len))
        for key in keys:
            meta = header.get(key)
            if meta is not None:
                start, end = meta["data_offsets"]
                sizes[key] = end - start
    return sizes


def build(checkpoint: str, ranking_csv: str, hot_experts: int, p7_gib: float,
          kv_gib: float, runtime_gib: float, gx1_gib: float) -> dict:
    sizes = tensor_sizes(checkpoint)
    expert_bytes: dict[tuple[int, int], int] = collections.defaultdict(int)
    dense_bytes = 0
    non_layer_bytes = 0
    for key, size in sizes.items():
        match = EXPERT_KEY.match(key)
        if match:
            expert_bytes[(int(match.group(1)), int(match.group(2)))] += size
        elif key.startswith("layers."):
            dense_bytes += size
        else:
            non_layer_bytes += size

    ranking = []
    rates = {}
    with open(ranking_csv, newline="", encoding="utf8") as f:
        for row in csv.DictReader(f):
            layer = int(row["layer"])
            expert = int(row["expert_id"])
            ranking.append((layer, expert, int(row["rank"])))
    if len(ranking) != len(expert_bytes):
        raise SystemExit(f"ranking has {len(ranking)} experts, checkpoint has {len(expert_bytes)}")
    if len(ranking) < hot_experts:
        raise SystemExit("hot_experts exceeds ranking size")

    hot = ranking[:hot_experts]
    warm = ranking[hot_experts:]
    hot_bytes = sum(expert_bytes[(layer, expert)] for layer, expert, _ in hot)
    warm_bytes = sum(expert_bytes[(layer, expert)] for layer, expert, _ in warm)
    non_expert = (dense_bytes + non_layer_bytes) / GIB
    p7_total = non_expert + kv_gib + runtime_gib + hot_bytes / GIB
    gx1_total = warm_bytes / GIB

    if p7_total > p7_gib:
        raise SystemExit(f"P7 does not fit: {p7_total:.3f} > {p7_gib:.3f} GiB")
    if gx1_total > gx1_gib:
        raise SystemExit(f"GX1 does not fit: {gx1_total:.3f} > {gx1_gib:.3f} GiB")

    hot_set = {(layer, expert) for layer, expert, _ in hot}
    by_layer: dict[int, dict[str, list]] = {}
    for layer in range(40):
        hot_layer = sorted(e for (l, e) in hot_set if l == layer)
        cold_layer = sorted(e for (l, e) in expert_bytes if l == layer and (l, e) not in hot_set)
        by_layer[str(layer)] = {
            "hot": hot_layer,
            "warm": cold_layer,
            "hot_bytes": sum(expert_bytes[(layer, e)] for e in hot_layer),
            "warm_bytes": sum(expert_bytes[(layer, e)] for e in cold_layer),
        }

    return {
        "schema": "qmesh.exl3-expert-placement/v1",
        "checkpoint": os.path.abspath(checkpoint),
        "ranking": os.path.abspath(ranking_csv),
        "hot_experts": hot_experts,
        "warm_experts": len(warm),
        "hot_bytes": hot_bytes,
        "warm_bytes": warm_bytes,
        "hot_gib": hot_bytes / GIB,
        "warm_gib": warm_bytes / GIB,
        "dense_layer_gib": dense_bytes / GIB,
        "non_layer_gib": non_layer_bytes / GIB,
        "p7_budget_gib": p7_gib,
        "p7_total_gib": p7_total,
        "gx1_budget_gib": gx1_gib,
        "gx1_total_gib": gx1_total,
        "layers": by_layer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--ranking", default="/home/treadall/qmesh-exl3-mixed/ranking/expert_ranking_2.25.csv")
    ap.add_argument("--hot-experts", type=int, default=4100)
    ap.add_argument("--p7-gib", type=float, default=94.97)
    ap.add_argument("--kv-gib", type=float, default=5.5)
    ap.add_argument("--runtime-gib", type=float, default=3.365)
    ap.add_argument("--gx1-gib", type=float, default=117.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plan = build(args.checkpoint, args.ranking, args.hot_experts, args.p7_gib,
                 args.kv_gib, args.runtime_gib, args.gx1_gib)
    with open(args.out, "w", encoding="utf8") as f:
        json.dump(plan, f, indent=2)
    print(json.dumps({k: v for k, v in plan.items() if k != "layers"}, indent=2))


if __name__ == "__main__":
    main()
