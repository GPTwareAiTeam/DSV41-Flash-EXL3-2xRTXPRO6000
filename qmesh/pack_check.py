#!/usr/bin/env python3
"""Pre-flight check of a reduced serving pack against the full checkpoint it was cut from.

Loads both safetensors indices (a few MB, no tensor bytes) and reports what the pack is missing,
what it adds, and whether every shard the index names exists and is non-empty. A hot-only pack is
expected to omit cold-expert tensors; any *other* missing tensor is a hard error for the loader.

Usage: pack_check.py <pack_dir> [<reference_dir>]
Env:   PACK_WIDE=1 lists every missing name instead of a per-class count
"""
from __future__ import annotations

import json
import os
import re
import sys


def index_of(path: str) -> dict:
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as fh:
            return json.load(fh)["weight_map"]
    # Single-shard pack: enumerate the one file
    for name in sorted(os.listdir(path)):
        if name.endswith(".safetensors"):
            return {name: name}
    return {}


def shard_sizes(path: str, shards) -> list:
    out = []
    for sh in sorted(set(shards)):
        p = os.path.join(path, sh)
        out.append((sh, os.path.getsize(p) if os.path.exists(p) else -1))
    return out


EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w[123])\.(\w+)$")


def classify(names, cold_counts):
    """Group omitted tensor names by whether a hot-only pack may legitimately drop them.

    The shipped pack keeps every expert's mul1/suh/svh and drops only trunk routed-expert
    `trellis` for the cold experts (plus the first cold expert's, which the loader probes for
    shape). So `trellis` omissions are expected up to the layer's cold count; anything else is a
    hard failure for load_cpu_split or for the hot slice.
    """
    out = {"cold_trellis": [], "hot_trellis": [], "expert_other": [], "nonexpert": []}
    for n in names:
        m = EXPERT.match(n)
        if not m:
            out["nonexpert"].append(n)
            continue
        layer, _eid, _w, kind = m.groups()
        cold = cold_counts.get(int(layer))
        if kind == "trellis":
            out["cold_trellis"].append((int(layer), n))
        elif kind in ("mul1", "suh", "svh"):
            out["expert_other"].append(n)
        else:
            out["expert_other"].append(n)
    # Per-layer trellis budget: at most the layer's cold count (one is kept for the probe)
    per_layer = {}
    for layer, _n in out["cold_trellis"]:
        per_layer[layer] = per_layer.get(layer, 0) + 1
    over = []
    for layer, cnt in sorted(per_layer.items()):
        cold = cold_counts.get(layer)
        # 384 experts x 3 projections = the per-expert tensor triple count
        if cold is None or cnt > 3 * cold:
            over.append((layer, cnt, cold))
    out["over_budget"] = over
    return out


def main():
    pack = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else None
    if not os.path.isdir(pack):
        print(f"PACK_FAIL no_dir={pack}")
        return 2
    pidx = index_of(pack)
    print(f"PACK index tensors={len(pidx)}")
    bad = [sh for sh, size in shard_sizes(pack, pidx.values()) if size <= 0]
    print(f"PACK shards={len(set(pidx.values()))} missing_or_empty={len(bad)}")
    for sh in bad[:10]:
        print(f"PACK_BAD_SHARD {sh}")
    total = 0
    for name in sorted(os.listdir(pack)):
        p = os.path.join(pack, name)
        if os.path.islink(p):
            p = os.path.realpath(p)
        if os.path.isfile(p):
            total += os.path.getsize(p)
    print(f"PACK bytes={total / 1e9:.1f}GB ({total / 2**30:.1f}GiB)")
    for extra in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        ok = os.path.exists(os.path.join(pack, extra))
        print(f"PACK file {extra}={'yes' if ok else 'NO'}")
    if not ref:
        return 0 if not bad else 1
    ridx = index_of(ref)
    missing = sorted(set(ridx) - set(pidx))
    added = sorted(set(pidx) - set(ridx))
    pl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "placement.final.json")
    cold_counts = {}
    if os.path.exists(pl_path):
        pl = json.load(open(pl_path))
        for layer, entry in pl["layers"].items():
            cold_counts[int(layer)] = len(entry.get("warm") or [])
    else:
        print(f"PACK_WARN no_placement={pl_path}")
    groups = classify(missing, cold_counts)
    print(f"PACK vs {ref}: ref_tensors={len(ridx)} omitted={len(missing)} added={len(added)}")
    for key in ("cold_trellis", "hot_trellis", "expert_other", "nonexpert"):
        val = groups[key]
        print(f"PACK omitted_{key}={len(val)}")
    for layer, cnt, cold in groups["over_budget"]:
        print(f"PACK_TRELLIS_OVER_BUDGET layer={layer} omitted={cnt} cold={cold}")
    cold_total = sum(cold_counts.values())
    print(f"PACK cold_experts_expected={cold_total} trellis_omitted={len(groups['cold_trellis'])}"
          f" (expected about {3 * cold_total} names, one kept per layer for the probe)")
    for n in groups["expert_other"][:20] + groups["nonexpert"][:20]:
        print(f"PACK_UNEXPECTED_OMISSION {n}")
    for n in added[:10]:
        print(f"PACK_EXTRA {n}")
    ok = not bad and not groups["expert_other"] and not groups["nonexpert"] \
        and not groups["over_budget"]
    print("PACK_OK" if ok else "PACK_NOT_READY")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
