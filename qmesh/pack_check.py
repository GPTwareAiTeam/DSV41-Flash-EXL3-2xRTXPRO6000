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


def classify(names):
    """Group omitted tensor names by what the loader can substitute for."""
    hot = re.compile(r"^model\.layers\.\d+\.(mlp\.experts|ffn)\.(gate|up|down)")
    small = re.compile(r"(mul1|suh|svh|trellis_rate|bias|scale)")
    out = {"hot_experts": [], "cold_small": [], "other": []}
    for n in names:
        if hot.search(n):
            out["hot_experts"].append(n)
        elif small.search(n):
            out["cold_small"].append(n)
        else:
            out["other"].append(n)
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
    groups = classify(missing)
    print(f"PACK vs {ref}: ref_tensors={len(ridx)} omitted={len(missing)} added={len(added)}")
    for key, names in groups.items():
        print(f"PACK omitted_{key}={len(names)}")
        if os.environ.get("PACK_WIDE") == "1":
            for n in names[:200]:
                print(f"  {key} {n}")
    if groups["other"]:
        print("PACK_UNEXPECTED_OMISSIONS")
        for n in groups["other"][:20]:
            print(f"  other {n}")
    for n in added[:10]:
        print(f"PACK_EXTRA {n}")
    print("PACK_OK" if not bad and not groups["other"] else "PACK_NOT_READY")
    return 0 if (not bad and not groups["other"]) else 1


if __name__ == "__main__":
    sys.exit(main())
