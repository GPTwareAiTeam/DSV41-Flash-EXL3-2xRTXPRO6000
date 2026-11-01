#!/usr/bin/env python3
"""Build the HC1 hot/cold DSpark drafter overlay for the single-device P7 production layout.

The v1.2 recipe cuts the drafter's routed-expert VRAM by keeping the experts that cover 99% of
the drafter's routing picks at their shipped rate and re-quantizing the rest at 2 bits
(qmesh/tp2/quant_drafter_experts.py + qmesh/tp2/drafter_experts.py, which only speaks TP2 shard
layouts). This is the same construction for one device: experts are assigned from the shipped
usage table, every expert takes all of its tensors (w1/w2/w3 trellis, suh, svh, mul1) from ONE
uniform-rate source set, and the result is written as a plain overlay:

  OUT/  every file of BASE symlinked verbatim, except
        * the safetensors shards that hold drafter routed-expert keys: rewritten copies with
          those keys removed (all other tensors byte-identical)
        * model-drafter-experts.safetensors: the per-expert tensors at their assigned rates,
          so every key exists exactly once. The runtime scans the directory, so no index update
          is needed.

Env:
  BASE    source checkpoint dir (default /models/dsv41/exl3-serve-final)
  OUT     overlay dir to create (default BASE + "-hc1")
  USAGE   usage table (default qmesh/tp2/data/drafter_expert_usage.json)
  TIERS   tiers, hottest first (default n99:4,rest:2)
  SET2    safetensors holding every routed-expert tensor at the cold rate, from
          qmesh/tp2/quant_drafter_experts.py (required)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from safetensors import safe_open
from safetensors.torch import save_file

G = 2 ** 30
HERE = os.path.dirname(os.path.abspath(__file__))
EXPERT_RE = re.compile(r"(mtp\.\d+\.ffn)\.experts\.(\d+)\.")


def assign_rates(usage, tiers):
    """{(block, expert): bits} + per-block [(bits, count)]; the tier semantics of
    qmesh/tp2/drafter_experts.assign_rates, which this tree cannot import (it needs the TP2
    shard_tool). Tiers are hottest first; counts are an integer, nNN (experts covering NN% of
    the block's picks) or "rest"."""
    rate, summary = {}, {}
    for block, u in usage.items():
        order, cum = u["order"], u["cum"]
        pos, counts = 0, []
        for count, bits in tiers:
            if count == "rest":
                n = len(order) - pos
            elif count.startswith("n"):
                q = float(count[1:]) / 100
                n = max(next(i for i, x in enumerate(cum) if x >= q - 1e-12) + 1 - pos, 0)
            else:
                n = int(count)
            n = min(n, len(order) - pos)
            for e in order[pos:pos + n]:
                rate[(block, e)] = bits
            counts.append((bits, n))
            pos += n
        if pos != len(order):
            raise SystemExit(f"{block}: tiers cover {pos} of {len(order)} experts "
                             f"(end the spec with rest:<bits>)")
        summary[block] = counts
    return rate, summary


def shard_keys(path):
    with safe_open(path, framework = "pt", device = "cpu") as f:
        return list(f.keys())



def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default = os.environ.get("BASE", "/models/dsv41/exl3-serve-final"))
    ap.add_argument("--out", default = os.environ.get("OUT", ""))
    ap.add_argument("--usage", default = os.environ.get(
        "USAGE", os.path.join(HERE, "tp2/data/drafter_expert_usage.json")))
    ap.add_argument("--tiers", default = os.environ.get("TIERS", "n99:4,rest:2"))
    ap.add_argument("--set2", default = os.environ.get("SET2", ""),
                    help = "all routed-expert tensors at the cold rate (qmesh/tp2/quant_drafter_experts.py)")
    args = ap.parse_args()
    if not args.out:
        args.out = args.base.rstrip("/") + "-hc1"

    tiers = [(c, int(b)) for c, b in (t.split(":") for t in args.tiers.split(","))]
    rates = sorted({b for _, b in tiers})
    if len(rates) != 2:
        raise SystemExit("exactly two rates are supported: the hot rate comes from BASE")
    hot_rate, cold_rate = rates[-1], rates[0]
    print(f"tiers={tiers}", flush = True)

    if os.path.abspath(args.out) == os.path.abspath(args.base):
        raise SystemExit("--out must differ from --base")
    if not args.set2 or not os.path.exists(args.set2):
        raise SystemExit(f"missing cold source set: {args.set2!r}")
    os.makedirs(args.out, exist_ok = True)

    # The shipped base holds the hot-rate tensors. Locate the shards that carry drafter
    # routed-expert keys; every other file in the overlay is a symlink.
    shard_keys_cache: dict = {}
    expert_shards: list = []
    for name in sorted(os.listdir(args.base)):
        if not name.endswith(".safetensors"):
            continue
        ks = shard_keys(os.path.join(args.base, name))
        shard_keys_cache[name] = ks
        if any(EXPERT_RE.match(k) for k in ks):
            expert_shards.append(name)
    print(f"shards={len(shard_keys_cache)} with_drafter_experts={expert_shards}", flush = True)
    base_expert_keys = {k for name in expert_shards for k in shard_keys_cache[name] if EXPERT_RE.match(k)}
    print(f"base drafter expert tensors={len(base_expert_keys)}", flush = True)

    rate, summary = assign_rates(json.load(open(args.usage)), tiers)
    print("assign " + json.dumps(summary), flush = True)

    def bits_of(k):
        m = EXPERT_RE.match(k)
        return rate[(m.group(1), int(m.group(2)))]

    need_cold = {k for k in base_expert_keys if bits_of(k) == cold_rate}
    need_hot = base_expert_keys - need_cold
    print(f"hot({hot_rate}bit)={len(need_hot)} cold({cold_rate}bit)={len(need_cold)}", flush = True)

    new: dict = {}
    with safe_open(args.set2, framework = "pt", device = "cpu") as cold:
        if set(cold.keys()) != base_expert_keys:
            diff = set(cold.keys()) ^ base_expert_keys
            raise SystemExit(f"--set2 key set differs from base by {len(diff)} keys, e.g. {sorted(diff)[:3]}")
        for k in sorted(need_cold):
            new[k] = cold.get_tensor(k).contiguous()
    for name in expert_shards:
        want = [k for k in shard_keys_cache[name] if k in need_hot]
        if not want:
            continue
        with safe_open(os.path.join(args.base, name), framework = "pt", device = "cpu") as g:
            for k in want:
                new[k] = g.get_tensor(k).contiguous()
    if set(new.keys()) != base_expert_keys:
        raise SystemExit("assembled key set differs from the base drafter expert key set")
    bad = [(k, list(t.shape)) for k, t in new.items()
           if k.endswith(".trellis") and t.shape[-1] != 16 * bits_of(k)]
    if bad:
        raise SystemExit(f"rate mismatch: {len(bad)} trellis tensors, e.g. {bad[:3]}")
    hot_n = sum(1 for k in new if k.endswith(".w1.trellis") and bits_of(k) == hot_rate)
    cold_n = sum(1 for k in new if k.endswith(".w1.trellis") and bits_of(k) == cold_rate)
    print(f"assembled {len(new)} tensors hot_experts={hot_n} cold_experts={cold_n} "
          f"({sum(t.numel() * t.element_size() for t in new.values()) / G:.2f} GiB)", flush = True)

    # Write the overlay: untouched shards symlinked, drafter-expert shards rewritten without the
    # moved keys, and the per-expert tensors in one new file so every key exists exactly once
    for name in sorted(os.listdir(args.base)):
        src = os.path.join(args.base, name)
        dst = os.path.join(args.out, name)
        if os.path.lexists(dst):
            os.remove(dst)
        if name in shard_keys_cache and any(k in new for k in shard_keys_cache[name]):
            with safe_open(src, framework = "pt", device = "cpu") as g:
                keep = {k: g.get_tensor(k) for k in shard_keys_cache[name] if k not in new}
            save_file(keep, dst, metadata = {"format": "pt"})
            print(f"rewrote {name}: kept {len(keep)} of {len(shard_keys_cache[name])} tensors "
                  f"({os.path.getsize(dst) / G:.2f} GiB)", flush = True)
        else:
            os.symlink(os.path.realpath(src), dst)
    dst_new = os.path.join(args.out, "model-drafter-experts.safetensors")
    save_file(new, dst_new, metadata = {"format": "pt"})
    print(f"wrote {dst_new} ({os.path.getsize(dst_new) / G:.2f} GiB)", flush = True)

    per_block: dict = {}
    for key in new:
        per_block.setdefault(EXPERT_RE.match(key).group(1), set()).add(0)
    total = sum(t.numel() * t.element_size() for t in new.values())
    print("HC1_OVERLAY_OK " + json.dumps({
        "out": args.out, "tensors": len(new), "gib": round(total / G, 3),
        "blocks": sorted(per_block), "summary": summary,
    }), flush = True)


if __name__ == "__main__":
    main()
