#!/usr/bin/env python3
"""Build a TP2 overlay whose DSpark drafter routed experts (mtp.*.ffn.experts.*) carry per-expert EXL3 rates.

Experts are ranked by how often the drafter's router picks them (a usage file), then assigned to rate tiers,
hottest first. Each expert takes all of its tensors (w1/w2/w3 trellis, suh, svh, mul1) from ONE uniform-rate
source set, so gate/up/down of an expert share a rate as the mixed per-expert MoE kernels require.

Default drafter (v1.2): hot 4 / cold 2 = the experts covering 99% of picks at 4 bpw, the rest at 2 bpw. The usage
table shipped in qmesh/tp2/data/drafter_expert_usage.json was recorded on 153 training prompts (weight 0.8) plus 24
multilingual prompts (weight 0.2); expert sets come from qmesh/tp2/quant_drafter_experts.py:
  drafter_experts.py --usage qmesh/tp2/data/drafter_expert_usage.json \\
      --set 4=experts4.safetensors --set 2=experts2.safetensors \\
      --tiers n99:4,rest:2 --base OVERLAY --shards SHARDED_ROOT --out OUT

  --set BITS=FILE   full (unsharded) routed-expert set at one rate, e.g. from a uniform-rate quantization
  --usage FILE      {"mtp.N.ffn": {"order": [expert ids, most picked first], "cum": [cumulative pick share]}}
  --tiers SPEC      comma list "<count>:<bits>" hottest first; count = integer, nNN (experts covering NN% of
                    picks in that block) or "rest"
  --base OVERLAY    existing overlay root with rank0/ rank1/ (target shards, engram tables, config)
  --shards ROOT     the pre-sharded checkpoint root with rank<r>/shard_manifest.json (split rules)

Per rank: the expert tensors are split with qmesh/tp2/shard_tool.py's rules and written to
model-drafter-experts.safetensors; base shards that hold drafter-expert keys are rewritten without them (every
other tensor byte-identical) and everything else is symlinked, so every key exists exactly once.
"""
import argparse
import json
import os
import re
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_tool import TP, plan, slice_tensor  # noqa: E402

EXPERT_RE = re.compile(r"(mtp\.\d+\.ffn)\.experts\.(\d+)\.")


def assign_rates(usage, tiers):
    """{(block, expert): bits} and a per-block summary [(bits, count), ...]."""
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
            raise SystemExit(f"{block}: tiers cover {pos} of {len(order)} experts (end the spec with rest:<bits>)")
        summary[block] = counts
    return rate, summary


def main():
    ap = argparse.ArgumentParser(description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--usage", required = True)
    ap.add_argument("--set", action = "append", required = True, metavar = "BITS=FILE")
    ap.add_argument("--tiers", required = True)
    ap.add_argument("--base", required = True)
    ap.add_argument("--shards", required = True)
    ap.add_argument("--out", required = True)
    args = ap.parse_args()

    sets = {int(b): f for b, f in (s.split("=", 1) for s in args.set)}
    tiers = [(c, int(b)) for c, b in (t.split(":") for t in args.tiers.split(","))]
    missing = {b for _, b in tiers} - set(sets)
    if missing:
        raise SystemExit(f"no --set for rates {sorted(missing)}")
    rate, summary = assign_rates(json.load(open(args.usage)), tiers)
    print(json.dumps(summary), flush = True)

    files = {b: safe_open(sets[b], framework = "pt", device = "cpu") for b in {b for _, b in tiers}}
    keys = list(files[tiers[0][1]].keys())
    for b, f in files.items():
        if set(f.keys()) != set(keys):
            raise SystemExit(f"--set {b} holds a different key set than --set {tiers[0][1]}")

    per_rank = {r: {} for r in range(TP)}
    for key in keys:
        m = EXPERT_RE.match(key)
        if not m:
            raise SystemExit(f"not a drafter routed-expert tensor: {key}")
        t = files[rate[(m.group(1), int(m.group(2)))]].get_tensor(key)
        _, by_rank = plan(key, list(t.shape))
        for r, entry in by_rank.items():
            if entry is not None:
                dst, pcs = entry
                per_rank[r][dst] = slice_tensor(t, pcs, key)

    for r in range(TP):
        manifest = json.load(open(os.path.join(args.shards, f"rank{r}", "shard_manifest.json")))["tensors"]
        new = per_rank[r]
        for k, t in new.items():
            want = manifest[k]["dst_shape"]
            # trellis last dim is 16 x bpw: only the leading dims are rate-independent
            ok = list(t.shape[:-1]) == want[:-1] if k.endswith(".trellis") else list(t.shape) == want
            if not ok:
                raise SystemExit(f"rank{r} {k}: shape {list(t.shape)} vs manifest {want}")
        dst = os.path.join(args.out, f"rank{r}")
        os.makedirs(dst, exist_ok = True)
        save_file(new, os.path.join(dst, "model-drafter-experts.safetensors"))
        base = os.path.join(args.base, f"rank{r}")
        rewritten = 0
        for name in sorted(os.listdir(base)):
            p = os.path.join(dst, name)
            if os.path.lexists(p):
                continue
            s = os.path.join(base, name)
            if not name.endswith(".safetensors"):
                shutil.copyfile(s, p)
                continue
            real = os.path.realpath(s)
            with safe_open(real, framework = "pt", device = "cpu") as g:
                ks = list(g.keys())
                if not any(k in new for k in ks):
                    os.symlink(real, p)
                    continue
                keep = {k: g.get_tensor(k) for k in ks if k not in new}
            save_file(keep, p, metadata = {"format": "pt"})
            rewritten += 1
        print(f"rank{r}: {len(new)} expert tensors, {rewritten} base shards rewritten -> {dst}", flush = True)


if __name__ == "__main__":
    main()
