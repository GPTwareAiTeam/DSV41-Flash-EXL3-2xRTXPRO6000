#!/usr/bin/env python3
"""Build a loader-ready TP2 overlay rank dir that swaps ONLY the DSpark drafter (mtp.*) weights.

Source: a diffbot-style drafter pack (`drafter-code-k7`): routed experts EXL3/MCG K=3
(`trellis`/`suh`/`svh`/`mcg`) + *all* non-expert linears DeepSeek-native FP8
(`weight` F8_E4M3 + `scale` F8_E8M0, 32x32 blocks), fused `attn.wo_a`, native
`markov_head.embed/head` names. Target: our exllamav3 TP2 rank dir (all-EXL3/mul1), consumed by
qmesh/tp2/run_tp2.py for `component="mtp"`.

Split rules are the ones in qmesh/tp2/shard_tool.py (see qmesh/tp2/SHARD_SPEC.md):
  OUT-split (dim 0 of an [out,in] weight): w1/w3 (routed + shared), wq_b, markov head rows
  IN-split  (dim 1): w2 (routed + shared), wo_b
  REPLICATED: wq_a, wkv, q_norm, kv_norm, norms, gate, hc_*, main_proj, confidence_head
  EXL3: OUT -> trellis dim1 + svh dim0; IN -> trellis dim0 + suh dim0; mcg copied whole
  FP8 : the `.scale` rides the same axis as `.weight`, block factor = weight/scale dim ratio (32)
  fused `attn.wo_a` [8192,4096] (+ scale [256,128]) -> rank r rows r*4096..(r+1)*4096, cut into
        4 groups of 1024 rows -> `attn.wo_a.slice.{0..3}.{weight,scale}` (scale 4x32 rows)
  renames: markov_head.embed.weight -> markov_head.markov_w1.weight,
           markov_head.head.weight  -> markov_head.markov_w2.weight
  dropped: ffn.gate.bias_vl (inert in our runtime; absent from the base rank dirs)

Output dir layout: rank<r>/ is a symlink farm over --base/rank<r> (which must already carry the
engram tables), except the shards that hold mtp.* keys (rewritten with the new drafter) and
config.json (copy of the base config with `dspark_block_size` = --block-size). No index/manifest is
written: exllamav3/loader/safetensors.py globs *.safetensors in the rank dir.

Usage:
  drafter_overlay.py --drafter DIR --base OVERLAY_ROOT --index-root SHARDED_ROOT --out OUT \
      [--block-size N] [--verify] [--dry-run]

`--verify` re-reads the written shards: byte-exact rank0+rank1 reconstruction of every source
tensor. Shape/logical-dim checks run always (against the base rank dir headers).
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TP = 2
OUT, IN, REP = "out", "in", "rep"
MCG_MULT = 3417055213  # 0xCBAC1FED, exllamav3 codebook_mcg_mult (validated at load)
BLOCK = 32             # FP8 e8m0 scale block size
TRELLIS_K = 48         # source trellis K slice width (K=3)

# full-tensor-name regex, role, kind
MOD_RULES = [
    (r"mtp\.\d+\.attn\.wq_b\.(weight|scale)", OUT, "fp8"),
    (r"mtp\.\d+\.attn\.wo_b\.(weight|scale)", IN, "fp8"),
    (r"mtp\.\d+\.attn\.(wq_a|wkv)\.(weight|scale)", REP, "fp8"),
    (r"mtp\.\d+\.attn\.wo_a\.(weight|scale)", "woa", "fp8"),
    (r"mtp\.\d+\.attn\.(q_norm|kv_norm)\.weight", REP, "plain"),
    (r"mtp\.\d+\.attn\.attn_sink", "sink", "plain"),
    (r"mtp\.\d+\.(attn_norm|ffn_norm|norm|main_norm)\.weight", REP, "plain"),
    (r"mtp\.\d+\.ffn\.gate\.(weight|bias)", REP, "plain"),
    (r"mtp\.\d+\.ffn\.experts\.\d+\.w[13]\.(trellis|suh|svh|mcg)", OUT, "exl3"),
    (r"mtp\.\d+\.ffn\.experts\.\d+\.w2\.(trellis|suh|svh|mcg)", IN, "exl3"),
    (r"mtp\.\d+\.ffn\.shared_experts\.w[13]\.(weight|scale)", OUT, "fp8"),
    (r"mtp\.\d+\.ffn\.shared_experts\.w2\.(weight|scale)", IN, "fp8"),
    (r"mtp\.0\.main_proj\.(weight|scale)", REP, "fp8"),
    (r"mtp\.\d+\.markov_head\.(markov_w1|markov_w2)\.weight", OUT, "markov"),
    (r"mtp\.\d+\.confidence_head\.proj\.weight", REP, "plain"),
    (r"mtp\.\d+\.(hc_attn|hc_ffn)_(base|fn|scale)", REP, "plain"),
]

RENAME = {
    "mtp.2.markov_head.embed.weight": "mtp.2.markov_head.markov_w1.weight",
    "mtp.2.markov_head.head.weight": "mtp.2.markov_head.markov_w2.weight",
}
DROP = {"mtp.0.ffn.gate.bias_vl", "mtp.1.ffn.gate.bias_vl", "mtp.2.ffn.gate.bias_vl"}

FP8_SUFFIXES = ("weight", "scale")
EXL3_SUFFIXES = ("trellis", "suh", "svh", "mcg")


class OverlayError(RuntimeError):
    pass


def die(msg):
    raise OverlayError(msg)


def read_header(path):
    with open(path, "rb") as fh:
        n = int.from_bytes(fh.read(8), "little")
        hdr = json.loads(fh.read(n))
    md = hdr.pop("__metadata__", None)
    return hdr, md


def scan_dir(d):
    """name -> (file, dtype, shape) for every tensor in a safetensors directory."""
    out = {}
    for fn in sorted(f for f in os.listdir(d) if f.endswith(".safetensors")):
        hdr, _ = read_header(os.path.join(d, fn))
        for k, v in hdr.items():
            if k in out:
                die(f"duplicate tensor {k} in {d}")
            out[k] = (fn, v["dtype"], tuple(v["shape"]))
    return out


def half(shape, dim, r):
    n = shape[dim]
    if n % TP:
        die(f"dim {dim} of {shape} not divisible by {TP}")
    k = n // TP
    return (dim, r * k, (r + 1) * k)


def scale_span(wshape, sshape, axis, start, end):
    """Weight-axis range -> e8m0 scale-grid range (block factor must be 32)."""
    if wshape[axis] % sshape[axis]:
        die(f"scale grid {sshape} does not tile weight dim {axis} of {wshape}")
    f = wshape[axis] // sshape[axis]
    if f != BLOCK:
        die(f"unexpected fp8 block factor {f} ({wshape} vs {sshape})")
    if start % f or end % f:
        die(f"slice {start}:{end} not on a {f}-block boundary")
    return (axis, start // f, end // f)


def dst_shape(src_shape, pieces):
    if not pieces:
        return tuple(src_shape)
    dim = pieces[0][0]
    s = list(src_shape)
    s[dim] = sum(e - b for _, b, e in pieces)
    return tuple(s)


def plan_tensor(name, wshape_map):
    """name -> (src dtype, src shape, kind, rule, {rank: [(dst_name, pieces)]})."""
    src = wshape_map[name]
    dtype, shape = src[1], src[2]
    if name in DROP:
        return dtype, shape, "drop", "drop", {}
    dst_base = RENAME.get(name, name)
    hits = [(role, kind) for rx, role, kind in MOD_RULES if re.fullmatch(rx, dst_base)]
    if len(hits) != 1:
        die(f"tensor '{dst_base}' matches {len(hits)} rules (need exactly 1)")
    role, kind = hits[0]
    prefix, _, suffix = dst_base.rpartition(".")
    if kind == "exl3" and suffix not in EXL3_SUFFIXES:
        die(f"exl3 tensor with unexpected leaf '{suffix}': {name}")
    if kind == "fp8" and suffix not in FP8_SUFFIXES:
        die(f"fp8 tensor with unexpected leaf '{suffix}': {name}")

    if role == "woa":
        if suffix == "weight" and (shape != (8192, 4096) or dtype != "F8_E4M3"):
            die(f"unexpected fused wo_a weight {dtype} {shape}")
        per = {}
        for r in range(TP):
            span = 8192 // TP          # 4096 weight rows per rank
            grp = span // 4            # 1024 rows per o-group
            if suffix == "scale":
                grp = grp // BLOCK     # 32 scale rows per o-group
            keys = []
            for g in range(4):
                s = r * span + g * grp if suffix == "weight" else (r * span // BLOCK) + g * grp
                keys.append((f"{prefix}.slice.{g}.{suffix}", [(0, s, s + grp)]))
            per[r] = keys
        return dtype, shape, kind, "woa-fused-split", per

    if kind == "exl3":
        if suffix == "mcg":
            return dtype, shape, kind, "mcg-copy", {r: [(dst_base, [])] for r in range(TP)}
        if suffix == "trellis" and len(shape) != 3:
            die(f"trellis not 3-D: {name} {shape}")
        if suffix in ("suh", "svh") and len(shape) != 1:
            die(f"{suffix} not 1-D: {name} {shape}")
        dim = ({"trellis": 1, "svh": 0} if role == OUT else {"trellis": 0, "suh": 0}).get(suffix)
        return dtype, shape, kind, f"exl3-{role}", {
            r: [(dst_base, [] if dim is None else [half(shape, dim, r)])] for r in range(TP)}

    if role == REP:
        return dtype, shape, kind, "rep", {r: [(dst_base, [])] for r in range(TP)}
    if role == "sink":
        return dtype, shape, kind, "sink-split", {r: [(dst_base, [half(shape, 0, r)])] for r in range(TP)}

    axis = 0 if role == OUT else 1
    if axis == 1 and len(shape) < 2:
        die(f"IN-split of a 1-D tensor: {name} {shape}")
    per = {}
    if kind == "fp8" and suffix == "scale":
        wname = name.rsplit(".", 1)[0] + ".weight"
        if wname not in wshape_map:
            die(f"scale without weight: {name}")
        wshape = wshape_map[wname][2]
        for r in range(TP):
            _, b, e = half(wshape, axis, r)
            per[r] = [(dst_base, [scale_span(wshape, shape, axis, b, e)])]
    else:
        for r in range(TP):
            per[r] = [(dst_base, [half(shape, axis, r)])]
    return dtype, shape, kind, f"{kind}-{role}", per


def build_plan(drafter_dir):
    src = scan_dir(drafter_dir)
    entries = []
    for name in sorted(src):
        dtype, shape, kind, rule, per = plan_tensor(name, src)
        entries.append({"src": name, "file": src[name][0], "dtype": dtype, "shape": shape,
                        "kind": kind, "rule": rule, "dst": per})
    return entries, src


def module_of(dst_name):
    return dst_name.rsplit(".", 1)[0]


def dst_bin_map(index_root, r):
    """Return (name -> base shard, module -> {part -> base shard}, module -> base part set)."""
    path = os.path.join(index_root, f"rank{r}", "model.safetensors.index.json")
    wmap = json.load(open(path))["weight_map"]
    where, parts, mods = {}, {}, {}
    for k, bin_name in wmap.items():
        if not k.startswith("mtp."):
            continue
        where[k] = bin_name
        parts.setdefault(module_of(k), {})[k.rsplit(".", 1)[1]] = bin_name
        mods.setdefault(module_of(k), set()).add(k.rsplit(".", 1)[1])
    return where, parts, mods


def bin_for(dst_name, where, parts):
    """Shard to write a dst tensor into: its own base key, else its module's trellis key."""
    if dst_name in where:
        return where[dst_name]
    m = module_of(dst_name)
    for part in ("trellis", "weight", "suh", "bias"):
        if part in parts.get(m, {}):
            return parts[m][part]
    die(f"{dst_name}: no base shard for module {m}")


def base_headers(base_rank_dir):
    d = {}
    for fn in sorted(f for f in os.listdir(base_rank_dir) if f.endswith(".safetensors")):
        hdr, _ = read_header(os.path.join(base_rank_dir, fn))
        for k, v in hdr.items():
            if k in d:
                die(f"duplicate tensor {k} in {base_rank_dir}")
            d[k] = (v["dtype"], tuple(v["shape"]))
    return d


def validate(entries, r, mods, base):
    """Cross-check produced modules/shapes against the base rank dir. Returns dst -> (entry, pieces)."""
    produced = {}
    for e in entries:
        for dst_name, pcs in e["dst"].get(r, []):
            if dst_name in produced:
                die(f"duplicate dst tensor on rank {r}: {dst_name}")
            produced[dst_name] = (e, pcs)

    prod_mods = {module_of(k) for k in produced}
    if prod_mods != set(mods):
        missing = sorted(set(mods) - prod_mods)
        extra = sorted(prod_mods - set(mods))
        die(f"rank {r} module mismatch: missing={missing[:6]} extra={extra[:6]}")

    for m in sorted(mods):
        base_parts = mods[m]
        mine = {k: v for k, v in produced.items() if module_of(k) == m}
        my_parts = {k.rsplit(".", 1)[1] for k in mine}
        if "trellis" in base_parts:
            if my_parts == set(FP8_SUFFIXES):
                kind = "fp8"
            elif "trellis" in my_parts and my_parts <= set(EXL3_SUFFIXES):
                kind = "exl3"
            else:
                die(f"{m}: parts {sorted(my_parts)} incompatible with base {sorted(base_parts)}")
        else:
            if not my_parts <= base_parts:
                die(f"{m}: parts {sorted(my_parts)} not in base {sorted(base_parts)}")
            kind = "plain"
        in_dim = out_dim = None
        if kind != "plain":
            if "trellis" in base_parts:
                in_dim, out_dim = base[m + ".suh"][1][0], base[m + ".svh"][1][0]
            elif m + ".weight" in base:
                w = base[m + ".weight"][1]
                out_dim, in_dim = w[0], w[1]
            else:
                die(f"{m}: cannot derive dims from base")
        for k, (e, pcs) in mine.items():
            part = k.rsplit(".", 1)[1]
            got = dst_shape(e["shape"], pcs)
            if kind == "exl3":
                exp = {"trellis": (in_dim // 16, out_dim // 16, TRELLIS_K),
                       "suh": (in_dim,), "svh": (out_dim,), "mcg": ()}[part]
            elif kind == "fp8":
                exp = {"weight": (out_dim, in_dim),
                       "scale": (out_dim // BLOCK, in_dim // BLOCK)}[part]
            else:
                if k not in base:
                    die(f"{k}: not present in base rank dir")
                exp = base[k][1]
            if got != exp:
                die(f"{k}: dst shape {got} != expected {exp}")
    return produced


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drafter", required=True)
    ap.add_argument("--base", required=True, help="overlay root with rank0/rank1 (engram tables included)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--index-root", required=True,
                    help="root with rank<r>/model.safetensors.index.json describing the base layout")
    ap.add_argument("--block-size", type=int, default=None, help="dspark_block_size written to config.json")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    base = {r: base_headers(os.path.join(args.base, f"rank{r}")) for r in range(TP)}
    maps = [dst_bin_map(args.index_root, r) for r in range(TP)]
    entries, src = build_plan(args.drafter)
    print(f"source: {len(src)} tensors, {sum(1 for e in entries if e['rule'] != 'drop')} mapped, "
          f"{sum(1 for e in entries if e['rule'] == 'drop')} dropped")

    produced = {r: validate(entries, r, maps[r][2], base[r]) for r in range(TP)}
    print(f"dst per rank: rank0={len(produced[0])} rank1={len(produced[1])}")

    mcg = [k for k in produced[0] if k.endswith(".mcg")]
    if mcg:
        e = produced[0][mcg[0]][0]
        with safe_open(os.path.join(args.drafter, e["file"]), framework="pt", device="cpu") as f:
            v = int(f.get_tensor(e["src"]).item()) & 0xFFFFFFFF
        if v != MCG_MULT:
            die(f"mcg marker 0x{v:08x} != 0xCBAC1FED")
        print(f"mcg marker ok (0x{v:08x}), {len(mcg)} expert markers on rank0")

    # dst -> shard, grouped for writing
    plan = {r: {} for r in range(TP)}
    for r in range(TP):
        where, parts, _ = maps[r]
        for dst_name, (e, pcs) in produced[r].items():
            plan[r].setdefault(bin_for(dst_name, where, parts), []).append((dst_name, e, pcs))
    for r in range(TP):
        for b in sorted(plan[r]):
            print(f"  rank{r} {b}: {len(plan[r][b])} new mtp tensors")

    if args.dry_run:
        print(f"dry run ok ({time.time() - t0:.1f}s)")
        return

    if os.path.isdir(args.out) and os.listdir(args.out):
        die(f"{args.out} exists and is not empty; remove it first")
    os.makedirs(args.out, exist_ok=True)

    for r in range(TP):
        base_rank = os.path.join(args.base, f"rank{r}")
        out_rank = os.path.join(args.out, f"rank{r}")
        os.makedirs(out_rank, exist_ok=True)
        rewritten = sorted(plan[r])
        for f in sorted(os.listdir(base_rank)):
            if f in rewritten or f == "config.json":
                continue
            os.symlink(os.path.realpath(os.path.join(base_rank, f)), os.path.join(out_rank, f))
        print(f"rank{r}: {len(os.listdir(out_rank))} symlinks, rewriting {len(rewritten)} shards")

        for b in rewritten:
            src_path = os.path.join(base_rank, b)
            hdr, md = read_header(src_path)
            keep = [k for k in hdr if not k.startswith("mtp.")]
            tensors = {}
            with safe_open(src_path, framework="pt", device="cpu") as fh:
                for k in keep:
                    tensors[k] = fh.get_tensor(k)
            handles = {}
            try:
                for dst_name, e, pcs in plan[r][b]:
                    p = os.path.join(args.drafter, e["file"])
                    if p not in handles:
                        handles[p] = safe_open(p, framework="pt", device="cpu")
                    t = handles[p].get_tensor(e["src"])
                    if pcs:
                        d, s, en = pcs[0]
                        t = t.narrow(d, s, en - s).contiguous()
                    tensors[dst_name] = t
            finally:
                for fh in handles.values():
                    fh.__exit__(None, None, None)
            tmp = os.path.join(out_rank, b + ".tmp")
            save_file(tensors, tmp, metadata={k: str(v) for k, v in (md or {}).items()} or None)
            n_new = len(plan[r][b])
            del tensors
            os.replace(tmp, os.path.join(out_rank, b))
            print(f"  rank{r} {b}: {len(keep)} base + {n_new} drafter tensors "
                  f"({os.path.getsize(os.path.join(out_rank, b)) / 2**30:.2f} GiB)")

        cfg = json.load(open(os.path.join(base_rank, "config.json")))
        if args.block_size is not None:
            cfg.get("text_config", cfg)["dspark_block_size"] = args.block_size
        with open(os.path.join(out_rank, "config.json"), "w") as fh:
            json.dump(cfg, fh, indent=2)
        print(f"rank{r}: dspark_block_size={cfg.get('text_config', cfg).get('dspark_block_size')}")

    if args.verify:
        verify(entries, args, plan)

    print(f"done in {time.time() - t0:.1f}s -> {args.out}")


def verify(entries, args, plan):
    """Byte-exact rank0+rank1 reconstruction of every source tensor from the written shards."""
    t0 = time.time()
    handles = {}
    for r in range(TP):
        for b in plan[r]:
            handles[(r, b)] = safe_open(os.path.join(args.out, f"rank{r}", b), framework="pt", device="cpu")
    loc = {(r, dst): b for r in range(TP) for b, items in plan[r].items() for dst, _, _ in items}
    n = 0
    try:
        for e in entries:
            if e["rule"] == "drop":
                continue
            with safe_open(os.path.join(args.drafter, e["file"]), framework="pt", device="cpu") as f:
                orig = f.get_tensor(e["src"])
            if e["rule"] == "woa-fused-split":
                for r in range(TP):
                    parts = [handles[(r, loc[(r, d)])].get_tensor(d) for d, _ in e["dst"][r]]
                    got = torch.cat(parts, dim=0)
                    exp = orig.narrow(0, r * orig.shape[0] // TP, orig.shape[0] // TP)
                    if not torch.equal(got, exp):
                        die(f"wo_a reconstruction mismatch {e['src']} rank{r}")
                    n += 1
                continue
            got_parts = []
            for r in range(TP):
                dst, pcs = e["dst"][r][0]
                t = handles[(r, loc[(r, dst)])].get_tensor(dst)
                got_parts.append((pcs[0][0] if pcs else None, t))
            if got_parts[0][0] is None:
                for _, t in got_parts:
                    if not torch.equal(t, orig):
                        die(f"replicated tensor mismatch {e['src']}")
            else:
                got = torch.cat([t for _, t in got_parts], dim=got_parts[0][0])
                if not torch.equal(got, orig):
                    die(f"split reconstruction mismatch {e['src']}")
            n += 1
    finally:
        for fh in handles.values():
            fh.__exit__(None, None, None)
    print(f"verify: {n} source tensors reconstructed byte-exactly ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    try:
        main()
    except OverlayError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
