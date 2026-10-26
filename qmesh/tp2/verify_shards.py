#!/usr/bin/env python3
"""Verify the TP=2 pre-sharded checkpoint written by shard_tool.py (spec: qmesh/tp2/SHARD_SPEC.md).

Independent of shard_tool.plan(): roles, expected shapes and reconstruction are re-derived here from
SHARD_SPEC.md, so a planner bug cannot pass by agreeing with itself.

Checks (exit 0 only if all pass):
 1. Accounting: every source tensor is present on each rank where the spec says it belongs, absent
    elsewhere; no extra tensors in any rank; index == manifest == file headers.
 2. Shapes: file header shape == manifest dst_shape == spec-expected shape.
 3. Bytes: per-rank tensor bytes == index total_size; rank0/rank1 totals printed.
 4. Byte-exact reconstruction for a sample: >=64 routed experts (16 per K tier, all layers mixed, incl. MTP)
    with all of w1/w2/w3 x trellis/suh/svh/mul1, plus shared experts, wq_b, wo_b, wo_a, head, embed,
    markov heads, aligner, sinks, and vision wqkv/mlp.w1/wo/mlp.w2 (fused layouts un-interleaved).
"""
import argparse
import json
import os
import random
import re
import shutil
import sys
from collections import defaultdict

import torch
from safetensors import safe_open

TP = 2
EXL3_PARTS = ("trellis", "suh", "svh", "mul1")
DT = {"BF16": 2, "F16": 2, "I16": 2, "I32": 4, "F32": 4, "I64": 8, "U8": 1}
TORCH_DT = {"BF16": torch.bfloat16, "F16": torch.float16, "I16": torch.int16, "I32": torch.int32,
            "F32": torch.float32, "I64": torch.int64, "U8": torch.uint8}
failures = []


def fail(msg):
    failures.append(msg)
    print(f"FAIL: {msg}", flush=True)


def ok(msg):
    print(f"ok:   {msg}", flush=True)


def read_headers(d):
    """name -> (file, dtype, shape) for all tensors in a directory of safetensors files."""
    hdr = {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".safetensors"):
            continue
        with open(os.path.join(d, fn), "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                hdr[k] = (fn, v["dtype"], tuple(v["shape"]))
    return hdr


def nbytes(dtype, shape):
    n = 1
    for d in shape:
        n *= d
    return n * DT[dtype]


# ---- independent spec rules -------------------------------------------------------------------------
def role_of(name):
    if re.search(r"\.ffn\.(shared_experts|experts\.\d+)\.w[13]\.", name):
        return "out"
    if re.search(r"\.ffn\.(shared_experts|experts\.\d+)\.w2\.", name):
        return "in"
    if re.search(r"(^|\.)embed\.weight$|^head\.|markov_w[12]\.weight$|^aligner\.w1\.", name):
        return "out"
    if re.search(r"\.attn\.wq_b\.", name):
        return "out"
    if re.search(r"\.attn\.wo_b\.|^aligner\.w2\.|vision\.blocks\.\d+\.attn\.wo\.|vision\.blocks\.\d+\.mlp\.w2\.",
                 name):
        return "in"
    if re.search(r"vision\.blocks\.\d+\.(attn\.wqkv|mlp\.w1)\.", name):
        return "vfused"
    if name.endswith(".attn_sink"):
        return "sink"
    if re.search(r"\.attn\.wo_a\.slice\.\d+\.", name):
        return "woa"
    return "rep"


def expected(name, shape, r):
    """Spec-expected rank-r tensor shape, or None if the tensor is absent on rank r."""
    role = role_of(name)
    leaf = name.rsplit(".", 1)[-1]
    shp = tuple(shape)
    if role == "woa":
        k = int(re.search(r"\.slice\.(\d+)\.", name).group(1))
        return shp if k // 4 == r else None
    if role == "sink":
        return (shp[0] // TP,)
    if role == "vfused":
        return (shp[0] // 2,) + shp[1:]
    if role == "rep" or leaf == "mul1":
        return shp
    if role == "out":
        if leaf == "trellis":
            return (shp[0], shp[1] // TP, shp[2])
        if leaf == "suh":
            return shp
        if leaf in ("svh", "bias", "weight"):
            return (shp[0] // TP,) + shp[1:]
    if role == "in":
        if leaf == "bias":
            return shp if r == 0 else None
        if leaf == "trellis":
            return (shp[0] // TP,) + shp[1:]
        if leaf == "suh":
            return (shp[0] // TP,)
        if leaf == "svh":
            return shp
        if leaf == "weight":
            return (shp[0], shp[1] // TP)
    raise AssertionError(f"no spec rule for {name} (role={role}, leaf={leaf})")


def reconstruct(name, shape, t0, t1):
    """Rebuild the full source tensor from rank0/rank1 pieces (inverse of the spec split)."""
    role = role_of(name)
    leaf = name.rsplit(".", 1)[-1]
    if role == "sink":
        return torch.cat([t0, t1], 0)
    if role == "vfused":
        parts = 3 if "wqkv" in name else 2
        h = shape[0] // parts // TP  # 512 (qkv) or 1408 (gate|up) rows per part per rank
        return torch.cat([torch.cat([t0[p * h:(p + 1) * h], t1[p * h:(p + 1) * h]], 0)
                          for p in range(parts)], 0)
    if role == "out":
        if leaf == "trellis":
            return torch.cat([t0, t1], 1)
        if leaf == "suh":
            return t0
        return torch.cat([t0, t1], 0)
    if role == "in":
        if leaf == "bias":
            return t0
        if leaf == "svh":
            return t0
        if leaf == "weight":
            return torch.cat([t0, t1], 1)
        return torch.cat([t0, t1], 0)  # trellis, suh
    return t0  # rep / scalar


def bytes_eq(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.reshape(-1).view(torch.uint8),
                                                                    b.reshape(-1).view(torch.uint8))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-tier", type=int, default=16, help="routed expert bases sampled per K tier")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    src_hdr = read_headers(args.src)
    src_idx = json.load(open(os.path.join(args.src, "model.safetensors.index.json")))["weight_map"]
    print(f"source: {len(src_hdr)} tensors")
    if set(src_hdr) != set(src_idx):
        fail("source index keys != source header keys")

    rank_dir = {r: os.path.join(args.out, f"rank{r}") for r in range(TP)}
    rank_hdr, rank_idx, manifests = {}, {}, {}
    for r in range(TP):
        rank_idx[r] = json.load(open(os.path.join(rank_dir[r], "model.safetensors.index.json")))
        rank_hdr[r] = read_headers(rank_dir[r])
        manifests[r] = json.load(open(os.path.join(rank_dir[r], "shard_manifest.json")))["tensors"]

    # ---- 1. accounting -------------------------------------------------------------------------------
    print("== 1. accounting")
    src_to_dst = {r: defaultdict(list) for r in range(TP)}
    for r in range(TP):
        if set(rank_idx[r]["weight_map"]) != set(rank_hdr[r]):
            fail(f"rank{r}: index keys != file header keys")
        if set(manifests[r]) != set(rank_hdr[r]):
            fail(f"rank{r}: manifest keys != file header keys (extra/missing tensors)")
        for dst, m in manifests[r].items():
            src_to_dst[r][m["src_name"]].append(dst)
    n_bad = 0
    for name, (_, _, shape) in src_hdr.items():
        for r in range(TP):
            exp = expected(name, shape, r)
            got = src_to_dst[r].get(name, [])
            if exp is None and got:
                n_bad += 1
                fail(f"{name}: must be absent on rank{r}, found {got}")
            elif exp is not None and len(got) != 1:
                n_bad += 1
                fail(f"{name}: expected 1 tensor on rank{r}, found {len(got)}")
    extra = {n for r in range(TP) for n in src_to_dst[r] if n not in src_hdr}
    if extra:
        fail(f"manifest references unknown source names: {sorted(extra)[:5]}")
    if n_bad == 0:
        ok(f"every source tensor ({len(src_hdr)}) placed per spec on both ranks; no extras")

    # ---- 2. shapes -----------------------------------------------------------------------------------
    print("== 2. shapes")
    n_bad = 0
    for r in range(TP):
        for dst, m in manifests[r].items():
            fshape = rank_hdr[r][dst][2]
            name = m["src_name"]
            exp = expected(name, src_hdr[name][2], r)
            if tuple(m["dst_shape"]) != fshape or exp != fshape:
                n_bad += 1
                if n_bad <= 5:
                    fail(f"rank{r} {dst}: file {fshape}, manifest {m['dst_shape']}, spec {exp}")
            if rank_hdr[r][dst][1] != src_hdr[name][1]:
                n_bad += 1
                fail(f"rank{r} {dst}: dtype {rank_hdr[r][dst][1]} != source {src_hdr[name][1]}")
    if n_bad == 0:
        ok("all rank tensor shapes/dtypes match spec and manifest")

    # ---- 3. bytes ------------------------------------------------------------------------------------
    print("== 3. bytes")
    rb = {}
    for r in range(TP):
        rb[r] = sum(nbytes(d, s) for _, d, s in rank_hdr[r].values())
        if rb[r] != rank_idx[r]["metadata"]["total_size"]:
            fail(f"rank{r}: sum of tensor bytes {rb[r]} != index total_size")
        print(f"  rank{r}: {rb[r]} B = {rb[r] / 2**30:.3f} GiB, {len(rank_hdr[r])} tensors")
    print(f"  |rank0 - rank1| = {abs(rb[0] - rb[1])} B")
    src_total = sum(nbytes(d, s) for _, d, s in src_hdr.values())
    print(f"  source total {src_total / 2**30:.3f} GiB; replicated-bytes overhead = "
          f"{(rb[0] + rb[1] - src_total) / 2**30:.3f} GiB")

    # ---- 4. byte-exact reconstruction ----------------------------------------------------------------
    print("== 4. byte-exact reconstruction")
    tiers = defaultdict(list)
    for name, (_, _, shape) in src_hdr.items():
        m = re.fullmatch(r"((?:layers|mtp)\.\d+\.ffn\.experts\.\d+)\.w1\.trellis", name)
        if m:
            tiers[shape[2]].append(m.group(1))
    print("  routed experts by trellis last dim (16K; K=last/16): " +
          ", ".join(f"{k} (K={k / 16:g}): {len(v)}" for k, v in sorted(tiers.items())))
    picked = []
    for k in sorted(tiers):
        cand = sorted(set(tiers[k]))
        rng.shuffle(cand)
        picked += cand[: args.per_tier]
    fixed = [
        "layers.0.ffn.shared_experts", "layers.39.ffn.shared_experts", "mtp.0.ffn.shared_experts",
        "layers.0.attn.wq_b", "layers.39.attn.wq_b", "layers.0.attn.wo_b", "layers.39.attn.wo_b",
        "head", "embed", "mtp.0.markov_head.markov_w1", "mtp.0.markov_head.markov_w2",
        "aligner.w1", "aligner.w2", "layers.0.attn.attn_sink", "mtp.0.attn.attn_sink",
        "layers.0.attn.wo_a.slice.0", "layers.0.attn.wo_a.slice.7",
    ]
    vision = ["vision.blocks.0.attn.wqkv", "vision.blocks.31.attn.wqkv", "vision.blocks.0.mlp.w1",
              "vision.blocks.31.mlp.w1", "vision.blocks.0.attn.wo", "vision.blocks.0.mlp.w2"]
    bases = picked + fixed + vision

    def names_for(base):
        if base in tiers or re.fullmatch(r"(?:layers|mtp)\.\d+\.ffn\.experts\.\d+", base):
            return [f"{base}.{w}.{p}" for w in ("w1", "w2", "w3") for p in EXL3_PARTS]
        if base.endswith(".ffn.shared_experts"):
            return [f"{base}.{w}.{p}" for w in ("w1", "w2", "w3") for p in EXL3_PARTS]
        if base == "head":
            return [f"head.{p}" for p in EXL3_PARTS]
        if base.startswith("vision.blocks"):
            return [base + ".weight", base + ".bias"] if base.endswith("wqkv") or base.endswith(".wo") else \
                [base + ".weight"]
        if base.startswith("aligner"):
            return [base + ".weight", base + ".bias"]
        if base in ("embed", "mtp.0.markov_head.markov_w1", "mtp.0.markov_head.markov_w2"):
            return [base + ".weight"] if base != "embed" else ["embed.weight"]
        if base.endswith(".attn_sink"):
            return [base]
        if re.search(r"\.wo_a\.slice\.\d+$", base):
            return [f"{base}.{p}" for p in EXL3_PARTS]
        return [f"{base}.{p}" for p in EXL3_PARTS]  # wq_b, wo_b

    checked, to_check = 0, []
    for base in bases:
        for n in names_for(base):
            if n in src_hdr:
                to_check.append(n)
    print(f"  sampled {len(picked)} routed expert bases (+{len(fixed) + len(vision)} other bases); "
          f"{len(to_check)} tensors to compare")

    handles = {}

    def get(path, name):
        if path not in handles:
            handles[path] = safe_open(path, framework="pt", device="cpu")
        return handles[path].get_tensor(name)

    for name in to_check:
        sfile, dtype, shape = src_hdr[name]
        src_t = get(os.path.join(args.src, sfile), name)
        role = role_of(name)
        if role == "woa":
            k = int(re.search(r"\.slice\.(\d+)\.", name).group(1))
            dst = src_to_dst[k // 4][name][0]
            dfile = rank_hdr[k // 4][dst][0]
            if bytes_eq(get(os.path.join(rank_dir[k // 4], dfile), dst), src_t):
                checked += 1
            else:
                fail(f"{name}: wo_a slice not byte-exact on rank{k // 4}")
            continue
        pieces = {}
        for r in range(TP):
            names_r = src_to_dst[r].get(name, [])
            if names_r:
                dst = names_r[0]
                pieces[r] = get(os.path.join(rank_dir[r], rank_hdr[r][dst][0]), dst)
        if role == "rep" or name.endswith(".mul1"):
            # replicated tensors and EXL3 mul1 (copied whole to both ranks) must equal the source on each rank
            if all(bytes_eq(pieces[r], src_t) for r in pieces):
                checked += 1
            else:
                fail(f"{name}: replicated/mul1 copy not byte-exact")
            continue
        if role == "in" and name.endswith(".bias"):
            recon = pieces[0]  # IN bias lives on rank0 only
        else:
            recon = reconstruct(name, shape, pieces[0], pieces[1])
        if bytes_eq(recon, src_t):
            checked += 1
        else:
            fail(f"{name}: reconstruction NOT byte-exact (recon {tuple(recon.shape)} vs src {tuple(src_t.shape)})")
    print(f"  byte-exact tensors checked: {checked}")
    if checked < 64:
        fail(f"only {checked} tensors checked (<64)")
    if len([b for b in picked]) < 64:
        fail(f"only {len(picked)} routed experts sampled (<64)")
    ok(f"reconstruction: {checked} tensors compared, {len(picked)} routed experts across {len(tiers)} K tiers")

    # ---- summary -------------------------------------------------------------------------------------
    print("== summary")
    print(f"  rank0 {rb[0] / 2**30:.3f} GiB, rank1 {rb[1] / 2**30:.3f} GiB")
    du = shutil.disk_usage(args.out)
    print(f"  disk {args.out}: {du.free / 2**30:.2f} GiB free of {du.total / 2**30:.2f} GiB")
    if failures:
        print(f"VERIFY FAILED: {len(failures)} check(s)")
        return 1
    print("VERIFY PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
