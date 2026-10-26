#!/usr/bin/env python3
"""Offline TP=2 pre-sharder for DeepSeek-V4.1-Flash EXL3 mixed-2.565 (tp2 checkpoint).

Spec: qmesh/tp2/SHARD_SPEC.md. Reads source safetensors one shard at a time (safe_open, framework=pt)
and writes rank0/ and rank1/ (safetensors + index + config + manifest). CPU only, no GPU.

Usage:
  shard_tool.py --src SRC --out OUT            # real run
  shard_tool.py --src SRC --out OUT --dry-run  # plan only, header scan, no writes

Any tensor not matched by an explicit rule raises ShardError (no silent default).
"""
import argparse
import json
import os
import re
import shutil
import sys
import time

TP = 2
SPEC_PATH = "qmesh/tp2/SHARD_SPEC.md"
EXL3_PARTS = ("trellis", "suh", "svh", "mul1")
OUT, IN, REP = "out", "in", "rep"

DTYPE_BYTES = {"BF16": 2, "F16": 2, "I16": 2, "U16": 2, "F32": 4, "I32": 4, "U32": 4,
               "F64": 8, "I64": 8, "U64": 8, "I8": 1, "U8": 1, "BOOL": 1,
               "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}
DTYPE_OF_TORCH = {"torch.bfloat16": "BF16", "torch.float16": "F16", "torch.int16": "I16", "torch.float32": "F32",
                   "torch.int32": "I32", "torch.uint8": "U8", "torch.int64": "I64", "torch.bool": "BOOL",
                   "torch.float8_e4m3fn": "F8_E4M3", "torch.float8_e8m0fnu": "F8_E8M0"}


class ShardError(RuntimeError):
    pass


# Module prefix (tensor name minus its last component) -> role. fullmatch; exactly one must match.
MODULE_ROLES = [
    # routed + shared experts, trunk and MTP
    (r"(layers|mtp)\.\d+\.ffn\.(shared_experts|experts\.\d+)\.w[13]", OUT),
    (r"(layers|mtp)\.\d+\.ffn\.(shared_experts|experts\.\d+)\.w2", IN),
    # attention
    (r"(layers|mtp)\.\d+\.attn\.wq_b", OUT),
    (r"(layers|mtp)\.\d+\.attn\.wo_b", IN),
    (r"(layers|mtp)\.\d+\.attn\.(wq_a|wkv|kv_norm|q_norm)", REP),
    (r"layers\.\d+\.attn\.compressor\.(wkv|wgate|norm)", REP),
    (r"layers\.\d+\.attn\.indexer\.(wq_b|weights_proj|wk|k_norm)", REP),
    # router, norms, HC, engram, MTP extras
    (r"(layers|mtp)\.\d+\.ffn\.gate", REP),
    (r"(layers|mtp)\.\d+\.(attn_norm|ffn_norm|norm|main_norm)", REP),
    (r"layers\.\d+\.engram\.wkv", REP),
    (r"mtp\.\d+\.main_proj", REP),
    (r"mtp\.\d+\.confidence_head\.proj", REP),
    (r"norm", REP),
    # embedding, head, MTP markov head (vocab-row split)
    (r"embed", OUT),
    (r"head", OUT),
    (r"mtp\.\d+\.markov_head\.markov_w[12]", OUT),
    # aligner
    (r"aligner\.w1", OUT),
    (r"aligner\.w2", IN),
    # vision
    (r"vision\.blocks\.\d+\.attn\.wqkv", "VWQKV"),
    (r"vision\.blocks\.\d+\.attn\.wo", IN),
    (r"vision\.blocks\.\d+\.mlp\.w1", "VW1"),
    (r"vision\.blocks\.\d+\.mlp\.w2", IN),
    (r"vision\.blocks\.\d+\.(norm1|norm2)", REP),
    (r"vision\.(norm|patch_embed\.proj)", REP),
]

# Full tensor name (no module suffix) -> rule.
FULLNAME_RULES = [
    (r"(layers|mtp)\.\d+\.attn\.attn_sink", "SINK"),
    (r"(layers|mtp)\.\d+\.(hc_attn|hc_ffn)_(base|fn|scale)", REP),
    (r"layers\.\d+\.attn\.compressor\.ape", REP),
    (r"layers\.\d+\.engram\.(k_weight|q_weight)", REP),
    (r"image_(start|newline|end)", REP),
]

WOA_RE = re.compile(r"((?:layers|mtp)\.\d+\.attn\.wo_a)\.slice\.(\d+)\.(trellis|suh|svh|mul1)")


def module_role(prefix):
    hits = [role for rx, role in MODULE_ROLES if re.fullmatch(rx, prefix)]
    if len(hits) != 1:
        raise ShardError(f"module '{prefix}' matches {len(hits)} rules (need exactly 1): {hits}")
    return hits[0]


def half(shape, dim, r):
    n = shape[dim]
    if n % TP:
        raise ShardError(f"dim {dim} of {shape} not divisible by {TP}")
    k = n // TP
    return (dim, r * k, (r + 1) * k)


def pieces_for(role, kind, suffix, shape, r):
    """Return list of (dim, start, end) slices to concat along pieces[0][0]; [] = whole tensor;
    None = tensor not present on rank r."""
    if role == REP:
        return []
    if kind == "plain":  # Linear weight [out,in] or bias [out]
        if suffix == "weight":
            return [half(shape, 0, r)] if role == OUT else [half(shape, 1, r)]
        if role == OUT:  # bias
            return [half(shape, 0, r)]
        return [] if r == 0 else None  # row-parallel bias only on rank 0
    # EXL3: trellis [in/16, out/16, 16K], suh [in], svh [out], mul1 scalar, optional bias [out]
    if suffix == "mul1":  # scalar flag, copied whole on both ranks
        return []
    if role == OUT:
        table = {"trellis": lambda: [half(shape, 1, r)], "svh": lambda: [half(shape, 0, r)],
                 "bias": lambda: [half(shape, 0, r)], "suh": lambda: []}
    else:
        table = {"trellis": lambda: [half(shape, 0, r)], "suh": lambda: [half(shape, 0, r)],
                 "svh": lambda: [], "bias": lambda: [] if r == 0 else None}
    if suffix not in table:
        raise ShardError(f"unexpected EXL3 component '{suffix}' (shape {shape})")
    return table[suffix]()


def plan(name, shape):
    """Return (rule, {rank: (dst_name, pieces) | None}) for one source tensor. Raises if unmatched."""
    m = WOA_RE.fullmatch(name)
    if m:  # o-group split: slices 0-3 -> rank0 (renamed 0-3), slices 4-7 -> rank1 (renamed 0-3)
        pre, k, comp = m.group(1), int(m.group(2)), m.group(3)
        if k >= 2 * 4:
            raise ShardError(f"wo_a slice index out of range: {name}")
        return "wo_a-rename", {r: ((f"{pre}.slice.{k % 4}.{comp}", []) if r == k // 4 else None)
                               for r in range(TP)}

    for rx, rule in FULLNAME_RULES:
        if re.fullmatch(rx, name):
            if rule == "SINK":
                return "sink-split", {r: (name, [half(shape, 0, r)]) for r in range(TP)}
            return "rep", {r: (name, []) for r in range(TP)}

    prefix, _, suffix = name.rpartition(".")
    if not prefix:
        raise ShardError(f"unmatched tensor (no module prefix): {name}")
    role = module_role(prefix)
    if role == REP:  # replicated: any suffix (e.g. gate.bias_vl) is copied whole
        return "rep", {r: (name, []) for r in range(TP)}
    kind = "exl3" if suffix in EXL3_PARTS else "plain"
    if suffix not in EXL3_PARTS and suffix not in ("weight", "bias"):
        raise ShardError(f"unmatched tensor suffix: {name}")

    if role in ("VWQKV", "VW1"):
        if suffix not in ("weight", "bias"):
            raise ShardError(f"vision fused tensor with suffix '{suffix}': {name}")
        b = shape[0] // (3 if role == "VWQKV" else 2)  # per-part rows (q|k|v or gate|up)
        h = b // TP
        out = {}
        for r in range(TP):
            if role == "VWQKV":  # q,k,v each 1024 rows -> take 512 rows of each part for this rank
                pcs = [(0, t * b + r * h, t * b + (r + 1) * h) for t in range(3)]
            else:  # fused gate|up (2816 each) -> 1408 rows of each for this rank
                pcs = [(0, t * b + r * h, t * b + (r + 1) * h) for t in range(2)]
            out[r] = (name, pcs)
        return ("vision-wqkv" if role == "VWQKV" else "vision-mlp-w1"), out

    rule_role = {OUT: "out", IN: "in", REP: "rep"}[role]
    rule = "rep" if role == REP else f"{kind}-{rule_role}"
    out = {}
    for r in range(TP):
        pcs = pieces_for(role, kind, suffix, shape, r)
        out[r] = None if pcs is None else (name, pcs)
    return rule, out



def read_headers(src):
    """Return ordered list of (name, src_file_basename, dtype, shape) from safetensors headers (no tensor reads)."""
    files = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    if not files:
        raise ShardError(f"no .safetensors in {src}")
    out = []
    for fn in files:
        with open(os.path.join(src, fn), "rb") as fh:
            n = int.from_bytes(fh.read(8), "little")
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            out.append((k, fn, v["dtype"], tuple(v["shape"])))
    return out


def dst_shape(src_shape, pcs):
    if not pcs:
        return tuple(src_shape)
    dim = pcs[0][0]
    s = list(src_shape)
    s[dim] = sum(e - b for _, b, e in pcs)
    return tuple(s)


def nbytes(dtype, shape):
    n = 1
    for d in shape:
        n *= d
    return n * DTYPE_BYTES[dtype]


def build_plan(src, bin_bytes):
    """Full plan: every source tensor -> per-rank dst entries in source-file order, grouped into output bins.

    Returns (entries, bins) where entries = list of dicts {src_name, src_file, src_shape, dtype, rule, rank_out:
    {r: {dst_name, pieces, dst_shape, nbytes, bin}} } and bins[r] = list of bin dicts {file, members}.
    Raises ShardError on any unmatched tensor."""
    headers = read_headers(src)
    entries = []
    bins = {r: [] for r in range(TP)}
    open_bin = {r: None for r in range(TP)}
    seen_dst = {r: set() for r in range(TP)}
    for name, fn, dtype, shape in headers:
        rule, per_rank = plan(name, shape)
        e = {"src_name": name, "src_file": fn, "src_shape": shape, "dtype": dtype, "rule": rule, "rank_out": {}}
        for r in range(TP):
            spec = per_rank.get(r)
            if spec is None:
                continue
            dname, pcs = spec
            if dname in seen_dst[r]:
                raise ShardError(f"duplicate dst name on rank {r}: {dname}")
            seen_dst[r].add(dname)
            dshape = dst_shape(shape, pcs)
            nb = nbytes(dtype, dshape)
            if open_bin[r] is None or open_bin[r]["bytes"] + nb > bin_bytes and open_bin[r]["members"]:
                open_bin[r] = {"bytes": 0, "members": 0}
                bins[r].append(open_bin[r])
            open_bin[r]["bytes"] += nb
            open_bin[r]["members"] += 1
            bi = len(bins[r]) - 1
            e["rank_out"][r] = {"dst_name": dname, "pieces": pcs, "dst_shape": dshape, "nbytes": nb, "bin": bi}
        entries.append(e)
    total = {r: sum(b["bytes"] for b in bins[r]) for r in range(TP)}
    return entries, bins, total


def slice_tensor(t, pcs, name):
    """Materialize the rank-local tensor from source tensor t and its pieces (contiguous copy)."""
    if not pcs:
        return t.clone().contiguous()
    parts = [t.narrow(d, b, e - b) for d, b, e in pcs]
    if len(parts) == 1:
        return parts[0].clone().contiguous()
    import torch
    return torch.cat(parts, dim=pcs[0][0]).contiguous()


def write_rank_shards(src, out, entries, bins, total, log):
    """Stream source shards one at a time; write each rank's dst bins as soon as they are complete."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    n_bins = {r: len(bins[r]) for r in range(TP)}
    remaining = {r: [b["members"] for b in bins[r]] for r in range(TP)}
    pending = {r: {} for r in range(TP)}
    file_names = {r: [f"model-{i + 1:05d}-of-{n_bins[r]:05d}.safetensors" for i in range(n_bins[r])]
                  for r in range(TP)}
    weight_map = {r: {} for r in range(TP)}
    manifest = {r: {} for r in range(TP)}
    for r in range(TP):
        os.makedirs(os.path.join(out, f"rank{r}"), exist_ok=True)

    by_file = {}
    for e in entries:
        by_file.setdefault(e["src_file"], []).append(e)

    for fn, members in by_file.items():
        t0 = time.time()
        with safe_open(os.path.join(src, fn), framework="pt", device="cpu") as f:
            for e in members:
                t = f.get_tensor(e["src_name"])
                if tuple(t.shape) != e["src_shape"]:
                    raise ShardError(f"{e['src_name']}: read shape {tuple(t.shape)} != header {e['src_shape']}")
                if DTYPE_OF_TORCH.get(str(t.dtype)) != e["dtype"]:
                    raise ShardError(f"{e['src_name']}: read dtype {t.dtype} != header {e['dtype']}")
                for r, ro in e["rank_out"].items():
                    local = slice_tensor(t, ro["pieces"], ro["dst_name"])
                    if tuple(local.shape) != ro["dst_shape"]:
                        raise ShardError(f"{e['src_name']} rank{r}: shape {tuple(local.shape)} != {ro['dst_shape']}")
                    b = ro["bin"]
                    pending[r].setdefault(b, {})[ro["dst_name"]] = local
                    weight_map[r][ro["dst_name"]] = file_names[r][b]
                    manifest[r][ro["dst_name"]] = {
                        "rule": e["rule"], "src_shape": list(e["src_shape"]), "dst_shape": list(ro["dst_shape"]),
                        "src_file": fn, "dst_file": file_names[r][b], "src_name": e["src_name"],
                        "dtype": e["dtype"],
                    }
                    remaining[r][b] -= 1
                    if remaining[r][b] == 0:
                        dst_path = os.path.join(out, f"rank{r}", file_names[r][b])
                        tmp = dst_path + ".tmp"
                        save_file(pending[r].pop(b), tmp, metadata={"format": "pt", "tp_rank": str(r),
                                                                     "tp_size": str(TP)})
                        os.replace(tmp, dst_path)
                        log(f"rank{r} wrote {file_names[r][b]} ({os.path.getsize(dst_path)/2**30:.2f} GiB)")
        log(f"source {fn} done in {time.time() - t0:.1f}s")
    return weight_map, manifest


def finalize_rank(src, out, r, weight_map, manifest, total_bytes, args):
    rank_dir = os.path.join(out, f"rank{r}")
    index = {"metadata": {"total_size": total_bytes[r]}, "weight_map": dict(sorted(weight_map[r].items()))}
    with open(os.path.join(rank_dir, "model.safetensors.index.json"), "w") as fh:
        json.dump(index, fh, indent=2)
    with open(os.path.join(src, "config.json")) as fh:
        cfg = json.load(fh)
    for k in ("tp_rank", "tp_size", "tp_shard_spec"):
        if k in cfg:
            raise ShardError(f"source config already has '{k}'")
    cfg["tp_rank"] = r
    cfg["tp_size"] = TP
    cfg["tp_shard_spec"] = SPEC_PATH
    with open(os.path.join(rank_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(rank_dir, "shard_manifest.json"), "w") as fh:
        json.dump({"tp_rank": r, "tp_size": TP, "tensors": dict(sorted(manifest[r].items()))}, fh, indent=1)
    for fn in os.listdir(src):
        if fn.startswith("tokenizer") or fn.endswith(".jinja") or fn in ("special_tokens_map.json",
                                                                          "generation_config.json",
                                                                          "quantization_config.json"):
            shutil.copy2(os.path.join(src, fn), os.path.join(rank_dir, fn))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bin-gib", type=float, default=4.0, help="target output shard size (GiB)")
    ap.add_argument("--dry-run", action="store_true", help="plan and report only; write nothing")
    args = ap.parse_args(argv)

    def log(msg):
        print(f"[shard_tool {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    t0 = time.time()
    entries, bins, total = build_plan(args.src, int(args.bin_gib * 2**30))
    rules = {}
    for e in entries:
        rules[e["rule"]] = rules.get(e["rule"], 0) + 1
    log(f"planned {len(entries)} source tensors in {time.time() - t0:.1f}s")
    for k in sorted(rules):
        log(f"  rule {k}: {rules[k]}")
    for r in range(TP):
        n = sum(1 for e in entries if r in e["rank_out"])
        log(f"rank{r}: {n} tensors, {len(bins[r])} bins, {total[r] / 2**30:.3f} GiB")
    if args.dry_run:
        return 0

    os.makedirs(args.out, exist_ok=True)
    weight_map, manifest = write_rank_shards(args.src, args.out, entries, bins, total, log)
    for r in range(TP):
        finalize_rank(args.src, args.out, r, weight_map, manifest, total, args)
        log(f"rank{r} finalized")
    log(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
