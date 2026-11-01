#!/usr/bin/env python3
"""Pack only what P7 GPU0 loads into a standalone checkpoint dir on fast local storage.

Counterpart of pack_cold.py (GX side). P7 loads every non-expert tensor, the drafter, and the hot
routed experts of each layer. For the cold experts of a layer (computed on GX) P7 only reads small
per-expert tensors at load: suh/svh/mul1/bias of every cold expert (registered with the remote host)
and the header shape of the first cold expert's trellis (bitrate probe in
BlockSparseMLP split setup). So the output keeps:
  * every tensor that is not a trunk routed-expert trellis
  * routed-expert trellis of the hot experts, plus the first cold expert of each layer
Every kept tensor's bytes are copied verbatim (same keys, dtypes, shapes, header metadata), one output
shard per source shard, then re-read and sha256-compared against the source bytes.

Usage: pack_hot.py --checkpoint SRC_DIR --placement placement.final.json --out OUT_DIR
SRC_DIR is scanned for *.safetensors like the runtime does (so an overlay such as the HC1 one, whose
drafter experts live in model-drafter-experts.safetensors, packs correctly). Small non-safetensors
files are copied. Shards that already live on the output's filesystem and need no filtering (the Engram
tables, the HC1 drafter-expert file) are symlinked, not copied; everything read from another device is
written as a real file.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import struct
import time

CHUNK = 64 * 1024 * 1024
EXPERT_TRELLIS = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w\d\.trellis$")


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--placement", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    placement = json.load(open(args.placement))
    num_experts = None
    hot, probe = {}, {}
    for layer, v in placement["layers"].items():
        hot[int(layer)] = set(v["hot"])
        cold = sorted(set(v["warm"]))
        probe[int(layer)] = cold[0]  # BlockSparseMLP orders cold ascending after the hot list

    def keep(key):
        m = EXPERT_TRELLIS.match(key)
        if not m:
            return True
        layer, e = int(m.group(1)), int(m.group(2))
        if layer not in hot:
            return True
        return e in hot[layer] or e == probe[layer]

    os.makedirs(args.out, exist_ok=True)
    for name in sorted(os.listdir(args.checkpoint)):
        src = os.path.join(args.checkpoint, name)
        if name.endswith(".safetensors") or name == "model.safetensors.index.json" or not os.path.isfile(src):
            continue
        shutil.copyfile(src, os.path.join(args.out, name))

    shards = sorted(glob.glob(os.path.join(args.checkpoint, "*.safetensors")))
    weight_map, digests = {}, {}
    total = kept_n = dropped_n = 0
    t0 = time.time()
    for spath in shards:
        fname = os.path.basename(spath)
        header, base = read_header(spath)
        meta = header.pop("__metadata__", None)
        keys = sorted((k for k in header if keep(k)), key=lambda k: header[k]["data_offsets"][0])
        dropped_n += len(header) - len(keys)
        dup = [k for k in keys if k in weight_map]
        if dup:
            raise SystemExit(f"duplicate keys across shards ({fname} vs {weight_map[dup[0]]}): {dup[:3]}")
        if not keys:
            continue
        real = os.path.realpath(spath)
        if len(keys) == len(header) and os.stat(real).st_dev == os.stat(args.out).st_dev:
            os.symlink(real, os.path.join(args.out, fname))
            for k in keys:
                weight_map[k] = fname
            kept_n += len(keys)
            print(f"{fname}: linked (same device, unfiltered), {len(keys)} tensors", flush=True)
            continue
        out_header, off = {}, 0
        for k in keys:
            a, b = header[k]["data_offsets"]
            out_header[k] = {"dtype": header[k]["dtype"], "shape": header[k]["shape"], "data_offsets": [off, off + b - a]}
            off += b - a
        if meta is not None:
            out_header["__metadata__"] = meta
        enc = json.dumps(out_header, separators=(",", ":")).encode()
        enc += b" " * (-len(enc) % 8)
        dpath = os.path.join(args.out, fname)
        with open(spath, "rb", buffering=0) as src, open(dpath, "wb") as dst:
            dst.write(struct.pack("<Q", len(enc)))
            dst.write(enc)
            for k in keys:
                a, b = header[k]["data_offsets"]
                src.seek(base + a)
                h, left = hashlib.sha256(), b - a
                while left:
                    blk = src.read(min(left, CHUNK))
                    if not blk:
                        raise EOFError(f"{fname}:{k}")
                    h.update(blk)
                    dst.write(blk)
                    left -= len(blk)
                digests[k] = h.hexdigest()
                weight_map[k] = fname
        kept_n += len(keys)
        total += os.path.getsize(dpath)
        print(f"{fname}: kept {len(keys)}/{len(header)} tensors, {os.path.getsize(dpath) / 2**30:.2f} GiB, "
              f"{total / 2**30:.1f} GiB total, {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)

    # Verify: re-read every copied tensor from the output and compare to the source-side digest
    bad = 0
    for fname in sorted(set(weight_map.values())):
        if os.path.islink(os.path.join(args.out, fname)):
            continue
        header, base = read_header(os.path.join(args.out, fname))
        header.pop("__metadata__", None)
        with open(os.path.join(args.out, fname), "rb") as f:
            for k, v in header.items():
                a, b = v["data_offsets"]
                f.seek(base + a)
                h, left = hashlib.sha256(), b - a
                while left:
                    blk = f.read(min(left, CHUNK))
                    h.update(blk)
                    left -= len(blk)
                if h.hexdigest() != digests[k]:
                    bad += 1
                    print(f"MISMATCH {fname}:{k}", flush=True)
    with open(os.path.join(args.out, "PACK_MANIFEST.json"), "w") as f:
        json.dump({"source": os.path.realpath(args.checkpoint), "placement": os.path.realpath(args.placement),
                   "placement_sha256": hashlib.sha256(open(args.placement, "rb").read()).hexdigest(),
                   "tensors": kept_n, "dropped_cold_trellis": dropped_n, "bytes": total,
                   "tensor_sha256": digests}, f)
    print(f"PACK {'OK' if not bad else 'FAIL'} tensors={kept_n} dropped={dropped_n} "
          f"size={total / 2**30:.2f} GiB mismatches={bad} {time.time() - t0:.0f}s", flush=True)
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
