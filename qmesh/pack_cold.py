#!/usr/bin/env python3
"""Pack only the GX1-cold EXL3 expert tensors into per-layer shards.

The worker needs no dense, shared-expert, head or Engram tensors. Each output
layer shard keeps the checkpoint's original tensor keys and byte payloads.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import struct

CHUNK = 64 * 1024 * 1024


def read_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len)), 8 + header_len


def copy_tensor(src, src_base, header, key, dst):
    start, end = header[key]["data_offsets"]
    src.seek(src_base + start)
    left = end - start
    while left:
        block = src.read(min(left, CHUNK))
        if not block:
            raise EOFError(key)
        dst.write(block)
        left -= len(block)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--placement", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index = json.load(open(os.path.join(args.checkpoint, "model.safetensors.index.json")))
    placement = json.load(open(args.placement))
    os.makedirs(args.out, exist_ok=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = os.path.join(args.checkpoint, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(args.out, name))

    weight_map = {}
    total_bytes = 0
    for layer, entries in placement["layers"].items():
        cold = set(entries["warm"])
        wanted = []
        for key in index["weight_map"]:
            match = re.match(rf"layers\.{layer}\.ffn\.experts\.(\d+)\.", key)
            if match and int(match.group(1)) in cold:
                wanted.append(key)
        # The worker's BlockSparseMLP still constructs a router; keep its small tensors.
        wanted += [k for k in index["weight_map"] if k.startswith(f"layers.{layer}.ffn.gate.")]
        by_file = collections.defaultdict(list)
        for key in wanted:
            by_file[index["weight_map"][key]].append(key)

        tensors = {}
        sources = {}
        for filename, keys in by_file.items():
            header, base = read_header(os.path.join(args.checkpoint, filename))
            sources[filename] = (header, base)
            for key in keys:
                if key not in header:
                    raise KeyError(f"{key} not in {filename}")
                tensors[key] = header[key]

        out_header = {}
        offset = 0
        for key, meta in tensors.items():
            start, end = meta["data_offsets"]
            out_header[key] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [offset, offset + end - start],
            }
            offset += end - start
        encoded = json.dumps(out_header, separators=(",", ":")).encode()
        filename = f"cold-L{int(layer):02d}.safetensors"
        path = os.path.join(args.out, filename)
        with open(path, "wb") as dst:
            dst.write(struct.pack("<Q", len(encoded)))
            dst.write(encoded)
            for key, meta in tensors.items():
                sfile = index["weight_map"][key]
                header, base = sources[sfile]
                with open(os.path.join(args.checkpoint, sfile), "rb") as src:
                    copy_tensor(src, base, header, key, dst)
        for key in tensors:
            weight_map[key] = filename
        total_bytes += os.path.getsize(path)
        print(f"layer {layer}: {len(cold)} cold experts, {len(tensors)} tensors, "
              f"{os.path.getsize(path) / 2**30:.2f} GiB", flush=True)

    index_out = {
        "metadata": {"total_size": total_bytes},
        "weight_map": weight_map,
    }
    with open(os.path.join(args.out, "model.safetensors.index.json"), "w") as f:
        json.dump(index_out, f, indent=2)
    print(f"total {total_bytes / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
