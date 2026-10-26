#!/usr/bin/env python3
"""GX1 expert server for the native P7<->GX1 seam (qmesh/native/qes.py).

One fused mixed-K BlockSparseMLP per layer holds that layer's cold experts (the
placement's `warm` list, in order). A seam request carries cold-local ids [rows,6]
(-1 = expert resident on P7) and routing weights [rows,6]. The compute hook runs on the
seam's proxy thread: it copies the request out of the pinned slab, runs the module with
`external_selected`/`external_weights` on a dedicated CUDA stream, and writes fp32 output
back into the pinned response region.

-1 picks are mapped to local expert 0 with weight 0, so they contribute exact zeros
(the module's own range check stays satisfied).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
from exllamav3 import Config, Model

from qmesh.native import qes

HIDDEN = qes.HIDDEN
TOPK = qes.TOPK
NL = qes.NL


def build_layer(base, cold_ids, device):
    """Subset the layer's routed-expert module to `cold_ids` (checkpoint expert ids, ordered).

    Mirrors worker_group_server.build_group, with all cold experts in one module so the
    fused mixed-K path runs one launch per layer."""
    ffn = copy.deepcopy(base)
    ffn.gates = [ffn.gates[e] for e in cold_ids]
    ffn.ups = [ffn.ups[e] for e in cold_ids]
    ffn.downs = [ffn.downs[e] for e in cold_ids]
    keep = set(ffn.gates + ffn.ups + ffn.downs)
    ffn.modules = [m for m in ffn.modules if m in keep]
    ffn.num_experts = len(cold_ids)
    ffn.num_local_experts = len(cold_ids)
    ffn.routing_first = 0
    ffn.routing_last = len(cold_ids)
    ffn.routing_gate = None
    ffn.shared_experts = None
    ffn.cpu_split_first = None
    ffn._split_perm = None
    ffn.load_routing = lambda **kw: None
    ffn.load(device)
    torch.cuda.synchronize()
    return ffn, len(cold_ids)


class SeamServer:
    def __init__(self, model_dir, placement_path, device):
        placement = json.load(open(placement_path))
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        config = Config.from_directory(model_dir)
        model = Model.from_config(config)
        self.layers = {}
        for layer in range(NL):
            cold = placement["layers"][str(layer)]["warm"]
            base = next(m for m in model.modules if m.key == f"layers.{layer}").mlp
            self.layers[layer] = build_layer(base, cold, device)
            del base
            torch.cuda.empty_cache()
            print(f"layer {layer}: {len(cold)} cold experts, "
                  f"{torch.cuda.memory_allocated(device) / 2**30:.2f} GiB", flush=True)

    def compute(self, layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr):
        """Pinned-slab pointers in; fp32 output written into the pinned response region."""
        ffn, _ = self.layers[layer]
        x = torch.from_numpy(view(x_ptr, rows * HIDDEN * 2, np.uint16).reshape(rows, HIDDEN).copy())
        x = x.view(torch.bfloat16).to(self.device).half()
        ids = torch.from_numpy(view(ids_ptr, rows * TOPK * 4, np.int32).reshape(rows, TOPK).astype(np.int64))
        w = torch.from_numpy(view(w_ptr, rows * TOPK * 2, np.uint16).reshape(rows, TOPK).copy())
        w = w.view(torch.bfloat16).float()

        miss = ids < 0
        sel = ids.clamp_min(0).to(self.device)
        gw = torch.where(miss, torch.zeros_like(w), w).to(self.device).half()
        with torch.cuda.stream(self.stream):
            part = ffn.forward(x.view(1, rows, HIDDEN),
                               {"external_selected": sel, "external_weights": gw})
        self.stream.synchronize()
        view(out_ptr, rows * HIDDEN * 4, np.float32).reshape(rows, HIDDEN)[:] = \
            part.view(rows, HIDDEN).float().cpu().numpy()


def view(addr, nbytes, dtype):
    """numpy view over raw pinned-slab memory (no copy)."""
    import ctypes
    buf = (ctypes.c_byte * nbytes).from_address(addr)
    return np.frombuffer(buf, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/cold")
    ap.add_argument("--placement", default="/cold/placement.exl3.json")
    ap.add_argument("--dev", default="rocep1s0f0")
    ap.add_argument("--gid", type=int, default=3)
    ap.add_argument("--port", type=int, default=47311)
    ap.add_argument("--buckets", default="1,6,64",
                    help="row buckets; one seam handle per bucket on consecutive ports from --port")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    t0 = time.time()
    server = SeamServer(args.model_dir, args.placement, device)
    print(f"SEAM_SERVER_LOADED {time.time() - t0:.1f}s", flush=True)

    # One handle per row bucket, on consecutive ports; the client uses the same mapping.
    seams = []
    for i, rows in enumerate(int(r) for r in args.buckets.split(",")):
        port = args.port + i
        seam = qes.Seam(qes.ROLE_SERVER, args.dev, args.gid, None, port, rows, 0)
        seam.set_server_fn(lambda L, R, x, ids, w, o: server.compute(L, R, x, ids, w, o))
        seams.append(seam)  # keep alive: the proxy thread owns the handle
        print(f"SEAM_LISTEN port={port} rows={rows}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
