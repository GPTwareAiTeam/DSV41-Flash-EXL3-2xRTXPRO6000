#!/usr/bin/env python3
"""GX1 EXL3 worker using one fused uniform-K MoE module per bpw group."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import pickle
import socket
import struct
import time

import torch
from exllamav3 import Config, Model


def recv_msg(conn):
    h = conn.recv(8)
    if not h:
        return None
    n = struct.unpack("<Q", h)[0]
    b = bytearray()
    while len(b) < n:
        b += conn.recv(n - len(b))
    return pickle.loads(b)


def send_msg(conn, obj):
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("<Q", len(b)) + b)


def build_group(base, indices, device, layer):
    ffn = copy.deepcopy(base)
    ffn.gates = [ffn.gates[e] for e in indices]
    ffn.ups = [ffn.ups[e] for e in indices]
    ffn.downs = [ffn.downs[e] for e in indices]
    keep = set(ffn.gates + ffn.ups + ffn.downs)
    ffn.modules = [m for m in ffn.modules if m in keep]
    ffn.num_experts = len(indices)
    ffn.num_local_experts = len(indices)
    ffn.routing_first = 0
    ffn.routing_last = len(indices)
    ffn.routing_gate = None
    ffn.shared_experts = None
    ffn.cpu_split_first = None
    ffn._split_perm = None
    ffn.load_routing = lambda **kw: None
    ffn.load(device)
    torch.cuda.synchronize()
    print(f"group layer={layer} n={len(indices)} vram={torch.cuda.memory_allocated(0)/2**30:.1f}", flush=True)
    return ffn


def load_worker(model_dir, placement_path, device):
    placement = json.load(open(placement_path))
    rank = {}
    rank_path = os.path.join(model_dir, "ranking.csv")
    for r in csv.DictReader(open(rank_path)):
        rank[(int(r["layer"]), int(r["expert_id"]))] = int(r["rank"])

    def rate(layer, expert):
        r = rank[(layer, expert)]
        return 4.0 if r <= 4100 else 3.0 if r <= 7660 else 2.5 if r <= 10360 else 2.0

    config = Config.from_directory(model_dir)
    model = Model.from_config(config)
    layers = {}
    for layer in range(40):
        cold = placement["layers"][str(layer)]["warm"]
        base = next(m for m in model.modules if m.key == f"layers.{layer}").mlp
        groups = {}
        mapping = {}
        for rate_value in (3.0, 2.5, 2.0):
            positions = [i for i, e in enumerate(cold) if rate(layer, e) == rate_value]
            if not positions:
                continue
            experts = [cold[i] for i in positions]
            groups[rate_value] = build_group(base, experts, device, layer)
            for local, pos in enumerate(positions):
                mapping[pos] = (rate_value, local)
        layers[layer] = (groups, mapping)
        del base
        torch.cuda.empty_cache()
        print(f"worker layer {layer}: {len(cold)} cold experts, "
              f"{torch.cuda.memory_allocated(0)/2**30:.1f} GiB", flush=True)
    return layers


_streams = {}

def forward_group(layers, layer, x, ids, weights):
    groups, mapping = layers[layer]
    out = None
    x = x.to("cuda:0", non_blocking=False)
    current = torch.cuda.current_stream()
    pending = []
    for rate_value, ffn in groups.items():
        sel = torch.zeros_like(ids)
        gw = torch.zeros_like(weights)
        hit = False
        for r in range(ids.shape[0]):
            for k in range(ids.shape[1]):
                cid = int(ids[r, k])
                if cid < 0:
                    continue
                mapped = mapping.get(cid)
                if mapped is not None and mapped[0] == rate_value:
                    sel[r, k] = mapped[1]
                    gw[r, k] = weights[r, k]
                    hit = True
        if not hit:
            continue
        stream = _streams.get(rate_value)
        if stream is None:
            stream = torch.cuda.Stream(device="cuda:0")
            _streams[rate_value] = stream
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            part = ffn.forward(x, {"external_selected": sel.to("cuda:0"),
                                   "external_weights": gw.to(torch.half).to("cuda:0")})
        pending.append((stream, part))
    for stream, part in pending:
        current.wait_stream(stream)
        out = part if out is None else out + part
    if out is None:
        out = torch.zeros((x.shape[0], 5120), dtype=torch.float32, device="cuda:0")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--placement", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=29517)
    args = ap.parse_args()
    device = torch.device("cuda:0")
    t0 = time.time()
    layers = load_worker(args.model_dir, args.placement, device)
    print(f"WORKER_READY {len(layers)} layers in {time.time()-t0:.1f}s", flush=True)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port)); srv.listen(1)
    print(f"LISTEN {args.host}:{args.port}", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"CLIENT {addr}", flush=True)
        try:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break
                if msg["cmd"] == "stats":
                    fields = open("/proc/self/stat").read().split(") ", 1)[1].split()
                    mem = {k: int(v.split()[0]) for k, v in (l.split(":", 1) for l in open("/proc/meminfo"))}
                    send_msg(conn, {"ok": True, "majflt": int(fields[9]), "minflt": int(fields[7]),
                                    "allocated": torch.cuda.memory_allocated(0),
                                    "mem_free_kb": mem["MemFree"], "mem_available_kb": mem["MemAvailable"]})
                elif msg["cmd"] == "forward":
                    out = forward_group(layers, int(msg["layer"]), msg["x"], msg["ids"], msg["weights"])
                    send_msg(conn, {"ok": True, "out": out.cpu()})
                else:
                    send_msg(conn, {"ok": False, "error": "bad cmd"})
        except Exception as e:
            print(f"CLIENT_ERROR {e!r}", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
