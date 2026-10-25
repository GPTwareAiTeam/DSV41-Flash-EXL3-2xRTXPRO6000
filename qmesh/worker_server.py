#!/usr/bin/env python3
"""GX1 EXL3 cold-expert worker.

Loads only the placement's warm expert tensors, then executes routed FFN work
sent by the P7 engine. This is the correctness-first path: one GEMV per selected
expert, FP32 accumulation.
"""
from __future__ import annotations

import argparse
import json
import pickle
import socket
import struct
import time

import torch

from exllamav3 import Config, Model


def recv_msg(conn):
    header = conn.recv(8)
    if not header:
        return None
    size = struct.unpack("<Q", header)[0]
    data = bytearray()
    while len(data) < size:
        block = conn.recv(size - len(data))
        if not block:
            raise EOFError
        data += block
    return pickle.loads(data)


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack("<Q", len(data)) + data)


def forward_experts(ffn, x, ids, weights):
    out = torch.zeros((x.shape[0], ffn.hidden_size), dtype=torch.float32, device=x.device)
    for row in range(x.shape[0]):
        for local, weight in zip(ids[row].tolist(), weights[row].tolist()):
            if local < 0:
                continue
            xi = x[row:row + 1]
            g = ffn.gates[local].forward(xi, {})
            u = ffn.ups[local].forward(xi, {})
            h = torch.nn.functional.silu(g) * u
            d = ffn.downs[local].forward(h, {})
            out[row] += d.float().reshape(-1) * weight
    return out


def load_worker(model_dir, placement_path, device):
    placement = json.load(open(placement_path))
    config = Config.from_directory(model_dir)
    config.infer_params.no_reconstruct = True
    model = Model.from_config(config)
    layers = {}
    for layer in range(40):
        cold = placement["layers"][str(layer)]["warm"]
        block = next(m for m in model.modules if m.key == f"layers.{layer}")
        ffn = block.mlp
        ffn.gates = [ffn.gates[e] for e in cold]
        ffn.ups = [ffn.ups[e] for e in cold]
        ffn.downs = [ffn.downs[e] for e in cold]
        keep = set(ffn.gates + ffn.ups + ffn.downs)
        router_key = f"layers.{layer}.ffn.gate"
        ffn.modules = [m for m in ffn.modules if m in keep or m.key == router_key]
        ffn.num_local_experts = len(cold)
        ffn.routing_first = 0
        ffn.routing_last = len(cold)
        ffn.cpu_split_first = None
        ffn._split_perm = None
        ffn.load(device)
        torch.cuda.synchronize()
        layers[layer] = ffn
        print(f"worker layer {layer}: {len(cold)} cold experts, "
              f"{torch.cuda.memory_allocated(0) / 2**30:.1f} GiB", flush=True)
    return layers


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
    print(f"WORKER_READY {len(layers)} layers in {time.time() - t0:.1f}s", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"LISTEN {args.host}:{args.port}", flush=True)
    while True:
        conn, addr = srv.accept()
        print(f"CLIENT {addr}", flush=True)
        try:
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    break
                cmd = msg["cmd"]
                if cmd == "ping":
                    send_msg(conn, {"ok": True})
                elif cmd == "forward":
                    layer = int(msg["layer"])
                    x = msg["x"].to(device, non_blocking=False)
                    ids = msg["ids"].to(device, non_blocking=False)
                    weights = msg["weights"].to(device, non_blocking=False)
                    out = forward_experts(layers[layer], x, ids, weights)
                    send_msg(conn, {"ok": True, "out": out.cpu()})
                else:
                    send_msg(conn, {"ok": False, "error": f"unknown {cmd}"})
        except Exception as e:
            print(f"CLIENT_ERROR {e!r}", flush=True)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
