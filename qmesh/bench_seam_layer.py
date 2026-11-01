#!/usr/bin/env python3
"""Per-layer fused mixed-K forward timing on GX1 (no seam). Times one layer's cold-expert
module with CUDA events, for the given rows. Run inside exl3-fork.

Usage: python3 -m qmesh.bench_seam_layer --rows 1 --layer 0 --iters 200
       python3 -m qmesh.bench_seam_layer --rows-list 1,2,3,4,5,6 --layers 5,20,35
"""
import argparse
import json
import statistics

import torch
from exllamav3 import Config, Model

from qmesh.worker_seam_server import build_layer, HIDDEN, TOPK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/cold")
    ap.add_argument("--placement", default="/cold/placement.exl3.json")
    ap.add_argument("--layers", default="0")
    ap.add_argument("--rows", type=int, default=1)
    ap.add_argument("--rows-list", default="",
                    help="comma list of row counts to time from one model load, e.g. 1,2,3,4,5,6")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    placement = json.load(open(args.placement))
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    layers = [int(x) for x in args.layers.split(",")]
    rows_list = ([int(r) for r in args.rows_list.split(",") if r.strip()]
                 if args.rows_list else [args.rows])

    for layer in layers:
        cold = placement["layers"][str(layer)]["warm"]
        base = next(m for m in model.modules if m.key == f"layers.{layer}").mlp
        ffn, _ = build_layer(base, cold, device)
        del base
        n = len(cold)
        g = torch.Generator(device="cpu").manual_seed(0)
        for rows in rows_list:
            x = torch.randn(1, rows, HIDDEN, generator=g).to(device).half()
            sel = torch.stack([torch.randperm(n, generator=g)[:TOPK] for _ in range(rows)]).to(device)
            w = torch.rand(rows, TOPK, generator=g).to(device).half()
            params = {"external_selected": sel, "external_weights": w}

            for _ in range(args.warmup):
                ffn.forward(x, params)
            torch.cuda.synchronize()

            times = []
            for _ in range(args.iters):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                ffn.forward(x, params)
                e.record()
                e.synchronize()
                times.append(s.elapsed_time(e) * 1000.0)  # ms -> us
            times.sort()
            p50 = statistics.median(times)
            p99 = times[min(len(times) - 1, int(len(times) * 0.99))]
            print(f"LAYER {layer} rows={rows} n_cold={n} iters={args.iters} "
                  f"p50_us={p50:.1f} p99_us={p99:.1f} min_us={times[0]:.1f}", flush=True)
        del ffn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
