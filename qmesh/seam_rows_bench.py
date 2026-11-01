#!/usr/bin/env python3
"""Row-count scaling of the GX1 cold-expert seam, measured on P7 without loading the model.

For every layer of the shipped placement, issue one cold-expert request per row count n = 1..7 and
time issue -> collect with CUDA events. Ids are cold-local (the server's id space, -1 = hot), so
every row fills all TOPK slots with distinct cold experts: the gather set, and with it the GX1
kernel time, is the worst case for that width.

The server pads to the smallest bucket it serves that is >= n, so the server's bucket list decides
which widths share a slab (default 1,2,3,4,5,6,64 -> one bucket per width); this table shows what a
narrower verify width does to the GX1 kernel time. The client requests SEAM_BUCKETS and adopts the
subset the server grants, so one long-lived server serves every arm: no restart per run.

Env: SEAM_DEV, SEAM_GID, SEAM_PEER, SEAM_PORT, SEAM_BUCKETS (requested), BENCH_REPS (default 5),
     BENCH_LAYERS (default 0 = all layers in the placement), PLACEMENT
"""
from __future__ import annotations

import json
import os
import statistics
import time

import torch

from qmesh.remote_seam_host import RemoteSeamHost, HIDDEN, TOPK

DEV = os.environ.get("SEAM_DEV", "rocep142s0f0")
PEER = os.environ.get("SEAM_PEER", "10.0.100.2")
PORT = int(os.environ.get("SEAM_PORT", "47311"))
REPS = int(os.environ.get("BENCH_REPS", "5"))
MAX_LAYERS = int(os.environ.get("BENCH_LAYERS", "0"))
PLACEMENT = os.environ.get("PLACEMENT", os.path.join(os.path.dirname(__file__), "placement.final.json"))
ROWS = [int(r) for r in os.environ.get("BENCH_ROWS", "1,2,3,4,5,6,7").split(",")]


def main():
    pl = json.load(open(PLACEMENT))
    cold = {int(k): len(v["warm"]) for k, v in pl["layers"].items()}
    layers = sorted(cold)
    if MAX_LAYERS:
        layers = layers[:MAX_LAYERS]
    host = RemoteSeamHost(PEER, PORT).connect()
    print(f"SEAMROWS layers={len(layers)} cold_min={min(cold[l] for l in layers)} "
          f"cold_max={max(cold[l] for l in layers)} dev={DEV} peer={PEER} "
          f"requested={','.join(map(str, host.BUCKETS))} "
          f"buckets={','.join(map(str, host.buckets))} reps={REPS} rows={ROWS}", flush=True)
    stream = torch.cuda.current_stream()
    torch.manual_seed(0)
    dev = torch.device("cuda:0")

    # Warm-up: connect one data handle per granted bucket, settle kernels and the slab copies
    for b in host.buckets:
        host.register_layer(f"layers.{layers[0]}", None)
        host.submit_issue(layers[0], torch.zeros(b, HIDDEN, dtype=torch.bfloat16, device=dev),
                          torch.zeros(b, TOPK, dtype=torch.int32, device=dev),
                          torch.zeros(b, TOPK, dtype=torch.bfloat16, device=dev))
    torch.cuda.synchronize(dev)

    table = {}
    t_start = time.time()
    for n in ROWS:
        x = (torch.randn(n, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02)
        w = torch.rand(n, TOPK, dtype=torch.bfloat16, device=dev)
        w = (w / w.sum(dim=-1, keepdim=True)).contiguous()
        dev_us, host_us = [], []
        for L in layers:
            nc = cold[L]
            ids = torch.stack([torch.randperm(nc, device=dev)[:TOPK] for _ in range(n)]).to(torch.int32)
            best_d, best_h = None, None
            for _ in range(REPS):
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record(stream)
                t0 = time.perf_counter_ns()
                host.submit_issue(L, x, ids, w)
                ev1.record(stream)
                torch.cuda.synchronize(dev)
                t1 = time.perf_counter_ns()
                d = ev0.elapsed_time(ev1) * 1e3
                h = (t1 - t0) / 1e3
                best_d = d if best_d is None else min(best_d, d)
                best_h = h if best_h is None else min(best_h, h)
            dev_us.append(best_d)
            host_us.append(best_h)
        md, mh = statistics.median(dev_us), statistics.median(host_us)
        table[n] = (md, mh)
        print(f"SEAMROWS n={n} bucket={next(b for b in host.buckets if b >= n)} layers={len(layers)} "
              f"dev_us_median={md:.1f} host_us_median={mh:.1f} dev_us_per_row={md / n:.1f} "
              f"dev_ms_all_layers={md * len(layers) / 1e3:.2f}", flush=True)
    print("SEAMROWS_TABLE rows bucket dev_us_median dev_us_per_row host_us_median", flush=True)
    for n, (md, mh) in table.items():
        b = next(b for b in host.buckets if b >= n)
        print(f"SEAMROWS_TABLE {n} {b} {md:.1f} {md / n:.1f} {mh:.1f}", flush=True)
    print(f"SEAMROWS_DONE wall_s={time.time() - t_start:.1f}", flush=True)


if __name__ == "__main__":
    main()
