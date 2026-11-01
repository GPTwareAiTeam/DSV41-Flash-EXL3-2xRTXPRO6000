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
from qmesh.native import qes as _qes

DEV = os.environ.get("SEAM_DEV", "rocep142s0f0")
PEER = os.environ.get("SEAM_PEER", "10.0.100.2")
PORT = int(os.environ.get("SEAM_PORT", "47311"))
REPS = int(os.environ.get("BENCH_REPS", "5"))
MAX_LAYERS = int(os.environ.get("BENCH_LAYERS", "0"))
PLACEMENT = os.environ.get("PLACEMENT", os.path.join(os.path.dirname(__file__), "placement.final.json"))
ROWS = [int(r) for r in os.environ.get("BENCH_ROWS", "1,2,3,4,5,6,7").split(",")]

# Wire slab geometry, mirrored from qmesh/native/qmesh_expert_seam.cu:geometry(): one pinned slab per
# row bucket *per side* (client cudaHostAlloc at qes_client_handle, server at qes_server_open), with
# QES_NL layer strides of align(req)+align(resp)+2 lines.
LINE = 64
NL = _qes.NL


def slab_bytes(rows: int) -> int:
    al = lambda v: (v + LINE - 1) // LINE * LINE  # noqa: E731
    req = rows * (HIDDEN * 2 + TOPK * 4 + TOPK * 2)
    resp = rows * HIDDEN * 4
    return (al(req) + al(resp) + 2 * LINE) * NL


def mem(tag: str, buckets=(), extra: str = "") -> None:
    """Memory state of the client side: container cgroup, this process, and the pinned slabs."""
    cur = 0
    try:
        cur = int(open("/sys/fs/cgroup/memory.current").read().strip())
    except OSError:
        pass
    rss = 0
    try:
        rss = int(open("/proc/self/statm").read().split()[1]) * 4096
    except (OSError, IndexError):
        pass
    pred = sum(slab_bytes(b) for b in buckets) / 2 ** 20
    print(f"SEAMROWS_MEM tag={tag} cgroup_mib={cur / 2 ** 20:.1f} "
          f"self_rss_mib={rss / 2 ** 20:.1f} slab_predict_mib={pred:.1f}"
          + (f" {extra}" if extra else ""), flush=True)


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

    # Only the buckets these row counts pad into are opened: the client allocates one pinned slab per
    # bucket it touches, so warming every bucket would hide the per-row-count memory cost.
    big = max(host.buckets)
    run_rows = [n for n in ROWS if n <= big]
    for n in ROWS:
        if n > big:
            print(f"SEAMROWS_DROP n={n} reason=no-bucket max_bucket={big}", flush=True)
    need = sorted({next(b for b in host.buckets if b >= n) for n in run_rows})
    mem("connect", ())
    # Warm-up: connect one data handle per bucket in use, settle kernels and the slab copies
    for b in need:
        host.register_layer(f"layers.{layers[0]}", None)
        host.submit_issue(layers[0], torch.zeros(b, HIDDEN, dtype=torch.bfloat16, device=dev),
                          torch.zeros(b, TOPK, dtype=torch.int32, device=dev),
                          torch.zeros(b, TOPK, dtype=torch.bfloat16, device=dev))
    torch.cuda.synchronize(dev)
    mem("warm", need, f"buckets={','.join(map(str, need))}")

    table = {}
    opened = []
    t_start = time.time()
    for n in run_rows:
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
        b = next(b for b in host.buckets if b >= n)
        if b not in opened:
            opened.append(b)
        print(f"SEAMROWS n={n} bucket={b} layers={len(layers)} "
              f"dev_us_median={md:.1f} dev_us_min={min(dev_us):.1f} dev_us_max={max(dev_us):.1f} "
              f"host_us_median={mh:.1f} dev_us_per_row={md / n:.1f} dev_us_per_slot={md / b:.1f} "
              f"dev_ms_all_layers={md * len(layers) / 1e3:.2f}", flush=True)
        print(f"SEAMROWS_MARK n={n} bucket={b}", flush=True)
        mem(f"n={n}", sorted(opened), f"bucket={b}")
    print("SEAMROWS_TABLE rows bucket dev_us_median dev_us_per_row host_us_median", flush=True)
    for n, (md, mh) in table.items():
        b = next(b for b in host.buckets if b >= n)
        print(f"SEAMROWS_TABLE {n} {b} {md:.1f} {md / n:.1f} {mh:.1f}", flush=True)
    print(f"SEAMROWS_DONE wall_s={time.time() - t_start:.1f}", flush=True)


if __name__ == "__main__":
    main()
