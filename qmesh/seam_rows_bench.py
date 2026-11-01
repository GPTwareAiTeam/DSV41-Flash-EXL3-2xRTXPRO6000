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
     BENCH_LAYERS (default 0 = all layers in the placement), PLACEMENT,
     BENCH_HOT_ROWS (default 0: that many rows of every request get an all-hot routing, id -1 in
     every slot with a nonzero weight - the drafter verify's "this token's top-k is hot" row, which
     must contribute nothing to the cold gather and must still come back as a finite row),
     BENCH_HOT_SLOTS (default 0: how many leading slots of the remaining rows are hot),
     BENCH_INTERLEAVE (default 0 = sweep one width across all layers at a time; N = N cycles of
     layer order with every width of BENCH_ROWS issued back to back on each layer, which is what a
     real client does and the only way this bench sees a per-layer scratch/streaming bug),
     BENCH_SYNC (default 1: host-synchronise after each call; 0 pipelines calls on the stream like
     the client does, so several requests are in flight across buckets).
"""
from __future__ import annotations

import json
import os
import statistics
import time

import torch

from qmesh.remote_seam_host import RemoteSeamHost, HIDDEN, TOPK
from qmesh import placement as seam_placement
from qmesh.native import qes as _qes

DEV = os.environ.get("SEAM_DEV", "rocep142s0f0")
PEER = os.environ.get("SEAM_PEER", "10.0.100.2")
PORT = int(os.environ.get("SEAM_PORT", "47311"))
REPS = int(os.environ.get("BENCH_REPS", "5"))
MAX_LAYERS = int(os.environ.get("BENCH_LAYERS", "0"))
PLACEMENT = seam_placement.resolve(os.environ.get("CHECKPOINT"))   # the pack's table when the pack ships one
ROWS = [int(r) for r in os.environ.get("BENCH_ROWS", "1,2,3,4,5,6,7").split(",")]
INTERLEAVE = int(os.environ.get("BENCH_INTERLEAVE", "0"))
SYNC = os.environ.get("BENCH_SYNC", "1") != "0"
HOT_ROWS = int(os.environ.get("BENCH_HOT_ROWS", "0"))
HOT_SLOTS = int(os.environ.get("BENCH_HOT_SLOTS", "0"))
POOL = int(os.environ.get("BENCH_POOL", "0"))

# Wire slab geometry, mirrored from qmesh/native/qmesh_expert_seam.cu:geometry() (protocol v3): one
# pinned slab per row bucket *per side* (client cudaHostAlloc in qes_client_handle, server in
# qes_server_open) holding SEAM_SLAB_SLOTS payload slots of 3 flag/header lines + align(req) +
# align(resp). The layer id rides in the request header, so this is ~30.2 KiB per row per slot
# instead of the 1.18 MiB per row the per-layer stride needed.
LINE = 64
SLOTS = int(os.environ.get("SEAM_SLAB_SLOTS", "1"))


def slab_bytes(rows: int) -> int:
    al = lambda v: (v + LINE - 1) // LINE * LINE  # noqa: E731
    req = rows * (HIDDEN * 2 + TOPK * 4 + TOPK * 2)
    resp = rows * HIDDEN * 4
    return (3 * LINE + al(req) + al(resp)) * SLOTS


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

def make_ids(n, nc, dev):
    """(n, TOPK) cold expert ids for one request, in [0, nc).

    BENCH_POOL=0 (default) spreads every assignment over the layer's whole cold range, so the
    per-expert row counts come out flat - max ~5 rows for 64 tokens over ~270 experts. Real routing
    is nothing like that: the failing arm's own tier trace printed a histogram with one expert
    holding 257+ rows and two holding 1..32, because the tokens of one sentence pick near-identical
    experts. BENCH_POOL=k draws every id from a pool of k experts, so an expert's row count is
    ~n*TOPK/k and the dispatch takes the per-row-tile launches (MTILE_T1=16, MTILE_T2=32, m_tile
    32/64) that a flat spread can never reach - the one path no earlier bench pass has exercised.
    """
    if POOL:
        return torch.randint(0, min(POOL, nc), (n, TOPK), device=dev).to(torch.int32)
    return torch.stack([torch.randperm(nc, device=dev)[:TOPK] for _ in range(n)]).to(torch.int32)


def main():
    pl, _path, _sha = seam_placement.load(os.environ.get("CHECKPOINT"))
    cold = {int(k): len(v["warm"]) for k, v in pl["layers"].items()}
    layers = sorted(cold)
    if MAX_LAYERS:
        layers = layers[:MAX_LAYERS]
    host = RemoteSeamHost(PEER, PORT).connect()
    print(f"SEAMROWS layers={len(layers)} cold_min={min(cold[l] for l in layers)} "
          f"cold_max={max(cold[l] for l in layers)} dev={DEV} peer={PEER} "
          f"requested={','.join(map(str, host.BUCKETS))} "
          f"buckets={','.join(map(str, host.buckets))} reps={REPS} rows={ROWS} "
          f"hot_rows={HOT_ROWS} hot_slots={HOT_SLOTS}", flush=True)
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

    # BENCH_INTERLEAVE: issue every row count of ROWS back to back *on the same layer*, in layer
    # order - the pattern a real client produces (a 64-row prefill piece, then its 2..6-row
    # remainder, then the next layer), and the only way this bench can show a per-layer scratch or
    # stream-ordering bug. The width sweep below runs all layers at one width and then all layers at
    # the next, so it never puts two widths through one module within a millisecond.
    if INTERLEAVE:
        widths = run_rows
        xs, ws, idss = {}, {}, {}
        for n in widths:
            xs[n] = (torch.randn(n, HIDDEN, dtype=torch.bfloat16, device=dev) * 0.02)
            w = torch.rand(n, TOPK, dtype=torch.bfloat16, device=dev)
            ws[n] = (w / w.sum(dim=-1, keepdim=True)).contiguous()
        torch.cuda.synchronize(dev)
        t0 = time.time()
        t_il = {}
        ncalls = 0
        for L in list(layers) * max(1, INTERLEAVE):
            nc = cold[L]
            for n in widths:
                idss[n] = make_ids(n, nc, dev)
                if HOT_SLOTS:
                    idss[n][:, :min(HOT_SLOTS, TOPK)] = -1
                if HOT_ROWS:
                    idss[n][:min(HOT_ROWS, n)] = -1
            for _ in range(REPS):
                # sync only per layer (SYNC=0): the client pipelines its calls on the stream and
                # never host-waits between the widths of one layer
                if SYNC:
                    torch.cuda.synchronize(dev)
                for n in widths:
                    s0 = time.perf_counter_ns()
                    host.submit_issue(L, xs[n], idss[n], ws[n])
                    if SYNC:
                        torch.cuda.synchronize(dev)
                        t_il.setdefault(n, []).append((time.perf_counter_ns() - s0) / 1e3)
                    ncalls += 1
            torch.cuda.synchronize(dev)
        for n in widths:
            v = t_il.get(n) or [0.0]
            print(f"SEAMROWS_INTERLEAVE n={n} calls={ncalls} sync={SYNC} "
                  f"host_us_median={statistics.median(v):.1f} host_us_max={max(v):.1f}", flush=True)
        print(f"SEAMROWS_INTERLEAVE_DONE layers={len(layers)} cycles={max(1, INTERLEAVE)} "
              f"widths={','.join(map(str, widths))} reps={REPS} calls={ncalls} "
              f"wall_s={time.time() - t0:.1f}", flush=True)
        mem("interleave", need, f"widths={','.join(map(str, widths))}")
        return

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
            ids = make_ids(n, nc, dev)
            if HOT_SLOTS:
                ids[:, :min(HOT_SLOTS, TOPK)] = -1
            if HOT_ROWS:
                ids[:min(HOT_ROWS, n)] = -1     # whole row hot: nothing for the cold gather
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
