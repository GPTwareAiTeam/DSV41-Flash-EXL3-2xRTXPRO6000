#!/usr/bin/env python3
"""Validate the streaming cuda-exl3 restack on a real cold layer, with the box under watch.

Run inside a GX container whose /src/exllamav3 is the tree this file lives in:

    SEAM_MOE_CX_MARGIN_MIB=0 python3 qmesh/cx_restack_probe.py --layer 0 --rows 64,4096

What it does, in order, printing a flushed line for each step so that a wedged box still leaves
evidence in the log:

  1. Samples /proc/meminfo (MemAvailable, Cached, SwapFree/Cached, Committed_AS) and this process's
     VmRSS/VmHWM every --every seconds from a thread that never blocks on CUDA.
  2. Loads the cold pack exactly as the server does (SeamServer.__init__), so the load's own
     per-layer MemAvailable lines are in the log too.
  3. Reports the layer's live cold bytes and the restack's peak (`exl3_moe_cuda.peak_mib`) against
     MemAvailable, then converts the layer only if `peak + --gate-mib <= MemAvailable`; otherwise it
     prints SEAM_CX_PROBE_NOFIT and stops - refusing costs nothing, a cudaMalloc that pages a 114 GiB
     pack out costs the box (GX1, 2026-11-01 16:44).
  4. If the conversion ran, issues one compute per row count through SeamServer.compute with
     synthetic ids inside the layer's cold set, and reports per-call time plus meminfo before/after.

Exit: 0 conversion measured, 4 refused (NOFIT), 5 the layer could not restack at all.
"""
import argparse
import ctypes
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qmesh import worker_seam_server as S  # noqa: E402


def memline():
    vals = {}
    try:
        for line in open("/proc/meminfo"):
            k, v = line.split(":", 1)
            vals[k] = int(v.split()[0]) // 1024
    except OSError:
        pass
    try:
        for line in open("/proc/self/status"):
            if line.startswith(("VmRSS:", "VmHWM:")):
                k, v = line.split(":", 1)
                vals[k.rstrip(":")] = int(v.split()[0]) // 1024
    except OSError:
        pass
    return (f"avail_mib={vals.get('MemAvailable', -1)} cached_mib={vals.get('Cached', -1)} "
            f"free_mib={vals.get('MemFree', -1)} swap_used_mib="
            f"{max(0, vals.get('SwapTotal', 0) - vals.get('SwapFree', 0))} "
            f"committed_mib={vals.get('Committed_AS', -1)} "
            f"rss_mib={vals.get('VmRSS', -1)} rss_peak_mib={vals.get('VmHWM', -1)}")


def sampler(every, stop):
    n = 0
    while not stop.is_set():
        n += 1
        print(f"SEAM_CX_PROBE_MEM t={time.strftime('%H:%M:%S')} n={n} {memline()}", flush=True)
        stop.wait(every)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("SEAM_MODEL", "/models/DeepSeek-V4.1-Flash-1017"))
    ap.add_argument("--placement", default=os.environ.get("SEAM_PLACEMENT", ""))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--rows", default="64,4096", help="comma-separated row counts to call")
    ap.add_argument("--every", type=float, default=3.0, help="seconds between memory samples")
    ap.add_argument("--gate-mib", type=int, default=128,
                    help="headroom the conversion must leave beyond its own peak")
    ap.add_argument("--calls", type=int, default=2, help="calls per row count")
    a = ap.parse_args(argv)

    placement = a.placement or os.path.join(a.model, "cold_placement.json")
    if not os.path.exists(placement):
        print(f"SEAM_CX_PROBE_FAIL no placement at {placement}", flush=True)
        return 2

    stop = threading.Event()
    th = threading.Thread(target=sampler, args=(a.every, stop), daemon=True)
    th.start()
    print(f"SEAM_CX_PROBE_START layer={a.layer} rows={a.rows} gate_mib={a.gate_mib} "
          f"cx_margin_mib={S.CX_MARGIN_MIB} {memline()}", flush=True)

    t0 = time.perf_counter()
    srv = S.SeamServer(a.model, placement, "cuda:0", ws_rows=0)
    print(f"SEAM_CX_PROBE_LOADED load_s={time.perf_counter() - t0:.0f} {memline()}", flush=True)

    from qmesh.exl3_moe_cuda import CudaExl3Moe, RestackBroken, _live_mib, transient_mib
    layer = a.layer
    ffn = srv.layers[layer][0]
    # With SEAM_MOE_PRELOAD=1 (the server default for cudaexl3) every layer was converted while it
    # loaded, which is the only moment the group transient fits; converting it a second time here
    # would restack a layer whose per-expert sources are already released.
    cx = srv._cx_layers.get(layer)
    if cx is not None:
        print(f"SEAM_CX_PROBE_PRELOADED layer={layer} groups={cx.describe()} "
              f"stacked_mib={cx.nbytes() / 2**20:.0f} {memline()}", flush=True)
    else:
        live = _live_mib(ffn)
        need = transient_mib(ffn)
        avail = S._avail_mib()
        print(f"SEAM_CX_PROBE_PLAN layer={layer} cold_mib={live:.0f} need_mib={need:.0f} "
              f"avail_mib={avail} gate_mib={a.gate_mib} "
              f"fits={int(avail >= 0 and need + a.gate_mib <= avail)} "
              f"{memline()}", flush=True)
        if avail >= 0 and need + a.gate_mib > avail:
            print(f"SEAM_CX_PROBE_NOFIT layer={layer} need_mib={need:.0f}+{a.gate_mib} > "
                  f"avail_mib={avail} - refusing the conversion (it would page the pack out)",
                  flush=True)
            stop.set()
            return 4
        t0 = time.perf_counter()
        try:
            cx = CudaExl3Moe(ffn, "cuda:0")
        except RestackBroken as e:
            print(f"SEAM_CX_PROBE_BROKEN layer={layer} err={e} {memline()}", flush=True)
            stop.set()
            return 5
        print(f"SEAM_CX_PROBE_RESTACK layer={layer} groups={cx.describe()} "
              f"stacked_mib={cx.nbytes() / 2**20:.0f} build_s={time.perf_counter() - t0:.1f} "
              f"{memline()}", flush=True)
        srv._cx_layers[layer] = cx

    cold = sorted(int(x) for x in (getattr(ffn, "_cold_experts", None) or []))
    if not cold:
        # The layer object does not remember its cold set; the ids only have to be in range and
        # routed, so take the first TOPK experts and give each row a uniform weight.
        cold = list(range(S.TOPK))
    rows_list = [int(r) for r in a.rows.split(",") if r.strip()]
    for rows in rows_list:
        rng = np.random.default_rng(1234 + rows)
        # The slab carries raw bf16 (worker_seam_server._forward_piece views it as uint16), so the
        # probe has to feed bf16-encoded words: an fp32 rounded down to its high half.
        x = (rng.standard_normal((rows, S.HIDDEN)).astype(np.float32)
             .view(np.uint32) >> 16).astype(np.uint16)
        ids = np.tile(np.array(cold[:S.TOPK], dtype=np.int32), (rows, 1))
        w = (np.full((rows, S.TOPK), 1.0 / S.TOPK, dtype=np.float32)
             .view(np.uint32) >> 16).astype(np.uint16)
        out = np.zeros((rows, S.HIDDEN), dtype=np.float32)
        for c in range(a.calls):
            t0 = time.perf_counter()
            srv.compute(layer, rows,
                        x.ctypes.data, ids.ctypes.data, w.ctypes.data, out.ctypes.data)
            dt = 1e3 * (time.perf_counter() - t0)
            fin = int(np.isfinite(out).all())
            print(f"SEAM_CX_PROBE_CALL layer={layer} rows={rows} call={c} ms={dt:.1f} "
                  f"finite={fin} absmax={float(np.abs(out).max()):.4f} {memline()}", flush=True)
    print(f"SEAM_CX_PROBE_DONE {memline()}", flush=True)
    stop.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
