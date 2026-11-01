#!/usr/bin/env python3
"""GX1 expert server for the native P7<->GX1 seam (qmesh/native/qes.py).

One fused mixed-K BlockSparseMLP per layer holds that layer's cold experts (the
placement's `warm` list, in order). A seam request carries cold-local ids [rows,6]
(-1 = expert resident on P7) and routing weights [rows,6]. The compute hook runs on the
seam's proxy thread: it copies the request out of the pinned slab, runs the module with
`external_selected`/`external_weights` on a dedicated CUDA stream, and writes fp32 output
back into the pinned response region.

The experts are loaded once and the server then serves client sessions back to back: the
control port (--port) answers a versioned hello with the row buckets it can serve and the
data port of each, and a session lives as long as its control connection. Client exit
(clean, ctrl-c, kill -9) tears down that session's QPs only, so benchmark arms can run one
after another - or one per pair - without reloading the cold pack.

-1 picks are mapped to local expert 0 with weight 0, so they contribute exact zeros
(the module's own range check stays satisfied).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import threading
import time

import numpy as np
import torch

from qmesh.native import qes

# exllamav3 is imported inside SeamServer.__init__: its module import JIT-builds exllamav3_ext
# (minutes on a cold cache, and it blocks on the torch extension baton), which a --stub server
# must never pay for - the wire/protocol baseline has to come up on a box whose GPU is busy.

HIDDEN = qes.HIDDEN
TOPK = qes.TOPK
NL = qes.NL

# Swap in use above this and the resident pack is paging. A few hundred MiB is noise on these
# boxes; GiBs mean the box is over its head before the first request.
SWAP_WARN_MIB = 512


def _meminfo_kib(key: str) -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith(key + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


def _avail_mib() -> int:
    kib = _meminfo_kib("MemAvailable")
    return -1 if kib < 0 else kib // 1024


def _swap_used_mib() -> int:
    total, free = _meminfo_kib("SwapTotal"), _meminfo_kib("SwapFree")
    if total < 0 or free < 0:
        return -1
    return (total - free) // 1024


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
    def __init__(self, model_dir, placement_path, device, stub = False, ws_rows = 0):
        # ws_rows: serve each request in pieces of at most this many rows, so the peak device
        # workspace follows ws_rows instead of the row bucket (see _forward). 0 = one piece.
        placement = json.load(open(placement_path))
        self.device = device
        self.stub = stub
        self.stream = None
        self.layers = {}
        self.ws_rows = int(ws_rows or 0)
        if stub:
            # Wire/protocol baseline: no experts are needed and nothing here touches CUDA or
            # exllamav3, so a stub server is up in seconds even while the GPU holds another pack.
            return
        from exllamav3 import Config, Model

        self.stream = torch.cuda.Stream(device=device)
        config = Config.from_directory(model_dir)
        model = Model.from_config(config)
        for layer in range(NL):
            cold = placement["layers"][str(layer)]["warm"]
            base = next(m for m in model.modules if m.key == f"layers.{layer}").mlp
            self.layers[layer] = build_layer(base, cold, device)
            del base
            torch.cuda.empty_cache()
            print(f"layer {layer}: {len(cold)} cold experts, "
                  f"{torch.cuda.memory_allocated(device) / 2**30:.2f} GiB", flush=True)

    def compute(self, layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr):
        """Pinned-slab pointers in; fp32 output written into the pinned response region.

        Fail-stop, not fail-soft: the C proxy posts the response only after this hook returns, so
        an exception allowed to escape here is swallowed by ctypes (it prints "Exception ignored")
        and the *previous* call's bytes in the pinned response region get posted as this call's
        answer. That is exactly what happened on GX2 on 2026-11-01 08:57: a CUDA launch failure
        inside _forward_piece left the context permanently broken, 170k callbacks raised, the
        server kept accepting sessions for 20 min and the client decoded garbage ("magnetic
        magnetic ..."). So any failure here kills the process: the RC QP goes away, the client's
        proxy sees the CQ error, marks the handle dead and raises - an error, never stale numbers.
        """
        # Per-call device time, reported every minute as SEAM_COMPUTE (median/mean/max us per row
        # bucket). Paired with the client's round-trip timing this splits the seam into GX1 compute
        # and wire: wire = round trip - compute.
        t0 = time.perf_counter_ns()
        self.seen += 1
        try:
            if self.fail_after and self.seen > self.fail_after:
                raise RuntimeError(f"injected compute failure on call {self.seen} "
                                   f"(--fail-after {self.fail_after})")
            self._forward(layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
        except Exception as e:  # noqa: BLE001 - see the docstring: no safe local recovery
            print(f"SEAM_COMPUTE_FAIL layer={layer} rows={rows} err={type(e).__name__}: {e} "
                  f"avail_mib={_avail_mib()} swapped_mib={_swap_used_mib()} "
                  f"calls={sum(len(v) for v in _times.values())} "
                  f"- exiting: a server that cannot compute must not answer", flush=True)
            sys.stderr.flush()
            os._exit(70)
        else:
            with _lock:
                _times.setdefault(int(rows), []).append((time.perf_counter_ns() - t0) / 1e3)

    def _forward(self, layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr):
        """Serve one request of `rows` rows, in internal pieces of at most ws_rows.

        The row bucket and the device workspace are separate bills and this is what decouples them.
        Measured on the dev pair (--ws-probe, pack resident): the cold-expert forward's scratch is
        ~150 KiB per row, so a bucket of N rows costs 150*N KiB of transient device memory however
        the slab is sized - 301 MiB at 2048 rows, 602 at 4096, 1202 at 8192, against 733 MiB of
        headroom on GX2 and 876 on GX1. The scratch scales with the rows handed to the module, not
        with the number of requests, so serving a big bucket as several ws_rows pieces makes the
        peak depend on ws_rows instead. Pieces run in order on the same CUDA stream and the single
        synchronise at the end still covers them; the pinned slab and the wire cost are unchanged,
        which is the point: a bigger bucket then buys fewer per-layer calls (remote_seam_host
        chunks a prefill at the largest granted bucket) without a bigger workspace.
        """
        if self.stub:
            view(out_ptr, rows * HIDDEN * 4, np.float32).reshape(rows, HIDDEN)[:] = 0.0
            return
        ffn, _ = self.layers[layer]
        step = self.ws_rows if 0 < self.ws_rows < rows else rows
        for r0 in range(0, rows, step):
            self._forward_piece(ffn, min(step, rows - r0),
                                x_ptr + r0 * HIDDEN * 2, ids_ptr + r0 * TOPK * 4,
                                w_ptr + r0 * TOPK * 2, out_ptr + r0 * HIDDEN * 4)

    def _forward_piece(self, ffn, rows, x_ptr, ids_ptr, w_ptr, out_ptr):
        """One mixed-K forward of `rows` rows straight out of, and back into, the pinned slab."""
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


_times: dict = {}
_lock = threading.Lock()


def _reporter():
    while True:
        time.sleep(60)
        with _lock:
            if not _times:
                continue
            snap = {k: v for k, v in _times.items()}
            _times.clear()
        for rows in sorted(snap):
            v = sorted(snap[rows])
            n = len(v)
            print(f"SEAM_COMPUTE rows={rows} n={n} p50_us={v[n // 2]:.1f} "
                  f"mean_us={sum(v) / n:.1f} max_us={v[-1]:.1f}", flush=True)


def view(addr, nbytes, dtype):
    """numpy view over raw pinned-slab memory (no copy)."""
    import ctypes
    buf = (ctypes.c_byte * nbytes).from_address(addr)
    return np.frombuffer(buf, dtype=dtype)


def slab_mib(buckets, slots = 1) -> float:
    """Pinned slab this server will allocate for `buckets`, MiB, one side.

    Same geometry as geometry() in qmesh/native/qmesh_expert_seam.cu (protocol v3): one slab per
    row bucket holding `slots` payload slots, each
    [64 B F_REQ][64 B F_RESP][64 B request header] + align64(rows*(HIDDEN*2 + TOPK*6)) request +
    align64(rows*HIDDEN*4) response bytes. Slots are in-flight requests, not layers: the layer id
    rides in the request header, so the whole staging cost is ~30.2 KiB per row per slot."""
    al = lambda v: (v + 63) // 64 * 64  # noqa: E731
    total = 0
    for rows in buckets:
        req = rows * (HIDDEN * 2 + TOPK * 4 + TOPK * 2)
        resp = rows * HIDDEN * 4
        total += (3 * 64 + al(req) + al(resp)) * slots
    return total / 2**20


# Measured, not predicted: --ws-probe on the dev pair with the 113.65 GiB pack resident gave
# 301 MiB of transient device memory at 2048 rows, 602 at 4096 and 1202 at 8192, i.e. 147 KiB per
# row, and the slab is on top of that (slab_mib). A request served in ws_rows pieces pays this for
# ws_rows, not for the bucket, so a big bucket is a wire/call-count decision, not a workspace one.
WS_KIB_PER_ROW = 147

# Swap in use above this and the resident pack is paging; see _swap_used_mib and
# SEAM_SERVER_SWAP_WARN in main(). A few hundred MiB is normal noise on these boxes, a few GiB is
# a box that is already over its head before the first request.
SWAP_WARN_MIB = 512


def ws_mib_for(rows: int, ws_rows: int = 0) -> float:
    """Transient device workspace the cold-expert forward needs for a request, MiB."""
    n = ws_rows if 0 < ws_rows < rows else rows
    return n * WS_KIB_PER_ROW / 1024


def _avail_mib() -> int:
    """Host MemAvailable, MiB: what is left on this box for the slabs (see seam_auto_buckets)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


def _swap_used_mib() -> int:
    """Swap currently in use, MiB. On a GX this is the early warning that matters: the pack is
    113.5 GiB of a 121.6 GiB unified pool, so a few GiB swapped out means every expert call takes
    page faults (seen as 4 s outliers in SEAM_COMPUTE) and a launch can fail outright."""
    total = free = 0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("SwapTotal:"):
                total = int(line.split()[1])
            elif line.startswith("SwapFree:"):
                free = int(line.split()[1])
    except OSError:
        return -1
    return (total - free) // 1024


def ws_probe(server, spec, min_avail_mib):
    """Measure the GX-side workspace of the real cold-expert forward, one row count at a time.

    The slab (slab_mib) is only the pinned staging bill. What a bigger row bucket really costs on
    the GX is the device memory today's mixed-K forward allocates for that many rows, and that has
    to be measured on the box with the pack already resident, not predicted: this runs the same
    _forward the server runs, at each row count, and prints the peak and the MemAvailable around
    it. An OOM or a nearly empty box is reported as such and the ladder continues, so one run maps
    the whole 2048/4096/8192 decision.
    """
    if server.stub:
        print("SEAM_WS stub=1: nothing to measure without the cold experts", flush = True)
        return
    dev = server.device
    ncold = server.layers[0][1]
    for rows in [int(x) for x in spec.split(",") if x.strip()]:
        avail0 = _avail_mib()
        if 0 <= avail0 < min_avail_mib:
            print(f"SEAM_WS rows={rows} SKIP avail_mib={avail0} min_avail_mib={min_avail_mib}: "
                  f"the pack is resident and the box has no room left to measure in", flush = True)
            continue
        lane = rows * (HIDDEN * 2 + TOPK * 4 + TOPK * 2) + rows * HIDDEN * 4
        stage = torch.zeros(lane, dtype = torch.uint8, pin_memory = True)
        base = stage.data_ptr()
        x_ptr = base
        ids_ptr = base + rows * HIDDEN * 2
        w_ptr = ids_ptr + rows * TOPK * 4
        out_ptr = w_ptr + rows * TOPK * 2
        # In-range expert ids (the module only has `ncold` experts) and a finite 1.0 in bf16, so the
        # probe exercises the same routing/fused path a real chunk does.
        view(x_ptr, rows * HIDDEN * 2, np.uint16).reshape(rows, HIDDEN)[:] = 0x3c00
        view(ids_ptr, rows * TOPK * 4, np.int32).reshape(rows, TOPK)[:] = \
            np.arange(rows * TOPK, dtype = np.int32).reshape(rows, TOPK) % max(ncold, 1)
        view(w_ptr, rows * TOPK * 2, np.uint16).reshape(rows, TOPK)[:] = 0x3c00
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
        t0 = time.perf_counter_ns()
        try:
            server._forward(0, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
            torch.cuda.synchronize(dev)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            print(f"SEAM_WS rows={rows} OOM avail_mib={avail0} err={str(e).splitlines()[0][:200]}",
                  flush = True)
            del stage
            torch.cuda.empty_cache()
            continue
        us = (time.perf_counter_ns() - t0) / 1e3
        print(f"SEAM_WS rows={rows} alloc_mib={torch.cuda.memory_allocated(dev) / 2**20:.1f} "
              f"reserved_mib={torch.cuda.memory_reserved(dev) / 2**20:.1f} "
              f"peak_alloc_mib={torch.cuda.max_memory_allocated(dev) / 2**20:.1f} "
              f"peak_reserved_mib={torch.cuda.max_memory_reserved(dev) / 2**20:.1f} "
              f"avail_before_mib={avail0} avail_after_mib={_avail_mib()} us={us:.0f} "
              f"us_per_row={us / max(rows, 1):.2f} fin={bool(np.isfinite(view(out_ptr, rows * HIDDEN * 4, np.float32)).all())}",
              flush = True)
        del stage
        torch.cuda.empty_cache()


def ws_check(server, rows):
    """Does serving a request in ws_rows pieces compute the same thing as one piece?

    Capping the workspace by chunking is only worth having if it is numerically harmless, and the
    answer has to be measured: the mixed-K path pads each expert's token run and the accumulation
    order inside a piece can differ from the order in a whole-bucket call. This runs the same
    deterministic request twice - one piece, then ws_rows pieces - and prints the largest absolute
    difference in the fp32 response (SEAM_WS_CHECK). `rows` must be a multiple of ws_rows.
    """
    if server.stub:
        print("SEAM_WS_CHECK stub=1: nothing to compare without the cold experts", flush = True)
        return
    step = server.ws_rows
    if not 0 < step < rows:
        print(f"SEAM_WS_CHECK rows={rows} skipped: needs 0 < --ws-rows < rows", flush = True)
        return
    ncold = next(iter(server.layers.values()))[1]
    lane = rows * (HIDDEN * 2 + TOPK * 4 + TOPK * 2) + rows * HIDDEN * 4
    stage = torch.zeros(lane, dtype = torch.uint8, pin_memory = True)
    base = stage.data_ptr()
    x_ptr, ids_ptr = base, base + rows * HIDDEN * 2
    w_ptr, out_ptr = ids_ptr + rows * TOPK * 4, ids_ptr + rows * TOPK * 4 + rows * TOPK * 2
    # A deterministic ramp in [0,1), not zeros: identical zeros would hide a piece that came back
    # empty. bf16 is the top half of an fp32, so the flat uint16 view takes the high 16 bits.
    ramp = (np.arange(rows * HIDDEN, dtype = np.float32) % 97 / 97).view(np.uint32) >> 16
    view(x_ptr, rows * HIDDEN * 2, np.uint16)[:] = ramp.astype(np.uint16)
    view(ids_ptr, rows * TOPK * 4, np.int32).reshape(rows, TOPK)[:] = \
        np.arange(rows * TOPK, dtype = np.int32).reshape(rows, TOPK) % max(ncold, 1)
    view(w_ptr, rows * TOPK * 2, np.uint16).reshape(rows, TOPK)[:] = 0x3f00   # 0.5 in bf16

    def run():
        server._forward(0, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
        torch.cuda.synchronize(server.device)
        return view(out_ptr, rows * HIDDEN * 4, np.float32).reshape(rows, HIDDEN).copy()

    whole = run()
    server.ws_rows = 0
    piece = run()
    server.ws_rows = step
    scale = float(np.abs(whole).max()) or 1.0
    print(f"SEAM_WS_CHECK rows={rows} ws_rows={step} pieces={rows // step} "
          f"max_abs={float(np.abs(whole - piece).max()):.6g} rel={float(np.abs(whole - piece).max()) / scale:.3g} "
          f"ref_absmax={scale:.6g} identical={bool(np.array_equal(whole, piece))}", flush = True)
    del stage
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/cold")
    ap.add_argument("--placement", default="/cold/placement.exl3.json")
    ap.add_argument("--dev", default="rocep1s0f0")
    ap.add_argument("--gid", type=int, default=3)
    ap.add_argument("--port", type=int, default=47311)
    ap.add_argument("--stub", action="store_true",
                    help="answer every request with zeros (wire/protocol baseline, no expert compute)")
    ap.add_argument("--buckets", default=os.environ.get("SERVER_BUCKETS", "1,6,256"),
                    help="row buckets this server can serve (ascending, one port per bucket from "
                         "--port+1). Clients request a subset; the superset default means one "
                         "long-lived server serves every benchmark arm without a reload. The largest "
                         "bucket is the per-call row limit the client splits prefill on, so it is "
                         "paid twice: protocol v3 staging is ~30.2 KiB pinned per row per slot, and "
                         "the cold-expert forward needs ~147 KiB per row of transient device memory "
                         "(measured: 301 MiB at 2048 rows, 602 at 4096, 1202 at 8192). With the "
                         "113.65 GiB pack resident a GB10 has 733-876 MiB left, so 2048 rows is the "
                         "largest bucket that fits with a margin; a bigger bucket needs --ws-rows to "
                         "cap the workspace, and the fit is printed as ws_mib in SEAM_SERVER_SLAB.")
    ap.add_argument("--slab-slots", type = int,
                    default = int(os.environ.get("SEAM_SLAB_SLOTS", "1")),
                    help="payload slots per bucket (1..8): how many requests one client handle may "
                         "keep in flight. NS>1 lets a client pipeline prefill micro-batches so the "
                         "wire transfer of slot k+1 overlaps the GX compute of slot k. Costs "
                         "NS x rows x 30.2 KiB of pinned memory per bucket.")
    ap.add_argument("--wait", action="store_true",
                    help="queue a second client behind the active session instead of rejecting it "
                         "(also settable per client with SEAM_WAIT=1)")
    ap.add_argument("--ws-probe", default = "",
                    help = "measure the GX workspace of the real forward at these row counts "
                           "(comma separated, ascending) and exit without listening: prints one "
                           "SEAM_WS line per row count with peak memory and MemAvailable")
    ap.add_argument("--ws-probe-min-avail", type = int, default = 96,
                    help = "skip a --ws-probe row count when MemAvailable is below this many MiB "
                           "(a full pack plus a nearly full box would swap, not fail, and the "
                           "number would be meaningless)")
    ap.add_argument("--ws-rows", type = int, default = int(os.environ.get("SEAM_WS_ROWS", "0")),
                    help = "serve each request in pieces of at most this many rows, so the peak "
                           "device workspace follows this instead of the row bucket (measured "
                           "~150 KiB per row of scratch in the cold-expert forward: 301 MiB at "
                           "2048 rows, 602 at 4096, 1202 at 8192). 0 = one piece per request, "
                           "which is what a bucket sized to the box's headroom needs. Set this "
                           "when the bucket is bigger than the box's workspace, e.g. 2048 rows "
                           "on a 733 MiB box keeps the peak near 150 MiB.")
    ap.add_argument("--ws-check", type = int, default = 0, metavar = "ROWS",
                    help = "run the same ROWS-row request once in one piece and once in --ws-rows "
                           "pieces and print the largest difference (SEAM_WS_CHECK), then exit: "
                           "proves that capping the workspace by chunking does not change the "
                           "numbers. ROWS must be a multiple of --ws-rows.")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    t0 = time.time()
    server = SeamServer(args.model_dir, args.placement, device, stub = args.stub,
                        ws_rows = args.ws_rows)
    print(f"SEAM_SERVER_LOADED {time.time() - t0:.1f}s stub={int(args.stub)}", flush=True)
    if args.ws_probe or args.ws_check:
        # Both modes share one pack load: the ladder first (it resets the peak counters), then the
        # equivalence check, whose own peak does not matter.
        if args.ws_probe:
            ws_probe(server, args.ws_probe, args.ws_probe_min_avail)
        if args.ws_check:
            ws_check(server, args.ws_check)
        return
    threading.Thread(target = _reporter, daemon = True).start()

    # One pinned slab per row bucket, registered once; each client session gets fresh QPs on
    # them and the server answers the control hello with the bucket -> port mapping. The slabs are
    # allocated here, after the pack load, so the list has to have been sized to the box's free
    # memory beforehand (seam_server_ensure.sh --buckets auto does that): print the bill first.
    buckets = sorted({int(r) for r in args.buckets.split(",") if r.strip()})
    want_mib = slab_mib(buckets, args.slab_slots)
    ws_mib = ws_mib_for(buckets[-1], args.ws_rows)
    avail_mib = _avail_mib()
    swapped_mib = _swap_used_mib()
    fits = int(avail_mib < 0 or want_mib + ws_mib <= avail_mib - 64)
    print(f"SEAM_SERVER_SLAB buckets={','.join(map(str, buckets))} slots={args.slab_slots} "
          f"slab_mib={want_mib} ws_rows={args.ws_rows} ws_mib={ws_mib} rows_max={buckets[-1]} "
          f"avail_mib={avail_mib} swapped_mib={swapped_mib} fits={fits}", flush = True)
    if not fits:
        # The pack is already resident, so memory that does not fit is paid by swapping the pack
        # out page by page: every expert call then faults, and the box looks alive while crawling.
        print(f"SEAM_SERVER_SLAB_WARN buckets=... slab_mib={want_mib} ws_mib={ws_mib} "
              f"avail_mib={avail_mib} over_mib={want_mib + ws_mib - avail_mib + 64}: lower "
              f"--buckets, set --ws-rows, or expect the pack to be swapped out (see "
              f"seam_server_ensure.sh --buckets auto)", flush = True)
    if swapped_mib > SWAP_WARN_MIB:
        # Not fatal to start (the pages fault back in), but every call pays for it and the box can
        # fail a launch outright: GX2 2026-11-01 ran with 4.4 GiB swapped, hit p50 47 ms / max
        # 4.15 s at rows=64 and then an "unspecified launch failure" in the decode arm.
        print(f"SEAM_SERVER_SWAP_WARN swapped_mib={swapped_mib} avail_mib={avail_mib}: the pack is "
              f"paging; expect multi-second SEAM_COMPUTE outliers and possible launch failures - "
              f"restart the box's server with a smaller --buckets/--slab-slots", flush = True)
    placement_sha = hashlib.sha256(open(args.placement, "rb").read()).hexdigest()[:12]
    srv = qes.Server(args.dev, args.gid, args.port, 0, buckets, build = qes.build_id(),
                     placement_sha = placement_sha, wait = args.wait, slots = args.slab_slots)
    srv.set_fn(lambda L, R, x, ids, w, o: server.compute(L, R, x, ids, w, o))
    try:
        srv.run()   # serves client sessions until SIGINT/SIGTERM
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        srv.close()


if __name__ == "__main__":
    main()
