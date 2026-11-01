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
    def __init__(self, model_dir, placement_path, device, stub = False, fail_after = 0):
        # fail_after: test hook for the fail-stop in compute() - raise on call N+1. 0 = never.
        self.fail_after = int(fail_after or 0)
        self.seen = 0
        placement = json.load(open(placement_path))
        self.device = device
        self.stub = stub
        self.stream = None
        self.layers = {}
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

        Fail-stop, not fail-soft. The C proxy posts the response only *after* this hook returns,
        so an exception that escapes here is swallowed by ctypes (it prints "Exception ignored")
        and the previous call's bytes in the pinned response region are posted as this call's
        answer. Any failure therefore kills the process: the RC QP goes away, the client's proxy
        sees the CQ error or a 30 s timeout, marks the handle dead and raises - an error, never
        stale numbers. A GX that cannot compute must not answer.
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
        with _lock:
            _times.setdefault(int(rows), []).append((time.perf_counter_ns() - t0) / 1e3)

    def _forward(self, layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr):
        if self.stub:
            view(out_ptr, rows * HIDDEN * 4, np.float32).reshape(rows, HIDDEN)[:] = 0.0
            return
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


def _avail_mib() -> int:
    """Host MemAvailable, MiB: what is left on this box for the slabs (see seam_auto_buckets)."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


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
                         "bucket is the per-call row limit the client splits prefill on, so size it "
                         "to the GX box's memory headroom: protocol v3 staging is ~30.2 KiB pinned "
                         "per row per slot, so 1,6,64,1024,2048 at one slot is 38 MiB and 4 slots "
                         "of 2048 rows is 242 MiB, which fits a 121 GiB GB10 with the pack loaded.")
    ap.add_argument("--slab-slots", type = int,
                    default = int(os.environ.get("SEAM_SLAB_SLOTS", "1")),
                    help="payload slots per bucket (1..8): how many requests one client handle may "
                         "keep in flight. NS>1 lets a client pipeline prefill micro-batches so the "
                         "wire transfer of slot k+1 overlaps the GX compute of slot k. Costs "
                         "NS x rows x 30.2 KiB of pinned memory per bucket.")
    ap.add_argument("--wait", action="store_true",
                    help="queue a second client behind the active session instead of rejecting it "
                         "(also settable per client with SEAM_WAIT=1)")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    t0 = time.time()
    server = SeamServer(args.model_dir, args.placement, device, stub = args.stub,
                        fail_after = args.fail_after)
    avail_mib, swapped_mib = _avail_mib(), _swap_used_mib()
    # build= is whatever release started this process (SEAM_BUILD_ID from the pair release system).
    # The v2 wire has no status verb, so this line is how a release check learns which build the
    # *running* server came from; "-" means the release env was not passed.
    build_id = os.environ.get("SEAM_BUILD_ID", "-") or "-"
    print(f"SEAM_SERVER_LOADED {time.time() - t0:.1f}s stub={int(args.stub)} "
          f"fail_after={args.fail_after} build={build_id} "
          f"avail_mib={avail_mib} swapped_mib={swapped_mib}", flush=True)
    # What is left after the pack is the crash budget for every forward's transient buffers.
    if swapped_mib > SWAP_WARN_MIB:
        print(f"SEAM_SERVER_SWAP_WARN swapped_mib={swapped_mib} avail_mib={avail_mib}: the pack is "
              f"paging; expect multi-second outliers or launch failures", flush=True)
    threading.Thread(target = _reporter, daemon = True).start()

    # One pinned slab per row bucket, registered once; each client session gets fresh QPs on
    # them and the server answers the control hello with the bucket -> port mapping. The slabs are
    # allocated here, after the pack load, so the list has to have been sized to the box's free
    # memory beforehand (seam_server_ensure.sh --buckets auto does that): print the bill first.
    buckets = sorted({int(r) for r in args.buckets.split(",") if r.strip()})
    want_mib = slab_mib(buckets, args.slab_slots)
    avail_mib = _avail_mib()
    fits = int(avail_mib < 0 or want_mib <= avail_mib - 64)
    print(f"SEAM_SERVER_SLAB buckets={','.join(map(str, buckets))} slots={args.slab_slots} "
          f"predicted_mib={want_mib} rows_max={buckets[-1]} "
          f"avail_mib={avail_mib} fits={fits}", flush = True)
    if not fits:
        # The pack is already resident, so a slab that does not fit is paid by swapping the pack
        # out page by page: every expert call then faults, and the box looks alive while crawling.
        print(f"SEAM_SERVER_SLAB_WARN buckets=... predicted_mib={want_mib} avail_mib={avail_mib} "
              f"over_mib={want_mib - avail_mib + 64}: lower --buckets or expect the pack to be "
              f"swapped out (see seam_server_ensure.sh --buckets auto)", flush = True)
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
