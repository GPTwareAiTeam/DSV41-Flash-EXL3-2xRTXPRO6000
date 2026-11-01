"""P7-side expert host over the native RDMA seam (qmesh/native/qes.py).

Drop-in for RemoteExpertHost (same method surface used by block_sparse_mlp_cpu.py), selected
by REMOTE_TRANSPORT=seam. Requests are issued and collected on the caller's CUDA stream via
qes_issue / qes_collect, so the path is graph-capturable (no host sync, no socket).

One control connection per host (SEAM_PEER:SEAM_PORT) carries the session negotiation; the
server answers with the row buckets it can serve and the data port of each, so a long-lived
GX server serves run after run without a reload. The requested bucket list need not match
the server's: this class adopts whatever the server grants.

Env:
  SEAM_DEV       RDMA device on P7 (default rocep142s0f0)
  SEAM_GID       GID index (default 5)
  SEAM_PEER      GX1 address (default 10.0.100.2)
  SEAM_PORT      seam control port (default 47311)
  SEAM_BUCKETS   row buckets to *request* (default 1,6,64,256,512,1024,2048,4096); the
                 server's answer wins, so a superset request adapts to any server list
  SEAM_WAIT      1 = queue behind the server's active session instead of failing fast
  EXL3_SEAM_PROF 1 times issue -> collect per (bucket, rows) with CUDA events and dumps the per-layer
                 round trip at exit (wire + GX1 compute). Eager only: a captured graph records the
                 calls once at capture and then replays ops, so there is nothing to time per round.
"""
from __future__ import annotations

import atexit
import os
import statistics
import sys
import time

import torch

from qmesh.native import qes

HIDDEN = qes.HIDDEN
TOPK = qes.TOPK

_PROF = os.environ.get("EXL3_SEAM_PROF", "0") == "1"
# Decode batches only by default: prefill chunks would land in the same bucket keys
_PROF_MAX_ROWS = int(os.environ.get("EXL3_SEAM_PROF_ROWS", "6"))
_prof_rows = {}     # (bucket, n) -> [(host_ns_0, host_ns_1, ev0, ev1)]


def _prof_dump():
    if not _prof_rows:
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    print("[SEAMPROF] bucket rows    n   host_us   dev_ms  dev_us/row", flush = True)
    for (bucket, n), vals in sorted(_prof_rows.items()):
        hu = statistics.median((v[1] - v[0]) / 1e3 for v in vals)
        dv = statistics.median(v[2].elapsed_time(v[3]) for v in vals)
        print(f"[SEAMPROF] {bucket:6d} {n:5d} {len(vals):4d} {hu:9.3f} {dv:8.3f} "
              f"{dv * 1e3 / max(n, 1):10.2f}", flush = True)


if _PROF:
    atexit.register(_prof_dump)


class RemoteSeamHost:
    """Expert host over the seam. connect() negotiates a session with the GX server, which
    answers with the row buckets it can serve and one data port per bucket; a batch of n rows is
    padded to the smallest granted bucket >= n with id -1 and weight 0 (exact zeros on GX1), and
    batches above the largest bucket are issued in bucket-sized chunks. All work stays on the
    caller's stream.

    Prefill therefore splits on the client: a chunk of R rows costs floor(R / GX_MAX) round trips
    at the largest granted bucket plus one for the remainder, which is padded to the smallest
    granted bucket >= it (exact, no padding, when R is a multiple of that bucket). GX_MAX is not a
    client constant: the server advertises its buckets, the client asks for the standard set and
    adopts the intersection, so a server that can only afford 256 rows makes every prefill chunk
    go 16 calls deep instead of failing."""

    # Row buckets *requested* from the server. The answer is the server's own list restricted to
    # this request, in the server's order, so a request that is a superset of any plausible server
    # list costs nothing and adapts to the memory a given GX box can afford: 1 and 6 serve
    # decode/verify, 256..4096 are the prefill widths (powers of two), and a server configured with
    # 64 still narrows a 2..63-row verify to the 64-row bucket as before. The largest granted
    # bucket is the per-call row limit used for prefill splitting.
    BUCKETS = tuple(int(b) for b in os.environ.get("SEAM_BUCKETS", "1,6,64,256,512,1024,2048,4096")
                    .split(",") if b.strip())


    @classmethod
    def for_config(cls, infer_params):
        host = getattr(infer_params, "remote_expert_host", None)
        if host is None:
            raise RuntimeError("remote_expert_host is not configured")
        hosts = getattr(infer_params, "remote_expert_hosts", None)
        if hosts is None:
            hosts = {}
            infer_params.remote_expert_hosts = hosts
        key = ("seam", host.get("host"), host.get("port"))
        if key not in hosts:
            hosts[key] = cls(host.get("host") or os.environ.get("SEAM_PEER", "10.0.100.2"),
                             host.get("port") or int(os.environ.get("SEAM_PORT", "47311")))
        return hosts[key]

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.dev = os.environ.get("SEAM_DEV", "rocep142s0f0")
        self.gid = int(os.environ.get("SEAM_GID", "5"))
        self.ctl = None
        # Granted buckets, in the server's order; None until connect().
        self.buckets = None
        self.seams = {}
        # Force the issue/collect branch for every batch shape (no socket fallback).
        self.stream_min_rows = 10**9
        self.layer_index = {}
        # Micro-batches kept in flight per layer call (see submit_prefill). The server advertises
        # its payload slot count in the negotiation reply; SEAM_NS caps how many of those slots
        # this client uses, so one server with 4 slots can be measured at NS=1/2/4 without a
        # reload. Decode is a single piece and is unaffected.
        self.slots = 1
        self.ns = 1
        self._splits = set()
        # Persistent per-(device, bucket) staging for the wire tensors (see _stage). These are the
        # only device buffers qes_issue reads, and they are never freed, so the copies it enqueues
        # can never race an allocation that reuses the same block.
        self._stage_bufs = {}

    def connect(self):
        """Negotiates the session with the GX server (idempotent). Raises RuntimeError with the
        server's own reason when it is busy, incompatible, or serves none of the requested
        buckets; SEAM_WAIT=1 queues instead of failing."""
        if self.ctl is not None:
            return self
        want = os.environ.get("SEAM_WAIT", "0") == "1"
        ctl = qes.Control(self.host, self.port, buckets = self.BUCKETS, wait = want)
        self.buckets = ctl.buckets
        self.chunk_rows = self.buckets[-1]
        self.ctl = ctl
        self.slots = max(1, int(getattr(ctl, "slots", 1)))
        cap = int(os.environ.get("SEAM_NS", "0") or 0)
        self.ns = self.slots if cap <= 0 else max(1, min(self.slots, cap))
        print(f"SEAM_NEGOTIATED host={self.host} port={self.port} "
              f"requested={','.join(map(str, self.BUCKETS))} "
              f"accepted={','.join(map(str, ctl.buckets))} "
              f"ports={','.join(map(str, ctl.ports))} chunk_rows={self.chunk_rows} "
              f"server_slots={self.slots} ns={self.ns} "
              f"server_build={ctl.build} placement_sha={ctl.placement_sha} "
              f"proto={qes.PROTO_VERSION} layers={ctl.layers}", flush = True)
        return self

    def _seam(self, bucket: int, device_index: int):
        """Data handle for one granted bucket, created on first use. The server assigned this
        bucket's port in the negotiation reply, so no ordering with the other buckets matters."""
        key = (bucket, device_index)
        if key not in self.seams:
            ctl = self.connect().ctl
            port = ctl.ports[ctl.buckets.index(bucket)]
            print(f"SEAM_CONNECT host={self.host} port={port} rows={bucket} dev_idx={device_index} "
                  f"dev={self.dev} gid={self.gid} buckets={','.join(map(str, ctl.buckets))}",
                  flush = True)
            self.seams[key] = ctl.seam(self.dev, self.gid, bucket, device_index)
        return self.seams[key]

    def register_layer(self, key, *args, **kwargs):
        idx = int(key.split(".")[1])
        self.layer_index[key] = idx
        return idx

    def _stage(self, x, ids, w):
        """Copies one piece into this host's persistent per-bucket staging tensors.

        qes_issue only *enqueues* the device->host copies of x/ids/w, so the tensors it reads must
        still be alive when those copies run. Building them per call (`.to(dtype).contiguous()`)
        hands the allocator blocks it can reuse for the next layer's routing, and a block reused by
        a kernel queued *ahead* of our copy makes the wire carry another layer's picks -- measured
        2026-11-01 as layer 7 id 277 of 274 cold and layer 30 id 284 of 275, with the mirror copy
        reading the same bad value out of the caller's tensor. Staging per bucket, owned for the
        host's lifetime, means the seam never reads a block the allocator can hand out again (and a
        captured graph never keeps a pointer to a freed block).

        One staging set per bucket, reused across requests, is safe because the copies are
        stream-ordered: a later request's staging write runs after the earlier request's copy,
        however the slots overlap. Pad rows are reset to id -1 / weight 0 so they contribute exactly
        zero and no stale id can reach the client's range check."""
        n = x.shape[0]
        dev = x.device
        bucket = next(b for b in self.buckets if b >= n)
        key = (dev.index or 0, bucket)
        st = self._stage_bufs.get(key)
        if st is None:
            st = (torch.empty(bucket, HIDDEN, dtype = torch.bfloat16, device = dev),
                  torch.full((bucket, TOPK), -1, dtype = torch.int32, device = dev),
                  torch.zeros(bucket, TOPK, dtype = torch.bfloat16, device = dev))
            self._stage_bufs[key] = st
        xp, idp, wp = st
        xp[:n] = x
        idp[:n] = ids
        wp[:n] = w
        if n < bucket:
            idp[n:].fill_(-1)
            wp[n:].zero_()
        return bucket, xp, idp, wp

    def _issue_chunk(self, layer_idx, x, ids, w, n_cold = None):
        """x [n,H], ids [n,K], w [n,K], n <= buckets[-1]. Stages to the smallest granted bucket >= n
        (pad rows: id -1, weight 0 -> exact zero contribution) and returns (seam, out)."""
        n = x.shape[0]
        bucket, xp, idp, wp = self._stage(x, ids, w)
        dev = x.device
        seam = self._seam(bucket, dev.index or 0)
        stream = torch.cuda.current_stream(dev)
        prof = _PROF and n <= _PROF_MAX_ROWS and not torch.cuda.is_current_stream_capturing()
        if prof:
            ev0 = torch.cuda.Event(enable_timing = True)
            ev0.record(stream)
            t0 = time.perf_counter_ns()
        try:
            seam.issue(layer_idx, xp.data_ptr(), idp.data_ptr(), wp.data_ptr(),
                       stream.cuda_stream, n_cold or 0)
        except RuntimeError:
            # The transport refused this request (e.g. an id its own cold set cannot hold): the
            # proxy's "qes:" line names layer, slot, values and cold count. The stack says which
            # module/path submitted them, which is what the id-space hunt needs.
            if os.environ.get("SEAM_ID_TRACE"):
                import traceback
                print(f"SEAM_ID_TRACE layer={layer_idx} rows={n} bucket={bucket} n_cold={n_cold}",
                      file = sys.stderr, flush = True)
                traceback.print_stack()
            raise
        out = torch.empty(bucket, HIDDEN, dtype = torch.float32, device = dev)
        seam.collect(layer_idx, out.data_ptr(), stream.cuda_stream)
        if prof:
            t1 = time.perf_counter_ns()
            ev1 = torch.cuda.Event(enable_timing = True)
            ev1.record(stream)
            _prof_rows.setdefault((bucket, n), []).append((t0, t1, ev0, ev1))
        return out[:n]

    @staticmethod
    def _prep(y, selected, weights):
        """Views of the caller's routing output, cast and cast-safe-copied later by _stage:
        x [rows,H], ids [rows,K], w [rows,K]."""
        return (y.reshape(-1, HIDDEN), selected.reshape(-1, TOPK), weights.reshape(-1, TOPK))

    def _split(self, rows: int, step: int | None = None):
        """Prefill split of `rows` onto the granted buckets, chunked at `step` (default chunk_rows)
        with the remainder in the smallest granted bucket >= it. Returns [(start, stop, bucket)];
        the bucket is not a multiple of `step` when the remainder pads (id -1 rows cost the GX
        nothing but a slot of compute)."""
        step = step or self.chunk_rows
        out = [(i, min(i + step, rows), step) for i in range(0, rows - rows % step, step)]
        rem = rows % step
        if rem:
            out.append((rows - rem, rows, next(b for b in self.buckets if b >= rem)))
        return out

    def _collect_piece(self, layer_idx, piece, out, stream):
        """Collects one in-flight piece into `out` at its row offset. The collect's host->device
        copy and the issue's device->host copy are on the caller's stream, so this never waits on
        the host: the wait kernel for piece m simply sits after piece m+1's issue in the stream."""
        pos, n, bucket, _xp, _idp, _wp, seam = piece
        direct = pos + bucket <= out.shape[0]
        if direct:
            dst = out[pos:pos + bucket]
        else:
            # Padded tail piece: the seam fills a whole bucket, only `n` rows are real.
            dst = torch.empty(bucket, HIDDEN, dtype = torch.float32, device = out.device)
        seam.collect(layer_idx, dst.data_ptr(), stream.cuda_stream)
        if not direct:
            out[pos:pos + n] = dst[:n]

    def submit_prefill(self, layer_idx, y, selected, weights, n_cold = None):
        """Prefill one layer's chunk and return fp32 [rows, HIDDEN] on the caller's stream.

        The chunk is split into micro-batches at chunk_rows (the largest granted bucket), and up
        to `ns` of them are kept in flight at once: micro-batch m+1's request is on the wire (and
        its slot's compute already queued on GX) while the client is still waiting for m's
        response, so the transport and the GX expert compute overlap instead of alternating. All
        of it is stream-ordered (issue = device->host copy + publish, collect = wait kernel +
        host->device copy), so it stays graph-capturable and needs no host synchronisation: the
        waits for m and m+1 are two kernels in the same stream, in order.

        Causality is per micro-batch, not per layer: a micro-batch's layer L+1 call needs its own
        layer L output, which the pieces keep in plan order. Overlap between layers of the *same*
        tokens is still impossible (L+1 reads L's output).

        n_cold is this layer's cold-expert count, carried on every request so the client proxy can
        refuse an id the placement cannot hold (see qes.Seam.issue)."""
        self.connect()
        x, ids, w = self._prep(y, selected, weights)
        rows = x.shape[0]
        plan = self._split(rows)
        ns = max(1, min(self.ns, len(plan)))
        shape = (rows, ns, tuple(b for _, _, b in plan))
        if shape not in self._splits:
            self._splits.add(shape)
            widths = ",".join(f"{b}x{sum(1 for s, e, _ in plan if _ == b)}" for b in
                              dict.fromkeys(b for _, _, b in plan))
            print(f"SEAM_SPLIT layer={layer_idx} rows={rows} chunk_rows={self.chunk_rows} "
                  f"calls={len(plan)} ns={ns} server_slots={self.slots} "
                  f"buckets={','.join(map(str, self.buckets))} plan={widths}", flush = True)
        if len(plan) == 1:
            return self._issue_chunk(layer_idx, x, ids, w, n_cold)
        # One fp32 [rows,H] output that every micro-batch writes into: torch.cat would hold the
        # parts and the copy at once (+rows*H*4 bytes, 235 MiB at 8192 rows), which the prefill
        # memory budget does not have.
        out = torch.empty(rows, HIDDEN, dtype = torch.float32, device = x.device)
        stream = torch.cuda.current_stream(x.device)
        dev_index = x.device.index or 0
        inflight = []
        for s, e, _ in plan:
            if len(inflight) >= ns:
                self._collect_piece(layer_idx, inflight.pop(0), out, stream)
            bucket, xp, idp, wp = self._stage(x[s:e], ids[s:e], w[s:e])
            seam = self._seam(bucket, dev_index)
            seam.issue(layer_idx, xp.data_ptr(), idp.data_ptr(), wp.data_ptr(),
                       stream.cuda_stream, n_cold or 0)
            inflight.append((s, e - s, bucket, xp, idp, wp, seam))
        while inflight:
            self._collect_piece(layer_idx, inflight.pop(0), out, stream)
        return out

    def submit_issue(self, layer_idx, y, selected, weights, n_cold = None):
        """Decode-size batches (<= largest bucket) run as one chunk; the output is produced
        by the collect enqueued inside, so the caller's 'collect' is a no-op handle."""
        return self.submit_prefill(layer_idx, y, selected, weights, n_cold)

    def submit_collect(self, handle):
        return handle

    def submit_issue_fused(self, *args, **kwargs):
        raise NotImplementedError("seam path uses EXL3_MOE_SPLIT_FUSED=0")

    def unregister(self):
        pass

    def shutdown(self):
        for seam in self.seams.values():
            seam.close()
        self.seams.clear()
        if self.ctl is not None:
            self.ctl.close()
            self.ctl = None
            self.buckets = None
