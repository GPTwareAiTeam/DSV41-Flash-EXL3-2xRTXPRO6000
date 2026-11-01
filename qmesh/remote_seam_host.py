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
  SEAM_BUCKETS   row buckets to *request* (default 1,6,64); the server's answer wins
  SEAM_WAIT      1 = queue behind the server's active session instead of failing fast
  EXL3_SEAM_PROF 1 times issue -> collect per (bucket, rows) with CUDA events and dumps the per-layer
                 round trip at exit (wire + GX1 compute). Eager only: a captured graph records the
                 calls once at capture and then replays ops, so there is nothing to time per round.
"""
from __future__ import annotations

import atexit
import os
import statistics
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
    BUCKETS = tuple(int(b) for b in os.environ.get("SEAM_BUCKETS", "1,6,256,512,1024,2048,4096")
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
        print(f"SEAM_NEGOTIATED host={self.host} port={self.port} "
              f"requested={','.join(map(str, self.BUCKETS))} "
              f"accepted={','.join(map(str, ctl.buckets))} "
              f"ports={','.join(map(str, ctl.ports))} chunk_rows={self.chunk_rows} "
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

    def _issue_chunk(self, layer_idx, x, ids, w):
        """x bf16 [n,H], ids i32 [n,K], w bf16 [n,K], n <= buckets[-1]. Pads to the smallest
        granted bucket >= n (pad rows: id -1, weight 0 -> exact zero contribution). Returns
        (seam, out)."""
        n = x.shape[0]
        bucket = next(b for b in self.buckets if b >= n)
        dev = x.device
        if bucket != n:
            xp = torch.zeros(bucket, HIDDEN, dtype=torch.bfloat16, device=dev)
            idp = torch.full((bucket, TOPK), -1, dtype=torch.int32, device=dev)
            wp = torch.zeros(bucket, TOPK, dtype=torch.bfloat16, device=dev)
            xp[:n] = x
            idp[:n] = ids
            wp[:n] = w
            x, ids, w = xp, idp, wp
        seam = self._seam(bucket, dev.index or 0)
        stream = torch.cuda.current_stream(dev)
        prof = _PROF and n <= _PROF_MAX_ROWS and not torch.cuda.is_current_stream_capturing()
        if prof:
            ev0 = torch.cuda.Event(enable_timing = True)
            ev0.record(stream)
            t0 = time.perf_counter_ns()
        seam.issue(layer_idx, x.data_ptr(), ids.data_ptr(), w.data_ptr(), stream.cuda_stream)
        out = torch.empty(bucket, HIDDEN, dtype=torch.float32, device=dev)
        seam.collect(layer_idx, out.data_ptr(), stream.cuda_stream)
        if prof:
            t1 = time.perf_counter_ns()
            ev1 = torch.cuda.Event(enable_timing = True)
            ev1.record(stream)
            _prof_rows.setdefault((bucket, n), []).append((t0, t1, ev0, ev1))
        return out[:n]

    @staticmethod
    def _prep(y, selected, weights):
        """Pack to the seam's wire dtypes on device: x bf16 [rows,H], ids i32 [rows,K], w bf16 [rows,K]."""
        x = y.reshape(-1, HIDDEN).to(torch.bfloat16).contiguous()
        ids = selected.reshape(-1, TOPK).to(torch.int32).contiguous()
        w = weights.reshape(-1, TOPK).to(torch.bfloat16).contiguous()
        return x, ids, w

    def _split(self, rows: int):
        """Prefill split of `rows` onto the granted buckets: full chunks at chunk_rows, then the
        remainder in the smallest granted bucket >= it. Returns [(start, stop, bucket)]; the bucket
        is not a multiple of chunk_rows when the remainder pads (id -1 rows cost GX1 nothing but a
        slot of compute)."""
        step = self.chunk_rows
        out = [(i, min(i + step, rows), step) for i in range(0, rows - rows % step, step)]
        rem = rows % step
        if rem:
            out.append((rows - rem, rows, next(b for b in self.buckets if b >= rem)))
        return out

    def submit_prefill(self, layer_idx, y, selected, weights):
        """Synchronous on the stream: each chunk is issued and collected before the next is
        issued, so one layer's request slot is never rewritten while the GX1 server still reads
        it. Returns fp32 [rows, HIDDEN] on the caller's stream.

        Chunks are cut at chunk_rows (the largest granted bucket) with the remainder in the
        smallest granted bucket that holds it, so a 8192-row prefill piece is 8 calls at 1024
        plus nothing on GX boxes that can afford 1024 rows per call. This is independent of the
        P7 prefill chunk limiter: whatever the limiter hands over is split here."""
        self.connect()
        x, ids, w = self._prep(y, selected, weights)
        plan = self._split(x.shape[0])
        shape = (x.shape[0], len(plan), tuple(b for _, _, b in plan))
        if shape not in self._splits:
            self._splits.add(shape)
            widths = ",".join(f"{b}x{sum(1 for s, e, _ in plan if _ == b)}" for b in
                              dict.fromkeys(b for _, _, b in plan))
            print(f"SEAM_SPLIT layer={layer_idx} rows={x.shape[0]} chunk_rows={self.chunk_rows} "
                  f"calls={len(plan)} buckets={','.join(map(str, self.buckets))} plan={widths}",
                  flush = True)
        parts = [self._issue_chunk(layer_idx, x[s:e], ids[s:e], w[s:e]) for s, e, _ in plan]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim = 0)

    def submit_issue(self, layer_idx, y, selected, weights):
        """Decode-size batches (<= largest bucket) run as one chunk; the output is produced
        by the collect enqueued inside, so the caller's 'collect' is a no-op handle."""
        return self.submit_prefill(layer_idx, y, selected, weights)

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
