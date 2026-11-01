"""P7-side expert host over the native RDMA seam (qmesh/native/qes.py).

Drop-in for RemoteExpertHost (same method surface used by block_sparse_mlp_cpu.py), selected
by REMOTE_TRANSPORT=seam. Requests are issued and collected on the caller's CUDA stream via
qes_issue / qes_collect, so the path is graph-capturable (no host sync, no socket).

Env:
  SEAM_DEV       RDMA device on P7 (default rocep142s0f0)
  SEAM_GID       GID index (default 5)
  SEAM_PEER      GX1 address (default 10.0.100.2)
  SEAM_PORT      seam TCP/RDMA rendezvous port (default 47311)
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
    """Expert host over the seam. The GX1 server exposes one handle per row bucket (ports
    SEAM_PORT + bucket index); a batch of n rows is padded to the smallest bucket >= n with
    id -1 and weight 0 (exact zeros on GX1), and batches above the largest bucket are issued in
    bucket-sized chunks. All work stays on the caller's stream."""

    BUCKETS = (1, 6, 64)


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
        self.seams = {}
        # Force the issue/collect branch for every batch shape (no socket fallback).
        self.stream_min_rows = 10**9
        self.layer_index = {}

    def _seam(self, bucket: int, device_index: int):
        """Handles for a device are created together in BUCKETS order on first use: the GX1
        server accepts its handles sequentially in that same order, so a lazy per-bucket
        connect would reach a port the server has not reached yet."""
        key = (bucket, device_index)
        if key not in self.seams:
            for b in self.BUCKETS:
                k = (b, device_index)
                if k not in self.seams:
                    port = self.port + self.BUCKETS.index(b)
                    self.seams[k] = qes.Seam(qes.ROLE_CLIENT, self.dev, self.gid, self.host,
                                             port, b, device_index)
        return self.seams[key]

    def register_layer(self, key, *args, **kwargs):
        idx = int(key.split(".")[1])
        self.layer_index[key] = idx
        return idx

    def _issue_chunk(self, layer_idx, x, ids, w):
        """x bf16 [n,H], ids i32 [n,K], w bf16 [n,K], n <= BUCKETS[-1]. Pads to the smallest
        bucket >= n (pad rows: id -1, weight 0 -> exact zero contribution). Returns (seam, out)."""
        n = x.shape[0]
        bucket = next(b for b in self.BUCKETS if b >= n)
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

    def submit_prefill(self, layer_idx, y, selected, weights):
        """Synchronous on the stream: each chunk is issued and collected before the next is
        issued, so one layer's request slot is never rewritten while the GX1 server still reads
        it. Returns fp32 [rows, HIDDEN] on the caller's stream."""
        x, ids, w = self._prep(y, selected, weights)
        step = self.BUCKETS[-1]
        parts = [self._issue_chunk(layer_idx, x[i:i + step], ids[i:i + step], w[i:i + step])
                 for i in range(0, x.shape[0], step)]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

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
