"""Eager prefill-only TP2 all-reduce: peer-push sum in one kernel per call.

PREFILL_AR=0 disables it; `int8` pushes each rank's contribution compressed to int8 with one bf16 scale per 32
contiguous fp32 elements, `exact` pushes the fp32 payload. Both modes are bit-identical across ranks. Anything
that is not a prefill-shaped fp32 reduction (2-D, >= 256 rows, numel a multiple of 512 within the arena cap,
contiguous, eager) is delegated to the wrapped backend unchanged, as are all non-fp32 dtypes.
"""
import json
import os
import time

import torch

from exllamav3.ext import exllamav3_ext as ext


class PrefillAR:
    def __init__(self, inner, comm, mode):
        if mode not in ("int8", "exact"):
            raise ValueError("PREFILL_AR must be 0, int8, or exact")
        if comm.transport != "dev":
            raise RuntimeError("PREFILL_AR requires the device peer transport")
        self.inner, self.comm = inner, comm
        self.fmt = 0 if mode == "int8" else 1
        self.mode = mode
        cap = int(os.environ.get("PREFILL_AR_CAP", str(8192 * 5120)))
        self.cap = cap - cap % 512
        if self.cap <= 0:
            raise ValueError("PREFILL_AR_CAP must be at least 512 elements")
        occ = ext.prefill_ar_occupancy(self.fmt)
        if occ < 1:
            raise RuntimeError("prefill AR kernel cannot be resident")
        # Two CTAs per SM keeps residency headroom for concurrent kernels while giving
        # enough tile owners to cover the push/consume pipeline.
        self.grid = torch.cuda.get_device_properties(comm.device).multi_processor_count * min(occ, 2)
        self.mine = ext.prefill_ar_alloc(ext.prefill_ar_layout(self.cap, self.fmt)[4], comm.device.index)

        path = os.environ["P2P_RDV"]
        mine = os.path.join(path, f"prefill_ar{comm.rank}.json")
        peer = os.path.join(path, f"prefill_ar{comm.peer}.json")
        data = {"handle": ext.p2p_dev_arena_handle(self.mine), "mode": mode, "cap": self.cap, "grid": self.grid}
        with open(mine + ".tmp", "w") as f:
            json.dump(data, f)
        os.replace(mine + ".tmp", mine)
        deadline = time.monotonic() + 120
        while not os.path.exists(peer):
            if time.monotonic() > deadline:
                raise TimeoutError("prefill AR peer rendezvous timed out")
            time.sleep(0.01)
        with open(peer) as f:
            other = json.load(f)
        if any(other[k] != data[k] for k in ("mode", "cap", "grid")):
            raise RuntimeError("prefill AR rank configuration mismatch")
        self.peer = ext.p2p_dev_arena_open(other["handle"], comm.device.index, comm.peer)
        self.seq = 0
        self.event = torch.cuda.Event()
        self.stream = None

    def all_reduce(self, tensor, contribution=True):
        n = tensor.numel()
        eligible = (tensor.dim() > 1 and tensor.shape[-1] > 0 and
                    n // tensor.shape[-1] >= 256 and n <= self.cap and n % 512 == 0 and
                    tensor.is_contiguous() and tensor.dtype == torch.float32 and
                    tensor.data_ptr() % 16 == 0 and not torch.cuda.is_current_stream_capturing())
        if not eligible:
            return self.inner.all_reduce(tensor, contribution)
        if not contribution:
            tensor.zero_()
        stream = torch.cuda.current_stream(tensor.device)
        if self.stream is not None and stream != self.stream:
            stream.wait_event(self.event)
        if self.seq >= 0xffffffff:
            raise RuntimeError("prefill AR sequence exhausted; restart both ranks")
        self.seq += 1
        ext.prefill_ar_launch(tensor, tensor, self.peer, self.mine, n, self.cap, self.fmt, self.grid, self.seq,
                              self.comm.rank)
        self.event.record(stream)
        self.stream = stream

    def __getattr__(self, name):
        return getattr(self.inner, name)


def wrap_prefill_ar(backend, p2p):
    mode = os.environ.get("PREFILL_AR", "0")
    return backend if mode == "0" else PrefillAR(backend, p2p.comm, mode)
