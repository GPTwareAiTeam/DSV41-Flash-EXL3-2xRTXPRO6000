"""Piecewise CUDA graph segments for the decode step.

A Segment wraps a position-free stretch of the decode forward. Tensor arguments are copied into static
device buffers; the first call after one eager warmup records a CUDA graph over the wrapped function, and
every later call copies the tensors in, replays, and returns the static outputs.

Non-tensor arguments (the per-step params dict) are passed through live on every call and are NOT baked
into the graph: the captured body may only read per-step state through static device buffers.

All segments share one graph memory pool so intermediate allocations of different segments reuse blocks.
"""
import torch

_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = torch.cuda.graph_pool_handle()
    return _POOL


class Segment:
    """fn(*args) -> tensor or tuple of tensors.

    carry: params keys whose tensor values flow between segments through the params dict (e.g. the mHC
    "hc_pre" collapse weights each site hands to the next). A carried tensor present on entry is a static
    input; the value the captured body leaves under that key is a static output written back into the live
    params after every replay (Python side effects of the body do not run on replay)."""

    def __init__(self, fn, carry = ()):
        self.fn = fn
        self.carry = tuple(carry)
        self.graph = None
        self.static_in = None
        self.static_out = None
        self.carry_in = {}
        self.carry_out = {}
        self.warm = False

    @staticmethod
    def _params(args):
        return next((a for a in args if isinstance(a, dict)), None)

    def __call__(self, *args):
        if self.graph is None:
            if not self.warm:
                # Eager warmup: first-use allocations (workspaces, handles, tensor caches) happen here,
                # outside the capture
                self.warm = True
                out = self.fn(*args)
                return tuple(out) if isinstance(out, (tuple, list)) else (out,)
            self._capture(args)
        tens = [a for a in args if isinstance(a, torch.Tensor)]
        for dst, src in zip(self.static_in, tens):
            dst.copy_(src, non_blocking = True)
        params = self._params(args)
        for k, dst in self.carry_in.items():
            dst.copy_(params[k], non_blocking = True)
        self.graph.replay()
        for k, t in self.carry_out.items():
            params[k] = t
        return self.static_out

    def _capture(self, args):
        self.static_in = tuple(a.clone() for a in args if isinstance(a, torch.Tensor))
        params = self._params(args)
        cap_params = dict(params) if params is not None else None
        if cap_params is not None:
            for k in self.carry:
                v = cap_params.get(k)
                if isinstance(v, torch.Tensor):
                    self.carry_in[k] = cap_params[k] = v.clone()
        # Non-tensor args are fixed for the capture (the params dict is replaced by the capture copy)
        self.layout = [
            cap_params if a is params and params is not None else (a if not isinstance(a, torch.Tensor) else None)
            for a in args
        ]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, pool = _pool(), stream = s, capture_error_mode = "thread_local"):
                out = self.fn(*self._interleave(self.static_in))
        torch.cuda.current_stream().wait_stream(s)
        self.graph = g
        self.static_out = tuple(out) if isinstance(out, (tuple, list)) else (out,)
        if cap_params is not None:
            for k in self.carry:
                v = cap_params.get(k)
                if isinstance(v, torch.Tensor) and v is not self.carry_in.get(k):
                    self.carry_out[k] = v

    def _interleave(self, tens):
        it = iter(tens)
        return [next(it) if slot is None else slot for slot in self.layout]
