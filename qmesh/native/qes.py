"""ctypes binding for libqmesh_expert_seam (see qmesh_expert_seam.h).

Client (P7): Seam.issue / Seam.collect enqueue on a torch CUDA stream (graph-capturable).
Server (GX1): Seam(role=SERVER, ...).set_server_fn(fn) installs a Python compute callback
that the C proxy thread calls with pinned-slab numpy views. The callback runs under the
GIL on the proxy thread, so it must do its CUDA work on its own stream and synchronize
before returning.
"""
from __future__ import annotations

import ctypes
import os

HIDDEN = 5120
TOPK = 6
NL = 40
ROLE_CLIENT = 0
ROLE_SERVER = 1

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin",
                         f"libqmesh_expert_seam.{os.uname().machine}.so")

_SERVER_FN = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)


def _lib():
    lib = ctypes.CDLL(_LIB_PATH)
    lib.qes_create.restype = ctypes.c_void_p
    lib.qes_create.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.qes_issue.restype = ctypes.c_int
    lib.qes_issue.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_void_p]
    lib.qes_collect.restype = ctypes.c_int
    lib.qes_collect.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    lib.qes_set_server_fn.restype = None
    lib.qes_set_server_fn.argtypes = [ctypes.c_void_p, _SERVER_FN, ctypes.c_void_p]
    lib.qes_destroy.restype = None
    lib.qes_destroy.argtypes = [ctypes.c_void_p]
    return lib


_LIB = None


def lib():
    global _LIB
    if _LIB is None:
        _LIB = _lib()
    return _LIB


class Seam:
    def __init__(self, role, dev_name, gid_idx, peer_ip, port, rows, device_ordinal):
        self.role = role
        self.rows = rows
        self._cb = None
        self._h = lib().qes_create(role, dev_name.encode(), gid_idx,
                                   peer_ip.encode() if peer_ip else None, port, rows, device_ordinal)
        if not self._h:
            raise RuntimeError("qes_create failed (rows must be 1..64, RDMA device must exist)")

    def set_server_fn(self, fn):
        """fn(layer, rows, x_bf16, ids_i32, w_bf16, out_f32): raw pinned-slab pointers
        wrapped as ctypes arrays; see worker_seam_server.py for the typed wrapper."""
        def tramp(ctx, layer, rows, x, ids, w, out):
            fn(layer, rows, x, ids, w, out)
        self._cb = _SERVER_FN(tramp)
        lib().qes_set_server_fn(self._h, self._cb, None)

    def issue(self, layer, x_ptr, ids_ptr, w_ptr, stream_ptr):
        rc = lib().qes_issue(self._h, layer, x_ptr, ids_ptr, w_ptr, stream_ptr)
        if rc:
            raise RuntimeError(f"qes_issue layer={layer} rc={rc}")

    def collect(self, layer, out_ptr, stream_ptr):
        rc = lib().qes_collect(self._h, layer, out_ptr, stream_ptr)
        if rc:
            raise RuntimeError(f"qes_collect layer={layer} rc={rc}")

    def close(self):
        if self._h:
            lib().qes_destroy(self._h)
            self._h = None
