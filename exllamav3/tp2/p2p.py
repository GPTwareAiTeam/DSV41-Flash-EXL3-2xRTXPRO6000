"""One-shot 2-rank collectives for TP=2 (one process per GPU, rank r on cuda:r).

Transport: a named /dev/shm arena mapped by both processes and pinned for each GPU (see parallel/p2p_oneshot.cuh).
Peer SM access is unavailable in this container, so exchange goes through host-visible memory. A collective is one
kernel launch on the current stream. The sequence counter lives in device memory, so captured CUDA graphs replay.
All ranks must issue the same collectives, in the same order, with the same sizes, on the same stream.

Setup (both processes, same name; rank 0 creates the arena, rank 1 opens it):

    comm = P2PComm(rank, world=2, shm_name="/tp2_p2p_<tag>")
"""

from __future__ import annotations

import os
import time

import torch

from ..ext import exllamav3_ext as ext

ARENA_MAXB = 256 * 1024  # bytes of payload per parity slot (must match P2P_MAXB in p2p_oneshot.cuh)
MAX_BLOCKS = 8  # must match P2P_MAXBLK

_RENDEZVOUS_TIMEOUT_S = 120.0


class P2PComm:
    def __init__(self, rank: int, world: int = 2, shm_name: str = "", nblk: int = 1, device: int | None = None):
        if world != 2:
            raise ValueError("P2PComm supports exactly 2 ranks")
        if rank not in (0, 1):
            raise ValueError("rank must be 0 or 1")
        if not shm_name.startswith("/"):
            raise ValueError("shm_name must be a POSIX shm name starting with '/'")
        if not 1 <= nblk <= MAX_BLOCKS:
            raise ValueError(f"nblk must be in [1, {MAX_BLOCKS}]")

        self.rank = rank
        self.peer = 1 - rank
        self.world = world
        self.nblk = nblk
        self.device = torch.device("cuda", device if device is not None else rank)
        torch.cuda.set_device(self.device)

        # Each rank owns an inbox arena (<shm_name>.<rank>) that the peer writes into. Rank r creates its own inbox
        # and opens the peer's. Both device pointers are host-mapped, so the kernel reads and writes them directly.
        self.local = ext.p2p_shm_arena(f"{shm_name}.{rank}", True)
        peer_name = f"{shm_name}.{self.peer}"
        deadline = time.monotonic() + _RENDEZVOUS_TIMEOUT_S
        while True:
            try:
                self.peer_arena = ext.p2p_shm_arena(peer_name, False)
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"P2PComm: peer arena {peer_name} never appeared")
                time.sleep(0.01)

    def all_reduce_(self, t: torch.Tensor) -> torch.Tensor:
        """In-place fp32 sum over both ranks. Bit-identical result on both ranks (rank0 + rank1, fixed order)."""
        self._check_device(t)
        ext.p2p_all_reduce_f32(t, self.rank, self.local, self.peer_arena, self.nblk)
        return t

    def all_gather(self, t: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """out <- [rank0 | rank1] (flattened along dim 0 concatenation of the two payloads). out.numel == 2*t.numel."""
        self._check_device(t)
        ext.p2p_all_gather_f32(t, out, self.rank, self.local, self.peer_arena, self.nblk)
        return out

    def bcast_i64(self, t: torch.Tensor, root: int, out: torch.Tensor | None = None) -> torch.Tensor:
        """Broadcast int64 tensor from root. If out is None, writes into a new tensor."""
        self._check_device(t)
        if out is None:
            out = torch.empty_like(t)
        ext.p2p_bcast_i64(t, out, self.rank, root, self.local, self.peer_arena, self.nblk)
        return out

    def barrier(self) -> None:
        """Synchronizes the two ranks through a 1-element all-reduce on the current stream."""
        t = torch.zeros(4, dtype=torch.float32, device=self.device)
        self.all_reduce_(t)

    def _check_device(self, t: torch.Tensor) -> None:
        if t.device != self.device:
            raise ValueError(f"tensor on {t.device}, comm bound to {self.device}")
