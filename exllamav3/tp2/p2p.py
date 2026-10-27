"""One-shot 2-rank collectives for TP=2 (one process per GPU, rank r on cuda:r).

Transports (chosen once in setup, identical on both ranks):
  - "dev":  each rank's exchange arena is cudaMalloc'd on its GPU and exported with cudaIpc; the peer opens it and
            the kernel writes into it over P2P (SM peer access). This is the fast path.
  - "host": fallback when peer access is unavailable. Arena is a /dev/shm object, pinned and device-mapped.

A collective is one kernel launch on the current stream. Sequence counters live in the arena (device memory), so
captured CUDA graphs replay correctly. All ranks must issue the same collectives, in the same order, with the same
sizes, on the same stream.

Setup: P2PComm(rank, world=2, rendezvous_path=<dir>, nblk=1). rendezvous_path is a directory that must not already
contain rank files (use a fresh directory per run). Each rank writes <dir>/rank<r>.json with its transport and
handle, then waits for the peer's file. Both ranks can construct concurrently; no kernel runs during setup.

Diagnostics (opt-in, off by default; production kernels are the non-stats instantiation):
  P2PComm(..., stats=True) or env P2P_STATS=1 makes every collective record device-side stats into the local arena:
  per block, the number of acquire-spin iterations and globaltimer stamps (push / wait / combine), accumulated
  totals plus a ring of the last STATS_RING invocations. Read with comm.stats() (synchronizes the device) and
  write a rank-tagged JSON with comm.dump_stats(path_prefix) -> f"{prefix}.rank{rank}.json". Recording is plain
  device stores, so it is graph-capture safe; the ring is indexed by device-side seq, so replays are recorded too.
  Interpretation: a large wait_ns with spins > 0 means the block waited on the peer (peer lateness or the peer's
  transfer into our arena); small wait_ns with push_ns large means the cost is the local P2P write itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import torch

from ..ext import exllamav3_ext as ext

ARENA_MAXB = 256 * 1024  # bytes of payload per parity slot (must match P2P_MAXB in p2p_oneshot.cuh)
MAX_BLOCKS = 8  # must match P2P_MAXBLK
STATS_RING = 256  # must match P2P_STATS_RING
STATS_TOT_FIELDS = 5  # calls, spins, push_ns, wait_ns, total_ns (must match p2p_oneshot.cuh)
STATS_REC_FIELDS = 7  # seq, meta, spins, t_start, t_pushed, t_waited, t_end (must match P2PStatRec)
STATS_ENV = "P2P_STATS"  # "1" enables device stats on every P2PComm built in this process

_RENDEZVOUS_TIMEOUT_S = 120.0
_MODES = {0: "all_reduce", 1: "all_gather", 2: "bcast"}


class P2PComm:
    def __init__(self, rank: int, world: int = 2, rendezvous_path: str = "", nblk: int = MAX_BLOCKS,
                 device: int | None = None, transport: str = "auto", stats: bool | None = None):
        if world != 2:
            raise ValueError("P2PComm supports exactly 2 ranks")
        if rank not in (0, 1):
            raise ValueError("rank must be 0 or 1")
        if not rendezvous_path:
            raise ValueError("rendezvous_path must be set")
        if not 1 <= nblk <= MAX_BLOCKS:
            raise ValueError(f"nblk must be in [1, {MAX_BLOCKS}]")
        if transport not in ("auto", "dev", "host"):
            raise ValueError("transport must be 'auto', 'dev' or 'host'")

        self.rank = rank
        self.peer = 1 - rank
        self.world = world
        self.nblk = nblk
        self.device = torch.device("cuda", device if device is not None else rank)
        torch.cuda.set_device(self.device)
        self.stats_enabled = (os.environ.get(STATS_ENV, "0") == "1") if stats is None else bool(stats)

        peer_ok = torch.cuda.can_device_access_peer(rank, self.peer)
        if transport == "auto":
            mode = "dev" if peer_ok else "host"
        else:
            mode = transport
        if mode == "dev" and not peer_ok:
            raise RuntimeError(f"P2PComm: peer access {rank}->{self.peer} unavailable; use transport='host'")

        self.transport = mode
        self.local, self.peer_arena = self._rendezvous(rendezvous_path, mode)

    def _rendezvous(self, path: str, mode: str) -> tuple[int, int]:
        os.makedirs(path, exist_ok=True)
        mine = os.path.join(path, f"rank{self.rank}.json")
        theirs = os.path.join(path, f"rank{self.peer}.json")
        if os.path.exists(mine):
            raise RuntimeError(f"P2PComm: {mine} already exists; use a fresh rendezvous directory")

        if mode == "dev":
            local = ext.p2p_dev_arena_alloc(self.rank)
            payload = {"transport": "dev", "handle": ext.p2p_dev_arena_handle(local)}
        else:
            name = "/p2p_" + hashlib.sha1(os.path.abspath(path).encode()).hexdigest()[:16]
            local = ext.p2p_shm_arena(f"{name}.{self.rank}", True)
            payload = {"transport": "host", "handle": f"{name}.{self.rank}"}

        tmp = mine + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, mine)

        deadline = time.monotonic() + _RENDEZVOUS_TIMEOUT_S
        while not os.path.exists(theirs):
            if time.monotonic() > deadline:
                raise TimeoutError(f"P2PComm: peer rank file {theirs} never appeared")
            time.sleep(0.01)
        with open(theirs) as f:
            peer_payload = json.load(f)
        if peer_payload["transport"] != mode:
            raise RuntimeError(f"P2PComm: transport mismatch: local {mode}, peer {peer_payload['transport']}")

        if mode == "dev":
            peer_arena = ext.p2p_dev_arena_open(peer_payload["handle"], self.rank, self.peer)
        else:
            peer_arena = ext.p2p_shm_arena(peer_payload["handle"], False)
        return local, peer_arena

    def all_reduce_(self, t: torch.Tensor) -> torch.Tensor:
        """In-place fp32 sum over both ranks. Bit-identical result on both ranks (rank0 + rank1, fixed order)."""
        self._check_device(t)
        ext.p2p_all_reduce_f32(t, self.rank, self.local, self.peer_arena, self.nblk, self.stats_enabled)
        return t

    def all_gather(self, t: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """out <- [rank0 | rank1] (flattened along dim 0 concatenation of the two payloads). out.numel == 2*t.numel."""
        self._check_device(t)
        ext.p2p_all_gather_f32(t, out, self.rank, self.local, self.peer_arena, self.nblk, self.stats_enabled)
        return out

    def bcast_i64(self, t: torch.Tensor, root: int, out: torch.Tensor | None = None) -> torch.Tensor:
        """Broadcast int64 tensor from root. If out is None, writes into a new tensor."""
        self._check_device(t)
        if out is None:
            out = torch.empty_like(t)
        ext.p2p_bcast_i64(t, out, self.rank, root, self.local, self.peer_arena, self.nblk, self.stats_enabled)
        return out

    def barrier(self) -> None:
        """Synchronizes the two ranks through a 1-element all-reduce on the current stream."""
        t = torch.zeros(4, dtype=torch.float32, device=self.device)
        self.all_reduce_(t)

    def reset_stats(self) -> None:
        """Zeroes the device-side stats. Synchronizes the device."""
        ext.p2p_stats_reset(self.local, self.device.index)

    def stats(self) -> dict:
        """Snapshot of the device-side stats of this rank. Synchronizes the device; call after collectives drained.

        Returns {"rank", "transport", "nblk", "enabled", "blocks": [...]} where each block entry has:
          calls, spins, push_ns, wait_ns, total_ns  (running totals over the block's invocations)
          recent: last <= STATS_RING invocations of that block, oldest first, each
                  {seq, mode, grid, units, spins, push_ns, wait_ns, combine_ns, total_ns, t_start_ns}
        Timestamps are per-GPU globaltimer; durations are the meaningful cross-rank quantities.
        """
        raw = ext.p2p_stats_read(self.local, self.device.index)
        tot_n = MAX_BLOCKS * STATS_TOT_FIELDS
        totals, recs = raw[:tot_n], raw[tot_n:]
        blocks = []
        for b in range(self.nblk):
            t = totals[b * STATS_TOT_FIELDS:(b + 1) * STATS_TOT_FIELDS]
            calls, spins, push, wait, total = t
            base = b * STATS_RING * STATS_REC_FIELDS
            recent = []
            for slot in range(STATS_RING):
                r = recs[base + slot * STATS_REC_FIELDS: base + (slot + 1) * STATS_REC_FIELDS]
                seq, meta, rspins, t0, tp, tw, te = r
                if seq == 0:
                    continue
                recent.append({
                    "seq": seq,
                    "mode": _MODES.get(meta & 0xFF, "?"),
                    "grid": (meta >> 8) & 0xFF,
                    "units": meta >> 16,
                    "spins": rspins,
                    "push_ns": tp - t0,
                    "wait_ns": tw - tp,
                    "combine_ns": te - tw,
                    "total_ns": te - t0,
                    "t_start_ns": t0,
                })
            recent.sort(key = lambda r: r["seq"])
            blocks.append({"block": b, "calls": calls, "spins": spins, "push_ns": push, "wait_ns": wait,
                           "total_ns": total, "recent": recent})
        return {"rank": self.rank, "transport": self.transport, "nblk": self.nblk,
                "enabled": self.stats_enabled, "blocks": blocks}

    def dump_stats(self, prefix: str) -> str:
        """Writes stats() to f"{prefix}.rank{rank}.json" and returns the path. Synchronizes the device."""
        path = f"{prefix}.rank{self.rank}.json"
        with open(path + ".tmp", "w") as f:
            json.dump(self.stats(), f)
        os.replace(path + ".tmp", path)
        return path

    def _check_device(self, t: torch.Tensor) -> None:
        if t.device != self.device:
            raise ValueError(f"tensor on {t.device}, comm bound to {self.device}")
