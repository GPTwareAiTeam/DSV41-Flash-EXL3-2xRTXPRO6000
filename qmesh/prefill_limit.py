"""
Depth-aware prefill chunk limiter (memory headroom guard).

A prefill chunk's transient over the steady allocation is the larger of two terms, fitted to measured chunks
on the 2xRTX TP=2 project (RESULTS 9c): activations, roughly flat in depth, and a depth term that takes over
past ~300k tokens (the uncompressed pool layer's staging window and the indexer score slab). A fixed chunk
that fits a shallow prompt can therefore run out of memory deep into a long one.

Generator.prefill_chunk_limit takes one of these (exllamav3/generator/job.py, prefill()): before every
prefill forward it reads free memory once and solves both terms for the largest page-aligned chunk that fits
under a small safety margin, capped at the planned chunk. On the hybrid seam pair there is a single rank, so
no cross-rank agreement is needed; the all-reduce is kept for multi-process use.

Refit per deployment: PREFILL_ROW_KIB, PREFILL_DEPTH_BASE_MIB and PREFILL_DEPTH_BYTES, with
PREFILL_ADAPTIVE_LOG=1 to print measured rise against prediction per chunk.
"""
from __future__ import annotations

import os
import time

import torch

PAGE_SIZE = 256

PREFILL_ADAPTIVE = os.environ.get("PREFILL_ADAPTIVE", "1") == "1"
PREFILL_ROW_BYTES = float(os.environ.get("PREFILL_ROW_KIB", "382")) * 2 ** 10     # activations per prefill row
PREFILL_DEPTH_BASE = float(os.environ.get("PREFILL_DEPTH_BASE_MIB", "461")) * 2 ** 20
PREFILL_DEPTH_BYTES = float(os.environ.get("PREFILL_DEPTH_BYTES", "1280"))         # per context token
PREFILL_DEPTH_ROWS = 2048                                                          # rows the depth term was fit at
# One-time transient of a prompt's FIRST chunk: the uncompressed-pool staging window and the indexer score
# slab are allocated on first use and stay (they are the `statics` in the memory report: 0.000 GiB after
# load, 0.559 GiB at 131k). Production pair, 8192-row chunks: chunk 1 measured 3573 MiB against a row term
# of 3056 MiB (+517), while chunks 2..16 measured 3056 MiB against the same prediction. A first-chunk term
# of 560 MiB covers that with 8 % headroom and leaves every later chunk's prediction unchanged.
PREFILL_FIRST_BYTES = float(os.environ.get("PREFILL_FIRST_MIB", "560")) * 2 ** 20
# Cushion left free on the card at the predicted peak. Torch's held-but-free memory counts as usable
# (expandable segments let any freed block serve any later shape); PREFILL_FRAG > 0 discounts that share of it
PREFILL_MARGIN_MIB = float(os.environ.get("PREFILL_MARGIN_MIB", "1024"))
PREFILL_FRAG = float(os.environ.get("PREFILL_FRAG", "0"))
PREFILL_ADAPTIVE_LOG = os.environ.get("PREFILL_ADAPTIVE_LOG", "0") == "1"


def log(message):
    print(message, flush = True)


def prefill_rise(start, rows):
    """Predicted transient bytes over the steady allocation for a chunk of `rows` starting at `start`."""
    deep = PREFILL_DEPTH_BASE + PREFILL_DEPTH_BYTES * (start + rows) \
        + PREFILL_ROW_BYTES * max(0, rows - PREFILL_DEPTH_ROWS)
    rise = max(PREFILL_ROW_BYTES * rows, deep)
    return rise + (PREFILL_FIRST_BYTES if start == 0 else 0)


def prefill_fit(start, room):
    """Largest row count whose prefill_rise at `start` is <= room: the minimum of each term solved for rows."""
    room -= PREFILL_FIRST_BYTES if start == 0 else 0
    by_rows = room / PREFILL_ROW_BYTES
    by_depth_lo = (room - PREFILL_DEPTH_BASE - PREFILL_DEPTH_BYTES * start) / PREFILL_DEPTH_BYTES   # rows <= 2048
    by_depth = by_depth_lo if by_depth_lo <= PREFILL_DEPTH_ROWS else \
        (room - PREFILL_DEPTH_BASE - PREFILL_DEPTH_BYTES * start + PREFILL_ROW_BYTES * PREFILL_DEPTH_ROWS) \
        / (PREFILL_DEPTH_BYTES + PREFILL_ROW_BYTES)
    return min(by_rows, by_depth)


class PrefillChunkLimiter:
    """One free-memory read and one solve per prefill chunk for the largest page-aligned chunk that fits (at
    most the planned chunk, at least one page). Keeps the highest per-chunk allocator peak it has seen (it
    resets torch's peak counter per chunk to measure each) and how long each decision took."""

    def __init__(self, device):
        self.device = device
        self.prev = None
        self.peak = 0
        self.shrunk = 0
        self.chunks = 0
        self.decide_us = []

    def _agree(self, rows):
        """Cross-rank agreement (MIN) when torch.distributed is live; single-process runs return as is"""
        try:
            import torch.distributed as dist
            if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
                return rows
            t = torch.tensor([rows], dtype = torch.int32, device = self.device)
            dist.all_reduce(t, op = dist.ReduceOp.MIN)
            return int(t.item())
        except Exception:
            return rows

    def __call__(self, start: int, chunk: int) -> int:
        t0 = time.perf_counter()
        dev = self.device
        alloc = torch.cuda.memory_allocated(dev)
        peak = torch.cuda.max_memory_allocated(dev)
        self.peak = max(self.peak, peak)
        if PREFILL_ADAPTIVE_LOG and self.prev is not None:
            p_start, p_rows, p_alloc, p_t = self.prev
            dt = t0 - p_t
            log(f"PREFILL_CHUNK_PEAK start={p_start} rows={p_rows} rise_MiB={(peak - p_alloc) / 2**20:.0f} "
                f"predicted_MiB={prefill_rise(p_start, p_rows) / 2**20:.0f} ms={dt * 1e3:.0f} "
                f"rows_per_s={p_rows / dt:.0f}")
        torch.cuda.reset_peak_memory_stats(dev)
        # The cushion is kept on the card, not inside torch's pool: the caching allocator holds freed blocks
        # (reserved - allocated) that a larger chunk cannot always reuse, so count only memory the card still
        # has free plus what torch can hand back, less the cushion
        free, _ = torch.cuda.mem_get_info(dev)
        reserved = torch.cuda.memory_reserved(dev)
        room = free + reserved - alloc - PREFILL_MARGIN_MIB * 2 ** 20 - (reserved - alloc) * PREFILL_FRAG
        rows = max(PAGE_SIZE, min(chunk, int(prefill_fit(start, room)) // PAGE_SIZE * PAGE_SIZE))
        rows = self._agree(rows)
        self.chunks += 1
        self.decide_us.append((time.perf_counter() - t0) * 1e6)
        if rows < chunk or PREFILL_ADAPTIVE_LOG:
            self.shrunk += rows < chunk
            if PREFILL_ADAPTIVE_LOG:
                total = torch.cuda.mem_get_info(dev)[1]
                log(f"PREFILL_CHUNK_ADAPT start={start} planned={chunk} rows={rows} room_MiB={room / 2**20:.0f} "
                    f"free_MiB={free / 2**20:.0f} reserved_MiB={reserved / 2**20:.0f} alloc_MiB={alloc / 2**20:.0f} "
                    f"non_torch_MiB={(total - free - reserved) / 2**20:.0f}")
        self.prev = (start, rows, alloc, time.perf_counter())
        return rows

    def prefill_done(self):
        """A prompt's prefill finished: record its last chunk's peak, then hand torch's held-but-free blocks
        back to the card. Prefill chunks free their transients into torch's general pool, while decode captures
        its CUDA graphs from a separate pool that can only grow into memory the card still has free; without
        this a decode capture after a prefill that used big chunks finds the card full."""
        dev = self.device
        self.peak = max(self.peak, torch.cuda.max_memory_allocated(dev))
        if PREFILL_ADAPTIVE_LOG and self.prev is not None:
            p_start, p_rows, p_alloc, _p_t = self.prev
            log(f"PREFILL_PREFILL_DONE start={p_start} rows={p_rows} "
                f"rise_MiB={(torch.cuda.max_memory_allocated(dev) - p_alloc) / 2**20:.0f} "
                f"predicted_MiB={prefill_rise(p_start, p_rows) / 2**20:.0f}")
        self.prev = None
        torch.cuda.empty_cache()

    def summary(self):
        us = self.decide_us
        return dict(chunks = self.chunks, shrunk = self.shrunk, peak_mib = round(self.peak / 2**20, 1),
                    decide_us_median = round(sorted(us)[len(us) // 2], 1) if us else None)
