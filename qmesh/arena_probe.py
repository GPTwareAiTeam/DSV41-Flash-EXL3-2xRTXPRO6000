"""
Slab-arena audit: what a load leaves resident beyond the weights it hands to the modules.

A weight tensor loaded inside a deferred-load bracket is a slice of a 128 MiB slab block, so the
block's unused tail (everything past the last tensor placed in it) is unreachable memory that stays
resident as long as any tensor from that block does. This probe loads the target and the drafter the
same way qmesh/mem_budget.py does, then reports, per owner:
  ARENA_BLOCK   per block: capacity, high-water fill, dead tail, tensors still carving from it
  ARENA_REPORT  per device: blocks, capacity, used, dead, and the tail the itemised weights miss
  MEM_WITEM     the itemised weight bytes by module category (the engine's weight figure is compared)
  MEMB          engine accounting (weights / kv / recurrent / statics / generator / other)
It needs no GX1 server: nothing is forwarded, the load is local.

Env: the same as qmesh/mem_budget.py (CHECKPOINT, DRAFT_CHECKPOINT, DSPARK, KV_BITS, CACHE_TOKENS),
plus the arena knobs themselves, which the loader reads before this file runs:
  EXL3_LOAD_ARENA        1 (shipped slabs) | 0 (one allocation per tensor) | compact (relocate tails)
  EXL3_ARENA_BLOCK_MIB   128 (shipped) | 32
  EXL3_ARENA_MAX_OPEN    8 (shipped)
  EXL3_ARENA_REPORT      1 (device totals) | 2 (one line per block)
  ARENA_PROBE_ARMS       space separated per-arm env overrides, e.g. "EXL3_LOAD_ARENA=0 EXL3_ARENA_BLOCK_MIB=32"
                         -- each arm is a separate process run by scripts/arena_probe.sh, not here
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import torch

from exllamav3 import Cache, Config, Model
from exllamav3.util.memory import vram_accounting
from qmesh.mem_budget import (
    CACHE_TOKENS, DEVICE, KV_CACHE_KWARGS, PLACEMENT, gib, engine_report, itemize, classify_target,
    classify_drafter, dump_byname, walk_named, _io_counters,
)

G = 2 ** 30


def digest(root, skip_ids, owner: str):
    """Content digest of every CUDA tensor an owner holds: name, shape, dtype, nbytes and a strided
    4 KiB sample of the bytes. Compaction and allocator changes must not alter any of it (the sample
    stride is derived from the size, so a shifted or truncated copy changes most samples)."""
    if os.environ.get("ARENA_PROBE_HASH", "1") != "1":
        return
    h = hashlib.sha256()
    n = nbytes = skipped = 0
    for path, (_key, t) in sorted(walk_named(root, skip_ids).items()):
        if not t.is_cuda:
            continue
        n += 1
        nbytes += t.nbytes
        h.update(f"{path}|{tuple(t.shape)}|{t.dtype}|{t.nbytes}".encode())
        if t.is_contiguous() and t.numel():
            b = t.view(torch.uint8).flatten()
            step = max(1, b.numel() // 4096)
            h.update(b[::step][:4096].cpu().numpy().tobytes())
        else:
            skipped += 1
    print(f"ARENA_HASH owner={owner} digest={h.hexdigest()[:16]} tensors={n} "
          f"bytes={nbytes} gib={gib(nbytes):.3f} non_contiguous={skipped}", flush = True)


def arena_lines(tag: str):
    """The arena knobs this process ran with, so a log of several arms stays readable."""
    print(f"ARENA_PROBE tag={tag} load_arena={os.environ.get('EXL3_LOAD_ARENA', '1')} "
          f"block_mib={os.environ.get('EXL3_ARENA_BLOCK_MIB', '128')} "
          f"max_open={os.environ.get('EXL3_ARENA_MAX_OPEN', '8')} "
          f"report={os.environ.get('EXL3_ARENA_REPORT', '0')}", flush = True)


def main():
    placement = json.load(open(PLACEMENT))
    hot = {f"layers.{layer}.ffn": v["hot"] for layer, v in placement["layers"].items()}
    config = Config.from_directory(os.environ["CHECKPOINT"])
    config.infer_params.moe_cpu_split = 1
    config.infer_params.moe_remote_hot = hot
    config.infer_params.remote_expert_host = {
        "host": os.environ.get("SEAM_PEER", "10.0.100.2"),
        "port": int(os.environ.get("SEAM_PORT", "47311")),
    }
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, max_history=5, **KV_CACHE_KWARGS)
    arena_lines("target_start")
    t0 = time.time()
    io0 = _io_counters()
    model.load(device="cuda:0", progressbar=False)
    load_s = time.time() - t0
    io1 = _io_counters()
    free, total = torch.cuda.mem_get_info(DEVICE)
    print(f"ARENA_PROBE target_load_s={load_s:.2f} allocated={gib(torch.cuda.memory_allocated(DEVICE)):.3f} "
          f"reserved={gib(torch.cuda.memory_reserved(DEVICE)):.3f} "
          f"unallocated_reserved={gib(torch.cuda.memory_reserved(DEVICE) - torch.cuda.memory_allocated(DEVICE)):.3f} "
          f"free={gib(free):.3f} read_gib={gib(io1['read_bytes'] - io0['read_bytes']):.2f} "
          f"cpu_s={time.process_time():.1f}", flush = True)
    r0 = vram_accounting(model, cache, None, [DEVICE])[0]
    engine_report(model, cache, None, "target")
    # engine_gib makes itemize print the tail the itemised weights miss: the slab block bytes no
    # weight tensor accounts for (this probe's subject)
    itemize(model, "target", classify_target,
            {id(model.config), id(getattr(model.config, "stc", None)), id(cache)},
            engine_gib = gib(r0.weights))

    digest(model, {id(model.config), id(getattr(model.config, "stc", None)), id(cache)}, "target")

    draft_model = None
    if os.environ.get("DSPARK", "1") == "1":
        draft_config = Config.from_directory(os.environ["DRAFT_CHECKPOINT"])
        draft_model = Model.from_config(draft_config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=CACHE_TOKENS, max_history=5, **KV_CACHE_KWARGS)
        t1 = time.time()
        draft_model.load(device="cuda:0", progressbar=False)
        print(f"ARENA_PROBE drafter_load_s={time.time() - t1:.2f}", flush = True)
        free, total = torch.cuda.mem_get_info(DEVICE)
        print(f"ARENA_PROBE drafter_allocated={gib(torch.cuda.memory_allocated(DEVICE)):.3f} "
              f"reserved={gib(torch.cuda.memory_reserved(DEVICE)):.3f} "
              f"unallocated_reserved={gib(torch.cuda.memory_reserved(DEVICE) - torch.cuda.memory_allocated(DEVICE)):.3f} "
              f"free={gib(free):.3f}", flush = True)
        r1 = vram_accounting(draft_model, draft_cache, None, [DEVICE])[0]
        engine_report(draft_model, draft_cache, None, "drafter")
        skip = {id(draft_model.config), id(getattr(draft_model.config, "stc", None)), id(draft_cache),
                id(model), id(cache)}
        itemize(draft_model, "drafter", classify_drafter, skip, engine_gib = gib(r1.weights))
        dump_byname(draft_model, "drafter", skip,
                    min_mib = float(os.environ.get("BYNAME_MIN_MIB", "0.5")))
        digest(draft_model, skip, "drafter")

    free, total = torch.cuda.mem_get_info(DEVICE)
    print(f"ARENA_PROBE end allocated={gib(torch.cuda.memory_allocated(DEVICE)):.3f} "
          f"reserved={gib(torch.cuda.memory_reserved(DEVICE)):.3f} "
          f"non_torch={(total - free - torch.cuda.memory_reserved(DEVICE)) / G:.3f} "
          f"free={gib(free):.3f} total={gib(total):.3f}", flush = True)
    print("ARENA_PROBE_DONE", flush = True)


if __name__ == "__main__":
    main()
