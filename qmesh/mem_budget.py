#!/usr/bin/env python3
"""Per-owner VRAM budget for the Recipe A hybrid on the production pair (P7 GPU0 hot + GX1 cold).

Loads exactly as qmesh/e2e_prompts.py does at the shipped config, then reports, per device:

  MEM_OWNER                      resident CUDA tensors by owner (deduplicated by storage)
  MEMB                           engine accounting: weights / kv cache / recurrent / statics /
                                 generator / other, plus statics by tag (graph pools, workspaces)
  MEM_KV                         pool bytes per token, scaled to 64k / 131k / 1M tokens
  MEM_PREFILL / MEM_DECODE       peak allocated+reserved during a long-context prefill, then
                                 during the decode steps that follow it, with free at peak

Phases: after load, after the warm-up decode (graphs captured -> statics exist), after the long
prefill, and after decode.

Env:
  CHECKPOINT, DRAFT_CHECKPOINT (defaults to CHECKPOINT), PLACEMENT,
  KV_BITS (4|8|16, default 4), CACHE_TOKENS (default 1048576),
  DSPARK (1 loads the drafter), NUM_DRAFT (default 5), WARMUP_NEW (decode tokens of the warm-up
  prompt, default 24), LONG_PROMPT_FILE or CORPUS+LONG_TOKENS, LONG_NEW (decode tokens after the
  long prompt, default 16), MAX_CHUNK (default 2048), PREFILL_ADAPTIVE, EXL3_WHOLE_GRAPH
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
os.environ.setdefault("EXL3_WHOLE_GRAPH", "1")

import torch

from exllamav3 import Cache, CacheLayer_quant, Config, Generator, GreedySampler, Job, Model, Tokenizer
from exllamav3.util.memory import vram_accounting
from qmesh.prefill_limit import PREFILL_ADAPTIVE, PrefillChunkLimiter

CHECKPOINT = os.environ.get("CHECKPOINT", "/models/dsv41/exl3-serve-final")
DRAFT_CHECKPOINT = os.environ.get("DRAFT_CHECKPOINT", CHECKPOINT)
PLACEMENT = os.environ.get("PLACEMENT", os.path.join(os.path.dirname(__file__), "placement.final.json"))
KV_BITS = int(os.environ.get("KV_BITS", "4"))
CACHE_TOKENS = int(os.environ.get("CACHE_TOKENS", "1048576"))
DSPARK = os.environ.get("DSPARK") == "1"
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", "5"))
WARMUP_NEW = int(os.environ.get("WARMUP_NEW", "24"))
LONG_PROMPT_FILE = os.environ.get("LONG_PROMPT_FILE", "")
LONG_TOKENS = int(os.environ.get("LONG_TOKENS", "0"))
LONG_NEW = int(os.environ.get("LONG_NEW", "16"))
CORPUS_FILES = [p for p in os.environ.get("CORPUS", "").split(",") if p.strip()]
MAX_CHUNK = int(os.environ.get("MAX_CHUNK", "2048"))
LOAD_ONLY = os.environ.get("LOAD_ONLY") == "1"
DEVICE = torch.device("cuda:0")
G = 2 ** 30

assert KV_BITS in (4, 8, 16), "KV_BITS must be 4, 8 or 16"

KV_CACHE_KWARGS = ({} if KV_BITS == 16 else
                   {"kv_format": "fp4"} if KV_BITS == 4 else
                   {"layer_type": CacheLayer_quant, "k_bits": 8, "v_bits": 8})


def _io_counters() -> dict:
    """This process's byte counters (/proc/self/io), for separating disk reads from CPU work."""
    out = {}
    try:
        with open("/proc/self/io") as f:
            for line in f:
                k, _, v = line.partition(":")
                try:
                    out[k.strip()] = int(v)
                except ValueError:
                    pass
    except OSError:
        pass
    return out

def gib(n) -> float:
    return n / G


def walk_tensors(root, seen: set, out: dict):
    """CUDA tensors reachable from root through attributes/containers, deduplicated by storage.
    `seen` is shared between calls, so nothing is counted twice across owners."""
    stack = [root]
    while stack:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        if isinstance(o, torch.Tensor):
            if o.is_cuda and o.device == DEVICE:
                key = o.untyped_storage().data_ptr()
                if key not in out:
                    out[key] = o.numel() * o.element_size()
            continue
        if isinstance(o, (str, bytes, int, float, bool, type(None))):
            continue
        if isinstance(o, (list, tuple, set)):
            stack.extend(o)
            continue
        if isinstance(o, dict):
            stack.extend(o.values())
            continue
        try:
            stack.extend(list(vars(o).values()))
        except TypeError:
            pass


class Owner:
    def __init__(self):
        self.seen: set = set()
        self.accounted: set = set()
        self.rows: list = []
        self.count: dict = {}

    def take(self, label, root, count = 0):
        out: dict = {}
        walk_tensors(root, self.seen, out)
        new = sum(v for k, v in out.items() if k not in self.accounted)
        self.accounted.update(out)
        self.rows.append((label, gib(new)))
        if count:
            self.count[label] = count
        return new

    def totals(self) -> dict:
        t: dict = {}
        for label, gb in self.rows:
            t[label.split()[0]] = t.get(label.split()[0], 0.0) + gb
        return t


def take_mlp(own: Owner, mlp, tag: str, expert_label: str):
    """Split a routed MoE: router, shared experts (+latent projections, norms), hot experts, rest."""
    for name in ("routing_gate", "gate"):
        m = getattr(mlp, name, None)
        if isinstance(m, torch.nn.Module):
            own.take(f"router {tag}", m)
            break
    shared = []
    for name in ("shared_gate", "shared_experts", "latent_in", "latent_out",
                 "shared_experts_post_norm", "router_pre_norm", "routed_pre_norm", "routed_post_norm"):
        m = getattr(mlp, name, None)
        if isinstance(m, torch.nn.Module):
            shared.append(m)
    own.take(f"shared {tag}", shared)
    experts, n_hot = [], 0
    for list_name in ("gates", "ups", "downs"):
        for lin in getattr(mlp, list_name, []) or []:
            experts.append(lin)
            probe: dict = {}
            walk_tensors(lin, set(), probe)
            if probe:
                n_hot += 1
    own.take(f"{expert_label} {tag}", experts, count = n_hot)
    own.take(f"moe_other {tag}", mlp)


def take_block(own: Owner, m, tag: str):
    attn = getattr(m, "attn", None)
    if isinstance(attn, torch.nn.Module):
        own.take(f"attn {tag}", attn)
    mlp = getattr(m, "mlp", None)
    if isinstance(mlp, torch.nn.Module):
        if "BlockSparse" in type(mlp).__name__:
            take_mlp(own, mlp, tag, "hot_experts")
        else:
            own.take(f"dense_mlp {tag}", mlp)
    own.take(f"block_other {tag}", m)


def report(own: Owner, title: str):
    for label, gb in own.rows:
        if gb >= 0.02:
            extra = f" n={own.count[label]}" if label in own.count else ""
            print(f"MEM_OWNER {title} {gb:.3f} {label}{extra}", flush = True)
    for cat, gb in sorted(own.totals().items(), key = lambda kv: -kv[1]):
        if gb >= 0.02:
            print(f"MEM_GROUP {title} {gb:.3f} {cat}", flush = True)


def engine_report(model, cache, generator, title: str):
    r = vram_accounting(model, cache, generator, [DEVICE])[0]
    print(f"MEMB {title} weights={gib(r.weights):.3f} kv_cache={gib(r.cache):.3f} "
          f"recurrent={gib(r.recurrent):.3f} statics={gib(r.statics):.3f} "
          f"generator={gib(r.generator):.3f} other={gib(r.other):.3f}", flush = True)
    for tag, b in sorted(r.statics_by_tag.items(), key = lambda kv: -kv[1]):
        print(f"MEMB statics_tag {title} {tag}={gib(b):.3f}", flush = True)
    free, total = torch.cuda.mem_get_info(DEVICE)
    print(f"MEMB device {title} total={gib(total):.3f} used={(total - free) / G:.3f} "
          f"torch_reserved={gib(torch.cuda.memory_reserved(DEVICE)):.3f} "
          f"torch_allocated={gib(torch.cuda.memory_allocated(DEVICE)):.3f} "
          f"non_torch={(total - free - torch.cuda.memory_reserved(DEVICE)) / G:.3f} "
          f"free={gib(free):.3f}", flush = True)


def peaks(title: str, base_alloc: int):
    free, total = torch.cuda.mem_get_info(DEVICE)
    peak_a = torch.cuda.max_memory_allocated(DEVICE)
    peak_r = torch.cuda.max_memory_reserved(DEVICE)
    print(f"MEM_PEAK {title} peak_alloc={gib(peak_a):.3f} peak_reserved={gib(peak_r):.3f} "
          f"rise_alloc={gib(peak_a - base_alloc):.3f} free_at_peak={gib(total - peak_r):.3f} "
          f"free_now={gib(free):.3f} used_now={(total - free) / G:.3f}", flush = True)


def main():
    placement = json.load(open(PLACEMENT))
    hot = {f"layers.{layer}.ffn": v["hot"] for layer, v in placement["layers"].items()}
    config = Config.from_directory(CHECKPOINT)
    config.infer_params.moe_cpu_split = 1
    config.infer_params.moe_remote_hot = hot
    config.infer_params.remote_expert_host = {
        "host": os.environ.get("SEAM_PEER", "10.0.100.2"),
        "port": int(os.environ.get("SEAM_PORT", "47311")),
    }
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, max_history=5, **KV_CACHE_KWARGS)
    t0 = time.time()
    io0 = _io_counters()
    model.load(device="cuda:0", progressbar=False)
    load_s = time.time() - t0
    free, total = torch.cuda.mem_get_info(DEVICE)
    print(f"P7_LOAD_OK {load_s:.1f}s vram={gib(torch.cuda.memory_allocated(DEVICE)):.2f}GiB "
          f"free={gib(free):.2f}GiB total={gib(total):.2f}GiB kv_bits={KV_BITS} "
          f"cache_tokens={CACHE_TOKENS} dspark={int(DSPARK)}", flush = True)
    io1 = _io_counters()
    print(f"LOADIO seconds={load_s:.1f} read_gib={gib(io1['read_bytes'] - io0['read_bytes']):.2f} "
          f"rchar_gib={gib(io1['rchar'] - io0['rchar']):.2f} "
          f"mib_per_s={gib(io1['read_bytes'] - io0['read_bytes']) * 1024 / max(load_s, 1e-9):.1f} "
          f"cpu_s={time.process_time():.1f}", flush = True)
    if LOAD_ONLY:
        return 0

    tokenizer = Tokenizer.from_config(config)
    draft_model = None
    draft_cache = None
    if DSPARK:
        draft_config = Config.from_directory(DRAFT_CHECKPOINT)
        draft_model = Model.from_config(draft_config, component="mtp")
        draft_cache = Cache(draft_model, max_num_tokens=CACHE_TOKENS, max_history=5, **KV_CACHE_KWARGS)
        t1 = time.time()
        draft_model.load(device="cuda:0", progressbar=False)
        print(f"DSPARK_LOADED {time.time() - t1:.1f}s drafter={DRAFT_CHECKPOINT}", flush = True)
    generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=MAX_CHUNK,
                          draft_model=draft_model, draft_cache=draft_cache,
                          num_draft_tokens=NUM_DRAFT if DSPARK else None)
    engine_report(model, cache, generator, "after_load")
    if draft_model is not None:
        engine_report(draft_model, draft_cache, None, "drafter_after_load")

    # KV/state cost, scaled from the allocation the shipped CACHE_TOKENS size actually makes
    r0 = vram_accounting(model, cache, generator, [DEVICE])[0]
    print(f"MEM_KV pool_bytes={gib(r0.cache):.3f} cache_tokens={CACHE_TOKENS} "
          f"bytes_per_token={r0.cache / max(CACHE_TOKENS, 1):.1f} "
          f"at_64k={gib(r0.cache * 65536 / max(CACHE_TOKENS, 1)):.3f} "
          f"at_131k={gib(r0.cache * 131072 / max(CACHE_TOKENS, 1)):.3f} "
          f"recurrent_bytes={gib(r0.recurrent):.3f}", flush = True)
    if draft_cache is not None:
        r1 = vram_accounting(draft_model, draft_cache, None, [DEVICE])[0]
        print(f"MEM_KV drafter_pool_bytes={gib(r1.cache):.3f} "
              f"drafter_recurrent_bytes={gib(r1.recurrent):.3f}", flush = True)

    # --- after load ------------------------------------------------------------------------
    skip = {id(draft_model), id(draft_cache), id(generator), id(model), id(model.config),
            id(getattr(model.config, "stc", None))}
    if draft_model is not None:
        skip |= {id(draft_model), id(draft_model.config), id(draft_cache)}
    skip.discard(None)
    own = Owner()
    own.seen |= skip
    for m in model.modules:
        name, key = type(m).__name__, getattr(m, "key", "") or ""
        if name == "TransformerBlock":
            take_block(own, m, key)
        elif name == "EngramLayer":
            own.take(f"engram {key}", m)
        elif name == "Embedding" or key in ("embed",):
            own.take("embed", m)
        elif name in ("HyperHead",) or key in ("head",):
            own.take("head", m)
        elif key in ("norm", "hc_head", "hc_expand") or "Norm" in name or name == "ExpandStreams":
            own.take(f"norms_hc {key or name}", m)
        else:
            own.take(f"other {key or name}", m)
    report(own, "target")
    if draft_model is not None:
        down = Owner()
        down.seen |= {id(draft_model), id(draft_model.config), id(draft_cache), id(generator)}
        for m in draft_model.modules:
            name, key = type(m).__name__, getattr(m, "key", "") or ""
            if hasattr(m, "attn") and hasattr(m, "mlp"):
                take_block(down, m, key)
            elif "BlockSparse" in name:
                take_mlp(down, m, key, "drafter_experts")
            elif "head" in key.lower() or "Head" in name:
                down.take(f"drafter_head {key or name}", m)
            else:
                down.take(f"drafter_other {key or name}", m)
        report(down, "drafter")
    # --- warm-up: capture graphs, then account for statics --------------------------------
    def run(ids, max_new, chunk_label):
        limiter = None
        if PREFILL_ADAPTIVE:
            generator.prefill_chunk_limit = limiter = PrefillChunkLimiter(DEVICE)
        job = Job(input_ids = ids, max_new_tokens = max_new, sampler = GreedySampler())
        generator.enqueue(job)
        t = time.time()
        first_t = None
        n = 0
        while generator.num_remaining_jobs():
            for _ in generator.iterate():
                if first_t is None:
                    first_t = time.time() - t
                n += 1
        return {"wall_s": time.time() - t, "ttft_s": first_t, "steps": n,
                "chunks": limiter.chunks if limiter else None,
                "shrunk": limiter.shrunk if limiter else None}

    warm_ids = tokenizer.encode("Write a paragraph about the history of sailing ships.",
                                add_bos = True, encode_special_tokens = True)
    torch.cuda.synchronize()
    w = run(warm_ids, WARMUP_NEW, "warm")
    print(f"MEM_WARM {json.dumps(w)}", flush = True)
    engine_report(model, cache, generator, "after_warm")

    # --- long-context prefill + decode peaks ----------------------------------------------
    if LONG_PROMPT_FILE:
        ids = tokenizer.encode(open(LONG_PROMPT_FILE).read(), add_bos = True,
                               encode_special_tokens = True)
        if LONG_TOKENS:
            ids = ids[:, :LONG_TOKENS]
        print(f"LONG_PROMPT tokens={int(ids.shape[-1])} file={LONG_PROMPT_FILE}", flush = True)
    elif LONG_TOKENS:
        pieces = [tokenizer.encode(open(p).read(), add_bos = False, encode_special_tokens = True)[0]
                  for p in CORPUS_FILES]
        flat = torch.cat(pieces)
        bos = tokenizer.encode("a", add_bos = True, encode_special_tokens = True)[0, :1]
        reps = LONG_TOKENS // int(flat.shape[-1]) + 1
        ids = torch.cat([bos, flat.repeat(reps)[: LONG_TOKENS - 1]]).unsqueeze(0)
        print(f"LONG_PROMPT tokens={int(ids.shape[-1])} corpus={len(CORPUS_FILES)}", flush = True)
    else:
        ids = None

    if ids is not None:
        assert int(ids.shape[-1]) + LONG_NEW <= CACHE_TOKENS, \
            f"long prompt {int(ids.shape[-1])} + {LONG_NEW} > CACHE_TOKENS={CACHE_TOKENS}"
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(DEVICE)
        base_alloc = torch.cuda.memory_allocated(DEVICE)
        limiter = None
        if PREFILL_ADAPTIVE:
            generator.prefill_chunk_limit = limiter = PrefillChunkLimiter(DEVICE)
        job = Job(input_ids = ids, max_new_tokens = LONG_NEW, sampler = GreedySampler())
        generator.enqueue(job)
        t = time.time()
        first = None
        rows = 0
        while generator.num_remaining_jobs():
            for _ in generator.iterate():
                if first is None:
                    first = time.time() - t
                    print(f"MEM_PREFILL tokens={int(ids.shape[-1])} wall_s={first:.1f} "
                          f"tok_per_s={int(ids.shape[-1]) / max(first, 1e-9):.1f} "
                          f"chunks={limiter.chunks if limiter else None} "
                          f"shrunk={limiter.shrunk if limiter else None}", flush = True)
                    peaks("prefill", base_alloc)
                    torch.cuda.reset_peak_memory_stats(DEVICE)
                    base_alloc = torch.cuda.memory_allocated(DEVICE)
                rows += 1
        print(f"MEM_DECODE steps={rows} wall_s={time.time() - t - (first or 0):.2f} "
              f"ms_per_step={1000 * (time.time() - t - (first or 0)) / max(rows - 1, 1):.1f}", flush = True)
        peaks("decode", base_alloc)
        engine_report(model, cache, generator, "after_long")


if __name__ == "__main__":
    main()
