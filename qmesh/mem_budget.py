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
# Deep-context probes for the statics (dsv4_attn_out and friends): tokens from a tiled corpus, and
# the wall time a probe may be predicted to take before it is skipped (the previous probe scales it).
DEPTH_MAX_S = float(os.environ.get("DEPTH_MAX_S", "420"))
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


ARENA_BLOCK_BYTES = 128 << 20


def _view_key(t: torch.Tensor):
    """Key a tensor the way vram_accounting does: pooled weights live as views into 128 MiB arena
    blocks (counted per view), everything else is counted once per storage."""
    s = t.untyped_storage()
    if s.nbytes() == ARENA_BLOCK_BYTES:
        return ("view", t.data_ptr(), t.numel() * t.element_size())
    return ("storage", s.data_ptr(), s.nbytes())


def walk_named(root, skip_ids = frozenset()):
    """Every CUDA tensor reachable from root, with the attribute path that first reached its key."""
    out: dict = {}
    ids: set = set()
    stack = [(root, "")]
    while stack:
        o, path = stack.pop()
        if id(o) in ids or id(o) in skip_ids:
            continue
        ids.add(id(o))
        if isinstance(o, torch.Tensor):
            if o.is_cuda and o.device == DEVICE:
                out.setdefault(_view_key(o), (path, o))
            continue
        if isinstance(o, (str, bytes, int, float, bool, type(None))):
            continue
        if isinstance(o, (list, tuple, set)):
            stack.extend((v, f"{path}[{i}]") for i, v in enumerate(o))
            continue
        if isinstance(o, dict):
            stack.extend((v, f"{path}[{k}]") for k, v in o.items())
            continue
        try:
            stack.extend((v, f"{path}.{k}") for k, v in vars(o).items())
        except TypeError:
            pass
    return out


def _norm_path(path: str) -> str:
    return path.replace("[", ".").replace("]", "")



def classify_target(path: str) -> str:
    p = _norm_path(path).lower()
    leaf = p.rsplit(".", 1)[-1]
    if ".attn." in p and ("compressor" in p or "indexer" in p):
        return "attn." + ("compressor" if "compressor" in p else "indexer")
    if ".experts." in p or ".experts" in p:
        if leaf == "trellis":
            return "hot_experts.trellis"
        if leaf in ("suh", "svh", "mul1", "scale"):
            return "hot_experts.aux_" + leaf
        return "hot_experts.other"
    if "shared_experts" in p:
        return "shared_experts.trellis" if leaf == "trellis" else "shared_experts.aux"
    if ".ffn.gate" in p or "routing_gate" in p:
        return "router"
    if ".attn." in p:
        for comp in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "sink", "norm"):
            if comp in p:
                return f"attn.{comp}"
        return "attn.other"
    if ".engram" in p:
        return "engram"
    if p.startswith("head.") or ".head." in p:
        return "head"
    if "embed" in p:
        return "embed"
    if "norm" in leaf or "hc_" in p or "expandstream" in p:
        return "norms_hc"
    if p.startswith(("aligner", "image_", "vision")):
        return "vision_aligner"
    return "other:" + p.rsplit(".", 2)[-2] + "." + leaf


def itemize(root, title: str, classify, skip_ids = frozenset(), engine_gib = None):
    """Print MEM_WITEM: CUDA weight bytes of root grouped by classify(path), deduplicated the way
    vram_accounting deduplicates (arena views once each, other tensors once per storage)."""
    rows = walk_named(root, skip_ids)
    by: dict = {}
    ex: dict = {}
    for key, (path, _t) in rows.items():
        cat = classify(path)
        by[cat] = by.get(cat, 0) + key[-1]
        ex.setdefault(cat, []).append(path)
    for cat, nb in sorted(by.items(), key = lambda kv: -kv[1]):
        if gib(nb) < 0.01:
            continue
        sample = ", ".join(ex[cat][:2])
        print(f"MEM_WITEM {title} {gib(nb):.3f} {cat} [{sample}]", flush = True)
    total = sum(by.values())
    tail = (f" engine={engine_gib:.3f} arena_tail={engine_gib - gib(total):.3f}"
            if engine_gib is not None else "")
    print(f"MEM_WITEM {title} {gib(total):.3f} TOTAL n_views={len(rows)}{tail}", flush = True)
    return by


def classify_generator(path: str) -> str:
    p = _norm_path(path)
    # first attribute under the walk root is the generator's own slot; keep two levels of detail
    parts = [s for s in p.split(".") if s]
    return ".".join(parts[:2]) if parts else "?"


def classify_drafter(path: str) -> str:
    p = _norm_path(path).lower()
    leaf = p.rsplit(".", 1)[-1]
    if ".experts" in p:
        return "drafter_experts.trellis" if leaf == "trellis" else "drafter_experts.aux"
    if "shared_experts" in p:
        return "drafter_shared.trellis" if leaf == "trellis" else "drafter_shared.aux"
    if ".attn." in p:
        return "drafter_attn"
    if "markov" in p or "conf" in p:
        return "drafter_head." + leaf
    if "norm" in leaf or "hc_" in p:
        return "drafter_norms_hc"
    if p.startswith("mtp.2") or p.startswith("mtp"):
        return "drafter_block_other." + leaf
    return "drafter_other." + leaf


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


    # --- itemised weight / generator split (owner question: every piece by module type) -------
    iskip = {id(model.config), id(getattr(model.config, "stc", None)), id(cache), id(generator),
             id(tokenizer)}
    if draft_model is not None:
        iskip |= {id(draft_model), id(draft_cache), id(draft_model.config),
                  id(getattr(draft_model.config, "stc", None))}
    iskip.discard(None)
    itemize(model, "target", classify_target, iskip, engine_gib = gib(r0.weights))
    if draft_model is not None:
        itemize(draft_model, "drafter", classify_drafter,
                {id(draft_model.config), id(getattr(draft_model.config, "stc", None)),
                 id(draft_cache), id(generator), id(model), id(cache)})
    itemize(generator, "generator", classify_generator,
            {id(model), id(cache), id(draft_model), id(draft_cache), id(tokenizer),
             id(model.config), id(getattr(model.config, "stc", None))})
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
        st = None
        prefill_done = False
        rows = 0
        # The first yielded event is stage "started", long before any prompt token is ingested, so the
        # split below uses the job's own counters (time_enqueued / time_prefill / time_generate) and the
        # peak is taken after synchronise() once the whole job has drained.
        while generator.num_remaining_jobs():
            for r in generator.iterate():
                if r.get("eos"):
                    st = r
                if not prefill_done and (r.get("text") or r.get("stage") == "streaming"):
                    prefill_done = True
                    peaks("prefill", base_alloc)
                    torch.cuda.reset_peak_memory_stats(DEVICE)
                    base_alloc = torch.cuda.memory_allocated(DEVICE)
                rows += 1
        torch.cuda.synchronize()
        wall = time.time() - t
        st = st or {}
        pf = float(st.get("time_enqueued", 0.0)) + float(st.get("time_prefill", 0.0))
        gen = float(st.get("time_generate", 0.0))
        print(f"MEM_PREFILL tokens={int(ids.shape[-1])} prefill_s={pf:.2f} "
              f"tok_per_s={int(ids.shape[-1]) / max(pf, 1e-9):.1f} "
              f"chunks={limiter.chunks if limiter else None} "
              f"shrunk={limiter.shrunk if limiter else None}", flush = True)
        print(f"MEM_DECODE steps={int(st.get('new_tokens', 0))} wall_s={gen:.2f} "
              f"events={rows} total_wall_s={wall:.2f}", flush = True)
        peaks("decode", base_alloc)
        engine_report(model, cache, generator, "after_long")

    # --- depth probes: statics (dsv4_attn_out et al.) at deeper context ---------------------
    probes = [int(v) for v in os.environ.get("DEPTH_PROBES", "524288,1048570").split(",") if v.strip()]
    if probes:
        texts = ([open(LONG_PROMPT_FILE).read()] if LONG_PROMPT_FILE else []) + \
                [open(p).read() for p in CORPUS_FILES]
        if not texts:
            print("MEM_DEPTH skipped=no-prompt-source", flush = True)
        else:
            base = torch.cat([tokenizer.encode(t, add_bos = False, encode_special_tokens = True)[0]
                              for t in texts])
            bos = tokenizer.encode("a", add_bos = True, encode_special_tokens = True)[0, :1]
            print(f"MEM_DEPTH source_tokens={int(base.shape[-1])} probes={probes}", flush = True)
            prev_want, prev_s = 0, 0.0
            for want in probes:
                if want + 1 > CACHE_TOKENS:
                    print(f"MEM_DEPTH tokens={want} skipped=exceeds-cache", flush = True)
                    continue
                if prev_want and prev_s:
                    pred = prev_s * want / prev_want
                    if pred > DEPTH_MAX_S:
                        print(f"MEM_DEPTH tokens={want} skipped=predicted_s={pred:.0f} "
                              f"limit={DEPTH_MAX_S}", flush = True)
                        continue
                reps = want // int(base.shape[-1]) + 1
                pids = torch.cat([bos, base.repeat(reps)[: want - 1]]).unsqueeze(0)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(DEVICE)
                pbase = torch.cuda.memory_allocated(DEVICE)
                limiter = None
                if PREFILL_ADAPTIVE:
                    generator.prefill_chunk_limit = limiter = PrefillChunkLimiter(DEVICE)
                pst = None
                generator.enqueue(Job(input_ids = pids, max_new_tokens = 1, sampler = GreedySampler()))
                t = time.time()
                while generator.num_remaining_jobs():
                    for r in generator.iterate():
                        if r.get("eos"):
                            pst = r
                torch.cuda.synchronize()
                pst = pst or {}
                ps = float(pst.get("time_enqueued", 0.0)) + float(pst.get("time_prefill", 0.0))
                prev_want, prev_s = want, ps
                print(f"MEM_DEPTH tokens={int(pids.shape[-1])} prefill_s={ps:.2f} "
                      f"tok_per_s={int(pids.shape[-1]) / max(ps, 1e-9):.1f} "
                      f"chunks={limiter.chunks if limiter else None} "
                      f"shrunk={limiter.shrunk if limiter else None} "
                      f"total_wall_s={time.time() - t:.2f}", flush = True)
                peaks(f"depth_{want}", pbase)
                engine_report(model, cache, generator, f"after_{want}")


if __name__ == "__main__":
    main()
