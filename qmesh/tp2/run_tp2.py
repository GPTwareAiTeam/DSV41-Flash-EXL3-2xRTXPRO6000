"""TP=2 per-rank runner for the pre-sharded DeepSeek-V4.1 checkpoint (qmesh/tp2).

One process per GPU (RANK / WORLD_SIZE / INIT_METHOD env, launched by qmesh/tp2/launch_tp2.sh). Each rank
loads ONLY its rank dir from the overlay (rank shards + engram-only tables) as a single-device model on
cuda:RANK, built with the local TP dims of the architecture. Decoding uses the standard Generator.

Collectives (torch.distributed NCCL, see TPBackendNCCL):
- attention and MoE partial sums: all_reduce(SUM) via the modules' tp_reduce hook (params["backend"]).
- vocab-split embedding: rank r owns rows [r*V, (r+1)*V); out-of-range ids -> zero, then all_reduce(SUM).
- vocab-split head: local logits [.., V] all_gather'd in rank order -> full [.., V*WORLD] on both ranks, so
  greedy sampling is identical on both ranks.

Env: OVERLAY (dir holding rank0/ rank1/), MAX_NEW (64), PROMPT_IDS (0,1,2), IGNORE_EOS (0|1),
INIT_METHOD (file://...), TOKENS_OUT (optional json path), PROFILE (0|1), COMM (nccl|p2p), P2P_RDV (fresh dir, p2p),
P2P_STATS_OUT (optional prefix, p2p: enables device spin/timing stats, dumped to <prefix>.rank<r>.json at run end).
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/.cache/torch_extensions")

import torch
import torch.distributed as dist

sys.path.insert(0, "/models/dsv41/DeepSeek-V4.1-Flash-1017-orig/encoding")
from encoding import encode_messages  # noqa: E402

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402
from exllamav3.modules.transformer import TransformerBlock  # noqa: E402

OVERLAY = os.environ.get("OVERLAY", "/artifacts/tp2-overlay")
DSPARK = os.environ.get("DSPARK", "0") == "1"
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", "5"))
# Recurrent-cache history must cover the speculative pass width (generator requires max_history >= K); 5 is the
# previous fixed value, so K<=5 runs are unchanged.
MAX_HISTORY = max(5, NUM_DRAFT)
MAX_NEW = int(os.environ.get("MAX_NEW", "64"))
MAX_CHUNK = int(os.environ.get("MAX_CHUNK", "2048"))
CACHE_TOKENS = int(os.environ.get("CACHE_TOKENS", "4096"))
IGNORE_EOS = os.environ.get("IGNORE_EOS", "0") == "1"
COMM = os.environ.get("COMM", "nccl")  # nccl | p2p | hybrid (hybrid: P2P <64KB, NCCL >=64KB)
P2P_STATS_OUT = os.environ.get("P2P_STATS_OUT")  # p2p only: prefix; enables device spin/timing stats, dumped at run end to <prefix>.rank<r>.json
PROFILE = os.environ.get("PROFILE", "0") == "1"
TOKENS_OUT = os.environ.get("TOKENS_OUT")
LOGITS_OUT = os.environ.get("LOGITS_OUT")
PROBE = [int(n) for n in os.environ.get("PROBE", "").split(",") if n.strip()]
PROMPT_IDS = [int(i) for i in os.environ.get("PROMPT_IDS", "0,1,2").split(",")]
RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
TAG = f"TP2R{RANK}"

PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the nth Fibonacci number.",
    "List three primary colors and explain why they matter in painting.",
    "Summarize the following passage in one sentence: " + "".join(
        "Paragraph %d: the river wound through the valley past terraced fields of barley, "
        "where farmers repaired stone walls each spring and traded seed in the market town. " % i
        for i in range(1, 36)),
]


def log(msg: str):
    print(f"[{TAG}] {msg}", flush=True)


class TPBackendNCCL:
    """Collective backend for the modules' params["backend"] hook (torch.distributed NCCL)."""

    def fwd_barrier(self):
        dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        # A non-contributing rank reduces a zero stand-in so the collective stays aligned
        if not contribution:
            tensor.zero_()
        dist.all_reduce(tensor, op = dist.ReduceOp.SUM)

    def broadcast(self, tensor: torch.Tensor, src_device: int):
        dist.broadcast(tensor, src = src_device)

    def all_gather(self, t: torch.Tensor, world: int) -> torch.Tensor:
        """Concatenate the per-rank t along the last dim, in rank order, on both ranks (== cat(parts, -1))."""
        assert world == 2
        parts = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(parts, t.contiguous())
        return torch.cat(parts, dim = -1)
    def run_cpu_reduce_jobs(self):
        pass

    def end_cpu_reduce_jobs(self):
        pass
P2P_CHUNK = 256 * 1024 // 4  # fp32 elements per P2P arena payload (ARENA_MAXB = 256 KiB, multiple of 4 floats)


class TPBackendP2P:
    """Collective backend on P2PComm (one-shot cudaIpc / pinned-host one-shot all-reduce, fp32 only).
    Same hook surface as TPBackendNCCL. Non-fp32 tensors go through an fp32 staging buffer; the payload must
    fit the P2P arena (<= 256 KiB fp32), and every rank must issue the same sequence of collectives."""

    def __init__(self, comm):
        self.comm = comm
        self._stage = {}

    def _buf(self, t: torch.Tensor) -> torch.Tensor:
        key = (t.numel(), t.device)
        if key not in self._stage:
            self._stage[key] = torch.empty(t.numel(), dtype = torch.float32, device = t.device)
        return self._stage[key]

    def fwd_barrier(self):
        self.comm.barrier()

    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        if not contribution:
            tensor.zero_()
        direct = tensor.dtype == torch.float32 and tensor.is_contiguous()
        s = tensor.view(-1) if direct else self._buf(tensor)
        if not direct:
            s.copy_(tensor.reshape(-1))
        assert s.numel() % 4 == 0, f"P2P all_reduce payload {s.numel()} fp32 not 16-byte divisible"
        # chunk to the arena size; sizes depend only on numel, so both ranks issue the same sequence
        for off in range(0, s.numel(), P2P_CHUNK):
            self.comm.all_reduce_(s[off:off + P2P_CHUNK])
        if not direct:
            tensor.copy_(s.view(tensor.shape))

    def broadcast(self, tensor: torch.Tensor, src_device: int):
        raise NotImplementedError("tp_owner broadcast not supported by P2PComm backend; TP2 splits every module")

    def all_gather(self, t: torch.Tensor, world: int) -> torch.Tensor:
        """Concatenate the per-rank t along the last dim, in rank order, on both ranks (== cat(parts, -1))."""
        assert world == 2
        t = t.contiguous()
        n = t.numel()
        src = t.float().reshape(-1)
        flat = torch.empty(2 * n, dtype = torch.float32, device = t.device)
        # chunk to the arena size; both ranks issue the same sequence (sizes depend only on numel)
        for off in range(0, n, P2P_CHUNK):
            m = min(P2P_CHUNK, n - off)
            part = torch.empty(2 * m, dtype = torch.float32, device = t.device)
            self.comm.all_gather(src[off:off + m].contiguous(), part)
            flat[off:off + m].copy_(part[:m])
            flat[n + off:n + off + m].copy_(part[m:])
        shards = flat.view(2, *t.shape)
        return torch.cat([shards[0], shards[1]], dim = -1).to(t.dtype)

    def run_cpu_reduce_jobs(self):
        pass

    def end_cpu_reduce_jobs(self):
        pass

HYBRID_NCCL_MIN_BYTES = int(os.environ.get("HYBRID_NCCL_MIN_BYTES", 64 * 1024))


class TPBackendHybrid:
    """P2PComm for small payloads (decode), NCCL for large ones (prefill rows). The choice depends only on
    numel and dtype, so both ranks issue the same collective sequence."""

    def __init__(self, nccl: "TPBackendNCCL", p2p: "TPBackendP2P"):
        self.nccl = nccl
        self.p2p = p2p

    def fwd_barrier(self):
        self.p2p.fwd_barrier()

    def _pick(self, tensor: torch.Tensor):
        return self.nccl if tensor.numel() * tensor.element_size() >= HYBRID_NCCL_MIN_BYTES else self.p2p

    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        self._pick(tensor).all_reduce(tensor, contribution)

    def broadcast(self, tensor: torch.Tensor, src_device: int):
        self.nccl.broadcast(tensor, src_device)

    def all_gather(self, t: torch.Tensor, world: int) -> torch.Tensor:
        return self.p2p.all_gather(t, world)

    def run_cpu_reduce_jobs(self):
        pass

    def end_cpu_reduce_jobs(self):
        pass


def set_tp_reduce(model: Model, backend: TPBackendNCCL) -> int:
    """Enable the attention/MoE partial-sum collectives on every TP-local block, and hand the backend to
    every forward through params (Generator builds its params internally, so the entrypoints inject it)."""
    n = 0
    for m in model.modules:
        if isinstance(m, TransformerBlock):
            m.attn.tp_reduce = True
            m.mlp.tp_reduce = True
            n += 1

    def with_backend(fn):
        def wrapped(input_ids, params = None):
            if params is None:
                params = {}
            params["backend"] = backend
            return fn(input_ids, params)
        return wrapped

    model.prefill = with_backend(model.prefill)
    model.forward = with_backend(model.forward)
    return n



def wrap_vocab_split(model: Model, rank: int, world: int, backend: TPBackendNCCL):
    """Vocab-split embedding (masked local lookup + all_reduce) and head (all_gather of local logits)."""
    emb = next(m for m in model.modules if m.key == "embed")
    head = model.modules[model.logit_layer_idx]
    V = emb.vocab_size
    lo = rank * V
    orig_emb = emb.forward
    orig_head = head.forward

    gpu = torch.device("cuda", rank)
    def embed_fwd(x, params, out_dtype = None):
        params.setdefault("input_ids", x)  # hash-MoE routing needs the global ids
        loc = x - lo
        mask = (loc >= 0) & (loc < V)
        loc = torch.where(mask, loc, torch.zeros_like(loc))
        y = orig_emb(loc, params, out_dtype)
        y = y.to(gpu) if y.device != gpu else y
        y = y * mask.to(gpu).unsqueeze(-1).to(y.dtype)
        backend.all_reduce(y)
        return y

    def head_fwd(x, params, out_dtype = None):
        y = orig_head(x, params, out_dtype)
        if isinstance(backend, (TPBackendP2P, TPBackendHybrid)):
            return backend.all_gather(y.contiguous(), world)
        parts = [torch.empty_like(y) for _ in range(world)]
        dist.all_gather(parts, y.contiguous())
        return torch.cat(parts, dim = -1)

    emb.forward = embed_fwd
    head.forward = head_fwd
    return V, head.out_features * world


def main():
    torch.cuda.set_device(RANK)
    dist.init_process_group(backend = "nccl", rank = RANK, world_size = WORLD,
                            init_method = os.environ["INIT_METHOD"])
    device = torch.device("cuda", RANK)
    rank_dir = os.path.join(OVERLAY, f"rank{RANK}")

    t0 = time.time()
    config = Config.from_directory(rank_dir)
    assert config.tp_size == WORLD and config.tp_rank == RANK, \
        f"checkpoint tp_size/tp_rank {config.tp_size}/{config.tp_rank} != launch {WORLD}/{RANK}"
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = CACHE_TOKENS, max_history = MAX_HISTORY)
    model.load(device = device, max_chunk_size = MAX_CHUNK, max_output_size = MAX_NEW,
               max_batch_size = 1, progressbar = False)
    if COMM in ("p2p", "hybrid"):
        from exllamav3.tp2.p2p import P2PComm
        rdv = os.environ["P2P_RDV"]
        p2p = TPBackendP2P(P2PComm(RANK, world = WORLD, rendezvous_path = rdv, device = RANK,
                                   stats = P2P_STATS_OUT is not None))
        backend = p2p if COMM == "p2p" else TPBackendHybrid(TPBackendNCCL(), p2p)
        comm = p2p.comm if hasattr(p2p, "comm") else None
    else:
        comm = None
        backend = TPBackendNCCL()
    n_blocks = set_tp_reduce(model, backend)
    local_v, full_v = wrap_vocab_split(model, RANK, WORLD, backend)
    load_s = time.time() - t0
    log(f"LOADED {load_s:.1f}s blocks={n_blocks} vocab_local={local_v} vocab_full={full_v} "
        f"max_alloc_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f}")

    tokenizer = Tokenizer.from_config(config)
    draft_model = draft_cache = None
    if DSPARK:
        draft_model = Model.from_config(config, component = "mtp")
        draft_cache = Cache(draft_model, max_num_tokens = CACHE_TOKENS, max_history = MAX_HISTORY)
        draft_model.load(device = device, progressbar = False)
        n_draft = set_tp_reduce(draft_model, backend)
        log(f"DRAFT_LOADED blocks={n_draft} free_GiB={torch.cuda.mem_get_info(device)[0] / 2**30:.2f}")
    generator = Generator(model, cache, tokenizer, max_batch_size = 1, max_chunk_size = MAX_CHUNK,
                          draft_model = draft_model, draft_cache = draft_cache,
                          num_draft_tokens = NUM_DRAFT if DSPARK else None)
    FORCE_EXPORT = os.environ.get("FORCE_EXPORT", "0") == "1"
    if FORCE_EXPORT and not DSPARK:
        _export_layers = set(config.dspark_target_layer_ids)
        _orig_forward = model.forward
        def _forward_export(*a, params = None, **kw):
            params["export_state_layers"] = _export_layers
            return _orig_forward(*a, params = params, **kw)
        model.forward = _forward_export
        log(f"FORCE_EXPORT layers={sorted(_export_layers)}")
    dist.barrier()
    # Warm pass: one short job so graph capture and allocator warmup stay out of the timed prompts
    _warm_text = encode_messages([{"role": "user", "content": PROMPTS[PROMPT_IDS[0]]}], thinking_mode = "chat")
    _warm_ids = tokenizer.encode(_warm_text, add_bos = False, encode_special_tokens = True)
    _warm = Job(input_ids = _warm_ids, max_new_tokens = 8, min_new_tokens = 8,
                sampler = GreedySampler(), stop_conditions = [])
    generator.enqueue(_warm)
    while generator.num_remaining_jobs():
        for _ in generator.iterate():
            pass
    out = {}
    WARM = os.environ.get("WARM", "0") == "1"
    run_ids = (PROMPT_IDS + PROMPT_IDS) if WARM else PROMPT_IDS
    for pass_i, idx in enumerate(run_ids):
        timed = not WARM or pass_i >= len(PROMPT_IDS)
        prompt = PROMPTS[idx]
        text_in = encode_messages([{"role": "user", "content": prompt}], thinking_mode = "chat")
        ids = tokenizer.encode(text_in, add_bos = False, encode_special_tokens = True)
        last_ids = ids
        job = Job(
            input_ids = ids,
            max_new_tokens = MAX_NEW,
            min_new_tokens = MAX_NEW if IGNORE_EOS else 0,
            sampler = GreedySampler(),
            stop_conditions = [] if IGNORE_EOS else [tokenizer.eos_token_id],
            return_logits = bool(LOGITS_OUT),
        )
        generator.enqueue(job)
        t1 = time.time()
        marks, text, tokens, eos_info, logit_rows, spec = [], "", [], None, [], None
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if "error" in result:
                    raise result["error"]
                chunk = result.get("text", "")
                if chunk:
                    text += chunk
                    marks.append(time.time() - t1)
                if "token_ids" in result:
                    tokens += result["token_ids"][0].tolist()
                if "logits" in result and LOGITS_OUT:
                    logit_rows.append(result["logits"].float().cpu())
                if "accepted_draft_tokens" in result:
                    spec = result
                if result.get("eos"):
                    eos_info = result.get("eos_reason")
        total = time.time() - t1
        if not timed:
            continue
        first = marks[0] if marks else float("nan")
        decode = (total - first) / max(len(marks) - 1, 1) if len(marks) > 1 else float("nan")
        out[str(idx)] = tokens
        if LOGITS_OUT and logit_rows:
            top = []
            for row in logit_rows:
                v, i = torch.topk(row.reshape(-1, row.shape[-1]), 2, dim = -1)
                top.append({"top1": int(i[-1, 0]), "top2": int(i[-1, 1]), "gap": float(v[-1, 0] - v[-1, 1])})
            logit_out[str(idx)] = top
        log(f"PROMPT {prompt!r} prompt_tokens={ids.shape[-1]} new_tokens={len(tokens)} total={total:.3f}s "
            f"first_token_ms={first * 1e3:.1f} decode_ms_per_tok={decode * 1e3:.2f} eos={eos_info} text={text!r}")
        log(f"TOKENS {idx} {tokens}")
        if spec is not None:
            acc, rej, rnd = spec["accepted_draft_tokens"], spec["rejected_draft_tokens"], spec.get("verify_rounds", 0)
            acc_rate = acc / max(acc + rej, 1)
            tpr = 1 + acc / max(rnd, 1)
            log(f"SPEC {idx} accepted={acc} rejected={rej} acc_rate={acc_rate:.4f} verify_rounds={rnd} "
                f"tokens_per_round={tpr:.3f} eff_tok_s={len(tokens) / total:.2f}")
        log(f"TIMING {idx} total={total:.3f}s eff_tok_s={len(tokens) / total:.2f}")

    if PROBE and "0" in out:
        # Seq>1 probe: prompt + the first N generated tokens as ONE prefill forward, then the next-token logits.
        # Compared against the 1-token-decode reference (logit_out["0"][N]) to isolate the multi-row target path.
        ids0 = tokenizer.encode(encode_messages([{"role": "user", "content": PROMPTS[0]}], thinking_mode = "chat"),
                                add_bos = False, encode_special_tokens = True)
        for n in PROBE:
            pj = Job(input_ids = torch.cat([ids0, torch.tensor([out["0"][:n]], dtype = ids0.dtype)], dim = -1),
                     max_new_tokens = 1, min_new_tokens = 1, sampler = GreedySampler(), stop_conditions = [],
                     return_logits = True)
            generator.enqueue(pj)
            got = None
            while generator.num_remaining_jobs():
                for r in generator.iterate():
                    if "error" in r: raise r["error"]
                    if "logits" in r: got = r["logits"].float().cpu()
                    if "token_ids" in r: probe_tok = r["token_ids"][0].tolist()
            v, i = torch.topk(got.reshape(-1, got.shape[-1])[-1], 2)
            log(f"PROBE n={n} next_tok={probe_tok} top1={int(i[0])} top2={int(i[1])} gap={float(v[0]-v[1]):.5f}")
    log(f"MEM max_alloc_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f}")
    if LOGITS_OUT:
        with open(f"{LOGITS_OUT}.rank{RANK}.json", "w") as f:
            json.dump(logit_out, f)
    if TOKENS_OUT:
        with open(f"{TOKENS_OUT}.rank{RANK}.json", "w") as f:
            json.dump(out, f)

    PROFILE_SEQ = [int(n) for n in os.environ.get("PROFILE_SEQ", "").split(",") if n.strip()]
    if PROFILE and PROFILE_SEQ:
        # Width-filtered profile: profiles exactly one model.forward call per width in PROFILE_SEQ
        # (verify = seq 6, decode = seq 1), because torch.profiler clears events per cycle.
        from torch.profiler import profile, ProfilerActivity
        _orig_fwd = model.forward
        widths, done_w = {}, set()
        profs = {}
        def _filtered_forward(input_ids, params = None, **kw):
            w = int(input_ids.shape[-1])
            widths[w] = widths.get(w, 0) + 1
            if w not in PROFILE_SEQ or w in done_w or widths[w] < 3:
                return _orig_fwd(input_ids, params = params, **kw)
            done_w.add(w)
            p = profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA])
            profs[w] = p
            torch.cuda.synchronize()
            p.start()
            try:
                return _orig_fwd(input_ids, params = params, **kw)
            finally:
                torch.cuda.synchronize()
                p.stop()
        model.forward = _filtered_forward
        job = Job(input_ids = last_ids, max_new_tokens = 64, min_new_tokens = 64,
                  sampler = GreedySampler(), stop_conditions = [])
        generator.enqueue(job)
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if "error" in result: raise result["error"]
        model.forward = _orig_fwd
        if RANK == 0:
            log(f"PROFILE_SEQ widths_seen={sorted(widths.items())} recorded={sorted(profs)}")
            for w in sorted(profs):
                print(f"=== PROFILE seq={w} (one forward) ===", flush = True)
                print(profs[w].key_averages().table(sort_by = "cuda_time_total", row_limit = 30), flush = True)
                profs[w].export_chrome_trace(f"{os.environ.get('PROFILE_TRACE', '/artifacts/tp2logs/profile_trace.json')}.seq{w}.json")
    if PROFILE and not PROFILE_SEQ:
        from torch.profiler import profile, ProfilerActivity
        # both ranks must run the same collective sequence; only rank 0 prints the table.
        # Profiling starts after the first emitted token, so prefill is excluded: decode steps only.
        prof = profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA])
        job = Job(input_ids = last_ids, max_new_tokens = 16, min_new_tokens = 16,
                  sampler = GreedySampler(), stop_conditions = [])
        generator.enqueue(job)
        started, steps, t_dec, wall_ms = False, 0, None, float("nan")
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if "token_ids" in result and not started:
                    torch.cuda.synchronize()
                    prof.start()
                    started, t_dec = True, time.time()
                elif "token_ids" in result:
                    steps += 1
        torch.cuda.synchronize()
        if started:
            wall_ms = (time.time() - t_dec) * 1e3 / max(steps, 1)
            prof.stop()
        if RANK == 0:
            log(f"PROFILE decode_steps={steps} wall_ms_per_step={wall_ms:.2f}")
            prof.export_chrome_trace(os.environ.get("PROFILE_TRACE", "/artifacts/tp2logs/profile_trace.json"))
            print(prof.key_averages().table(sort_by = "self_cpu_time_total", row_limit = 45), flush = True)

    if P2P_STATS_OUT and COMM == "p2p":
        # Rank-tagged dump after all collectives drained (stats() syncs the device). Written on every rank.
        path = comm.dump_stats(P2P_STATS_OUT)
        log(f"P2P_STATS dumped {path}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
