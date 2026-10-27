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
INIT_METHOD (file://...), TOKENS_OUT (optional json path), PROFILE (0|1).
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
MAX_NEW = int(os.environ.get("MAX_NEW", "64"))
MAX_CHUNK = int(os.environ.get("MAX_CHUNK", "2048"))
CACHE_TOKENS = int(os.environ.get("CACHE_TOKENS", "4096"))
IGNORE_EOS = os.environ.get("IGNORE_EOS", "0") == "1"
COMM = os.environ.get("COMM", "nccl")  # nccl | p2p (P2PComm one-shot collectives; needs P2P_RDV = fresh dir)
PROFILE = os.environ.get("PROFILE", "0") == "1"
TOKENS_OUT = os.environ.get("TOKENS_OUT")
PROMPT_IDS = [int(i) for i in os.environ.get("PROMPT_IDS", "0,1,2").split(",")]
RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
TAG = f"TP2R{RANK}"

PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the nth Fibonacci number.",
    "List three primary colors and explain why they matter in painting.",
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
        flat = torch.empty(2 * t.numel(), dtype = torch.float32, device = t.device)
        self.comm.all_gather(t.float().reshape(-1), flat)
        shards = flat.view(2, *t.shape)
        return torch.cat([shards[0], shards[1]], dim = -1).to(t.dtype)

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
        if isinstance(backend, TPBackendP2P):
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
    cache = Cache(model, max_num_tokens = CACHE_TOKENS, max_history = 5)
    model.load(device = device, max_chunk_size = MAX_CHUNK, max_output_size = MAX_NEW,
               max_batch_size = 1, progressbar = False)
    if COMM == "p2p":
        from exllamav3.tp2.p2p import P2PComm
        rdv = os.environ["P2P_RDV"]
        backend = TPBackendP2P(P2PComm(RANK, world = WORLD, rendezvous_path = rdv, device = RANK))
    else:
        backend = TPBackendNCCL()
    n_blocks = set_tp_reduce(model, backend)
    local_v, full_v = wrap_vocab_split(model, RANK, WORLD, backend)
    load_s = time.time() - t0
    log(f"LOADED {load_s:.1f}s blocks={n_blocks} vocab_local={local_v} vocab_full={full_v} "
        f"max_alloc_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f}")

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer, max_batch_size = 1, max_chunk_size = MAX_CHUNK)
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
    for idx in PROMPT_IDS:
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
        )
        generator.enqueue(job)
        t1 = time.time()
        marks, text, tokens, eos_info = [], "", [], None
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
                if result.get("eos"):
                    eos_info = result.get("eos_reason")
        total = time.time() - t1
        first = marks[0] if marks else float("nan")
        decode = (total - first) / max(len(marks) - 1, 1) if len(marks) > 1 else float("nan")
        out[str(idx)] = tokens
        log(f"PROMPT {prompt!r} prompt_tokens={ids.shape[-1]} new_tokens={len(tokens)} total={total:.3f}s "
            f"first_token_ms={first * 1e3:.1f} decode_ms_per_tok={decode * 1e3:.2f} eos={eos_info} text={text!r}")
        log(f"TOKENS {idx} {tokens}")

    log(f"MEM max_alloc_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f}")
    if TOKENS_OUT:
        with open(f"{TOKENS_OUT}.rank{RANK}.json", "w") as f:
            json.dump(out, f)

    if PROFILE:
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

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
