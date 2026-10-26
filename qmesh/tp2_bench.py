"""TP2/EP2 bring-up bench: load once, greedy MAX_NEW tokens x 3 chat prompts, per-rank memory, rank-0 profile.

Env: CHECKPOINT, MAX_NEW (64), MAX_CHUNK (2048), TP_BACKEND (native|nccl), MOE_TENSOR_SPLIT (0|1),
IGNORE_EOS (1: force MAX_NEW tokens; 0: stop at EOS via stop_conditions), PROFILE (0|1: torch.profiler
short generation on rank 0), TAG (log label), TRACE_DIR, TOKENS_OUT (json path for generated token ids),
EQUAL_TO (json path of a reference token file; prints per-prompt exact-token equality), PROMPT_IDS (comma list
of prompt indices to run, default all).
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")

import torch

sys.path.insert(0, "/models/dsv41/DeepSeek-V4.1-Flash-1017-orig/encoding")
from encoding import encode_messages  # noqa: E402

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402
import qmesh.tp2_probe as probe  # noqa: E402

CHECKPOINT = os.environ.get("CHECKPOINT", "/models/dsv41/exl3-serve-tp2")
MAX_NEW = int(os.environ.get("MAX_NEW", "64"))
MAX_CHUNK = int(os.environ.get("MAX_CHUNK", "2048"))
TP_BACKEND = os.environ.get("TP_BACKEND", "native")
MOE_TENSOR_SPLIT = os.environ.get("MOE_TENSOR_SPLIT", "0") == "1"
IGNORE_EOS = os.environ.get("IGNORE_EOS", "1") == "1"
PROFILE = os.environ.get("PROFILE", "0") == "1"
TAG = os.environ.get("TAG", "bench")
TOKENS_OUT = os.environ.get("TOKENS_OUT")
EQUAL_TO = os.environ.get("EQUAL_TO")
DSPARK = os.environ.get("DSPARK") == "1"
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", "5"))
VERIFY_ROUNDS = [0]
DRAFT_CACHE_TOKENS = int(os.environ.get("DRAFT_CACHE_TOKENS", "4096"))
CACHE_TOKENS = int(os.environ.get("CACHE_TOKENS", "4096"))
DRAFT_DEVICE = os.environ.get("DRAFT_DEVICE", "cuda:1")
PROMPT_IDS = [int(i) for i in os.environ.get("PROMPT_IDS", "0,1,2").split(",")]

PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the nth Fibonacci number.",
    "List three primary colors and explain why they matter in painting.",
]


def rank_mem(model) -> str:
    """Peak allocated memory per TP rank. Rank 0 is in-process; spawned ranks are queried by dispatch."""
    dev0 = model.tp_output_device
    parts = [f"cuda:{dev0}={torch.cuda.max_memory_allocated(dev0) / 2**30:.2f}GiB"]
    for d in model.active_devices:
        if d == dev0:
            continue
        # tp_worker_dispatch_single drains pending acks and returns the child's single reply
        dev, g = model.tp_worker_dispatch_single(d, probe.peak_mem, ())
        parts.append(f"cuda:{dev}={g:.2f}GiB")
    return " ".join(parts)


def run_prompt(generator, tokenizer, prompt, max_new):
    text_in = encode_messages([{"role": "user", "content": prompt}], thinking_mode = "chat")
    ids = tokenizer.encode(text_in, add_bos = False, encode_special_tokens = True)
    job = Job(
        input_ids = ids,
        max_new_tokens = max_new,
        min_new_tokens = max_new if IGNORE_EOS else 0,
        sampler = GreedySampler(),
        stop_conditions = [] if IGNORE_EOS else [tokenizer.eos_token_id],
    )
    generator.enqueue(job)
    t0 = time.time()
    marks = []
    text = ""
    eos_info = None
    tokens = []
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            chunk = result.get("text", "")
            if chunk:
                text += chunk
                marks.append(time.time() - t0)
            if "token_ids" in result:
                tokens += result["token_ids"][0].tolist()
            if result.get("eos"):
                eos_info = result.get("eos_reason")
    total = time.time() - t0
    first = marks[0] if marks else float("nan")
    decode = (total - first) / max(len(marks) - 1, 1) if len(marks) > 1 else float("nan")
    stats = (getattr(job, "accepted_draft_tokens", 0), getattr(job, "rejected_draft_tokens", 0))
    return total, first, decode, len(marks), text, eos_info, tokens, stats


def main():
    config = Config.from_directory(CHECKPOINT)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = CACHE_TOKENS, max_history = 5)

    t0 = time.time()
    model.load(
        tensor_p = True,
        tp_output_device = 0,
        tp_backend = TP_BACKEND,
        tp_options = {"moe_tensor_split": MOE_TENSOR_SPLIT},
        max_batch_size = 1,
        max_chunk_size = MAX_CHUNK,
        max_output_size = MAX_NEW,
        progressbar = False,
    )
    load_s = time.time() - t0

    draft_model = None
    draft_cache = None
    if DSPARK:
        draft_config = Config.from_directory(CHECKPOINT)
        draft_model = Model.from_config(draft_config, component = "mtp")
        draft_cache = Cache(draft_model, max_num_tokens = DRAFT_CACHE_TOKENS, max_history = 5)
        draft_model.load(device = DRAFT_DEVICE, progressbar = False)
        print(f"[{TAG}] DSPARK_LOADED free_GiB_cuda0={torch.cuda.mem_get_info(0)[0] / 2**30:.2f}", flush=True)
    print(f"[{TAG}] TP_LOAD_OK {load_s:.1f}s backend={TP_BACKEND} moe_tensor_split={MOE_TENSOR_SPLIT} "
          f"max_alloc {rank_mem(model)}", flush=True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer, max_batch_size = 1, max_chunk_size = MAX_CHUNK,
                          draft_model = draft_model, draft_cache = draft_cache,
                          num_draft_tokens = NUM_DRAFT if DSPARK else None)
    if DSPARK:
        _orig_iter = generator.iterate_gen
        def _counted_iter(results, draft_tokens = None):
            if draft_tokens is not None:
                VERIFY_ROUNDS[0] += 1
            return _orig_iter(results, draft_tokens)
        generator.iterate_gen = _counted_iter
    ref = None
    if EQUAL_TO:
        with open(EQUAL_TO) as f:
            ref = json.load(f)

    decodes = []
    out = {}
    for idx in PROMPT_IDS:
        prompt = PROMPTS[idx]
        VERIFY_ROUNDS[0] = 0
        total, first, decode, n, text, eos_info, tokens, stats = run_prompt(generator, tokenizer, prompt, MAX_NEW)
        decodes.append(decode * 1e3)
        out[str(idx)] = tokens
        eq = ""
        if ref is not None:
            eq = f" tokens_equal_ref={tokens == ref[str(idx)]}"
        print(f"[{TAG}] PROMPT {prompt!r} total={total:.3f}s first_token_ms={first*1e3:.1f} "
              f"decode_ms_per_tok={decode*1e3:.2f} chunks={n} eos={eos_info}{eq} "
              f"accepted_draft_tokens={stats[0]} rejected_draft_tokens={stats[1]} verify_rounds={VERIFY_ROUNDS[0]} text={text!r}", flush=True)

    mean = sum(decodes) / len(decodes)
    print(f"[{TAG}] DECODE_MEAN_MS_PER_TOK {mean:.2f} over {len(decodes)} prompts "
          f"max_alloc {rank_mem(model)}", flush=True)
    if TOKENS_OUT:
        with open(TOKENS_OUT, "w") as f:
            json.dump(out, f)

    if PROFILE:
        from torch.profiler import profile, ProfilerActivity
        with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run_prompt(generator, tokenizer, PROMPTS[0], 16)
        print(f"[{TAG}] PROFILE rank0 key_averages (sort cuda_time_total, top 25):", flush=True)
        print(prof.key_averages().table(sort_by = "cuda_time_total", row_limit = 25), flush=True)
        trace_dir = os.environ.get("TRACE_DIR", "/tmp")
        prof.export_chrome_trace(os.path.join(trace_dir, f"tp2_{TAG}.json"))


if __name__ == "__main__":
    sys.exit(main())
