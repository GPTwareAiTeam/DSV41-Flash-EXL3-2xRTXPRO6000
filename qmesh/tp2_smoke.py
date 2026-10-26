#!/usr/bin/env python3
"""TP2/EP2 smoke test: DeepSeek-V4.1-Flash EXL3 2.565 on GPU0+GPU1 with external Engram tables.

Runs on the checkpoint overlay /models/dsv41/exl3-serve-tp2 (symlinked shards, index with
engram embed keys mapped to the engram-only shards). Not run by the TP2Prep worker: GPUs are
owned by peers until released.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")

import torch

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer

CHECKPOINT = os.environ.get("CHECKPOINT", "/models/dsv41/exl3-serve-tp2")
MAX_NEW = int(os.environ.get("MAX_NEW", "32"))
MAX_CHUNK = int(os.environ.get("MAX_CHUNK", "2048"))
MAX_OUT = int(os.environ.get("MAX_OUT", "32"))
PROMPT = os.environ.get("PROMPT", "Hi")
TP_DEVICES = (0, 1)


def gib(dev: int) -> float:
    return torch.cuda.max_memory_allocated(dev) / 2**30


def main():
    config = Config.from_directory(CHECKPOINT)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=4096, max_history=5)

    t0 = time.time()
    model.load(
        tensor_p = True,
        tp_output_device = 0,
        tp_backend = "native",
        tp_options = {"moe_tensor_split": False},
        max_batch_size = 1,
        max_chunk_size = MAX_CHUNK,
        max_output_size = MAX_OUT,
        progressbar = False,
    )
    load_s = time.time() - t0
    mem = " ".join(f"cuda:{d}={gib(d):.2f}GiB" for d in TP_DEVICES)
    print(f"TP2_LOAD_OK {load_s:.1f}s max_alloc {mem}", flush=True)

    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=MAX_CHUNK)
    ids = tokenizer.encode(PROMPT, add_bos = True, encode_special_tokens = True)
    job = Job(input_ids = ids, max_new_tokens = MAX_NEW, min_new_tokens = MAX_NEW,
              sampler = GreedySampler())
    generator.enqueue(job)

    t0 = time.time()
    marks = []
    text = ""
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            chunk = result.get("text", "")
            if chunk:
                text += chunk
                marks.append(time.time() - t0)
    total = time.time() - t0
    first = marks[0] if marks else float("nan")
    decode = (total - first) / max(len(marks) - 1, 1) if len(marks) > 1 else float("nan")
    mem = " ".join(f"cuda:{d}={gib(d):.2f}GiB" for d in TP_DEVICES)
    print(f"TP2_SMOKE_OK total={total:.3f}s first_token_ms={first*1e3:.1f} "
          f"decode_ms_per_tok={decode*1e3:.2f} chunks={len(marks)} max_alloc {mem} "
          f"text={text!r}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
