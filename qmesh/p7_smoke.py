#!/usr/bin/env python3
"""P7 hot-expert + GX1 cold-expert smoke test."""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")

import torch

from exllamav3 import Cache, Config, Generator, Model, Tokenizer

CHECKPOINT = "/models/dsv41/exl3-serve"
PLACEMENT = os.path.join(os.path.dirname(__file__), "placement.exl3.json")
REMOTE_HOST = os.environ.get("REMOTE_HOST", "10.0.100.2")
REMOTE_PORT = int(os.environ.get("REMOTE_PORT", "29517"))


def main():
    placement = json.load(open(PLACEMENT))
    hot = {f"layers.{layer}.ffn": v["hot"] for layer, v in placement["layers"].items()}
    config = Config.from_directory(CHECKPOINT)
    config.infer_params.moe_cpu_split = 1
    config.infer_params.moe_remote_hot = hot
    config.infer_params.remote_expert_host = {"host": REMOTE_HOST, "port": REMOTE_PORT}
    model = Model.from_config(config)
    t0 = time.time()
    model.load(device="cuda:0", progressbar=False)
    print(f"P7_LOAD_OK {time.time() - t0:.1f}s vram={torch.cuda.memory_allocated(0)/2**30:.2f}GiB", flush=True)

    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=2048)
    t0 = time.time()
    text = generator.generate(
        prompt="Reply with exactly one word: pong",
        max_new_tokens=8,
        completion_only=True,
        add_bos=True,
        encode_special_tokens=True,
    )
    print(f"SMOKE_OK {time.time() - t0:.2f}s text={text!r}", flush=True)


if __name__ == "__main__":
    main()
