"""TP=2 per-rank runner for the pre-sharded DeepSeek-V4.1 checkpoint (qmesh/tp2).

One process per GPU (RANK / WORLD_SIZE / INIT_METHOD env). Each rank loads ONLY its rank dir from
the overlay (rank shards + engram-only tables), builds the arch with local TP dims, and runs a
greedy decode. Collectives: torch.distributed NCCL all_reduce (SUM) for attention/MoE partials and the
vocab-split embedding; head logits (local 64640) are all-gathered to the full 129280 on both ranks.

Env: OVERLAY (dir holding rank0/ rank1/), PROMPT, MAX_NEW, IGNORE_EOS (0|1), INIT_METHOD (file://...).
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")

import torch
import torch.distributed as dist

sys.path.insert(0, "/models/dsv41/DeepSeek-V4.1-Flash-1017-orig/encoding")
from encoding import encode_messages  # noqa: E402

from exllamav3 import Cache, Config, Model, Tokenizer  # noqa: E402

OVERLAY = os.environ["OVERLAY"]
PROMPT = os.environ.get("PROMPT", "Say hello in one short sentence.")
MAX_NEW = int(os.environ.get("MAX_NEW", "32"))
IGNORE_EOS = os.environ.get("IGNORE_EOS", "0") == "1"
RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
TAG = f"TP2R{RANK}"


def log(msg: str):
    print(f"[{TAG}] {msg}", flush=True)


class TPBackendNCCL:
    """Thin collective backend for the modules' params["backend"] hook (torch.distributed NCCL)."""

    def fwd_barrier(self):
        dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        # Non-contributing ranks reduce a zero stand-in so the collective stays aligned
        if not contribution:
            tensor.zero_()
        dist.all_reduce(tensor, op = dist.ReduceOp.SUM)

    def broadcast(self, tensor: torch.Tensor, src_device: int):
        dist.broadcast(tensor, src = src_device)

    def run_cpu_reduce_jobs(self):
        pass

    def end_cpu_reduce_jobs(self):
        pass


def set_tp_reduce(model: Model, backend):
    """Turn on the attention/MoE partial-sum all-reduce for every TP-local block."""
    from exllamav3.modules.transformer import TransformerBlock
    n = 0
    for m in model.modules:
        if isinstance(m, TransformerBlock):
            m.attn.tp_reduce = True
            m.mlp.tp_reduce = True
            n += 1
    return n


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
    model.load(device = device, progressbar = False)
    backend = TPBackendNCCL()
    n = set_tp_reduce(model, backend)
    cache = Cache(model, max_num_tokens = 4096)
    log(f"LOADED {time.time() - t0:.1f}s blocks={n} alloc_GiB={torch.cuda.memory_allocated(device) / 2**30:.2f}")

    tokenizer = Tokenizer.from_config(config)
    text_in = encode_messages([{"role": "user", "content": PROMPT}], thinking_mode = "chat")
    ids = tokenizer.encode(text_in, add_bos = False, encode_special_tokens = True)
    log(f"PROMPT_TOKENS {ids.shape[-1]}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
