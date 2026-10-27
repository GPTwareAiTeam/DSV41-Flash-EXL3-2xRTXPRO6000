"""TP2 prefill benchmark: time-to-first-token of synthetic prompts, per prompt length and chunk size.

Launch like the runner (qmesh/tp2/launch_tp2.sh, one process per rank). Env:
  PROMPT_LENS   (256,1024,2048,4096)  exact prompt token counts
  CHUNKS        (512,2048,0)          Generator max_chunk_size per run; 0 = the prompt length itself
  REPS          (2)                   timed repetitions per (length, chunk), after one warm pass
  MAX_CHUNK     (max of CHUNKS and PROMPT_LENS, else 4096)  model load-time chunk ceiling
  CACHE_TOKENS  (max prompt + 512)    KV cache tokens
  COMM          nccl | p2p (P2P_RDV required for p2p)

Timing: time from generator.enqueue() to the first token result, excluding model load. Each trial uses a unique
prefix so the prefix/page cache cannot shortcut the prefill. Prompts are exact token counts (ids are truncated
to N after tokenization). Every rank prints its own time; the wall is the max over ranks (ranks sync via barrier).
Prints: BENCH len=N chunk=C rep=R rank=r ms=... tok_s=... ; memory via nvidia-smi and torch on each rank.
"""
from __future__ import annotations

import os
import random
import subprocess
import sys
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", "/root/.cache/torch_extensions")

import torch
import torch.distributed as dist

sys.path.insert(0, "/models/dsv41/DeepSeek-V4.1-Flash-1017-orig/encoding")

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer  # noqa: E402
from exllamav3.tp2.p2p import P2PComm  # noqa: E402
from qmesh.tp2.run_tp2 import TPBackendNCCL, TPBackendP2P, TPBackendHybrid, set_tp_reduce, wrap_vocab_split  # noqa: E402

OVERLAY = os.environ.get("OVERLAY", "/artifacts/tp2-overlay")
PROMPT_LENS = [int(x) for x in os.environ.get("PROMPT_LENS", "256,1024,2048,4096").split(",")]
CHUNKS = [int(x) for x in os.environ.get("CHUNKS", "512,2048,0").split(",")]
REPS = int(os.environ.get("REPS", "2"))
COMM = os.environ.get("COMM", "nccl")
PROFILE_LEN = int(os.environ.get("PROFILE_LEN", "0"))
PROFILE_CHUNK = int(os.environ.get("PROFILE_CHUNK", "0"))
RANK = int(os.environ["RANK"])
WORLD = int(os.environ["WORLD_SIZE"])
TAG = f"BENCH{RANK}"
FILLER = "The quick brown fox jumps over the lazy dog near the quiet riverbank as the sun sets. "

MAX_CHUNK = int(os.environ.get("MAX_CHUNK", str(max([c for c in CHUNKS if c] + PROMPT_LENS))))
CACHE_TOKENS = int(os.environ.get("CACHE_TOKENS", str(max(PROMPT_LENS) + 512)))


def log(msg: str):
    print(f"[{TAG}] {msg}", flush=True)


def gpu_mem_line() -> str:
    """nvidia-smi per-GPU used memory (device-wide, all processes) plus this rank's torch peak."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output = True, text = True, timeout = 20).stdout
        smi = ";".join(line.replace(" ", "") for line in out.strip().splitlines())
    except Exception as e:  # nvidia-smi missing in the container should not kill the bench
        smi = f"unavailable({e})"
    dev = torch.cuda.current_device()
    return (f"smi_used_MiB[{smi}] torch_alloc_GiB={torch.cuda.memory_allocated(dev) / 2**30:.2f} "
            f"torch_peak_alloc_GiB={torch.cuda.max_memory_allocated(dev) / 2**30:.2f} "
            f"torch_peak_reserved_GiB={torch.cuda.max_memory_reserved(dev) / 2**30:.2f}")


def make_ids(tokenizer, n: int, trial: int) -> torch.Tensor:
    """Exactly n tokens: a unique header (defeats prefix cache) then repeated filler, truncated to n."""
    text = f"Benchmark run {trial}-{random.getrandbits(32)}. " + FILLER * (n // 8 + 8)
    ids = tokenizer.encode(text, add_bos = False, encode_special_tokens = True)
    assert ids.shape[-1] >= n, f"filler too short: {ids.shape[-1]} < {n}"
    return ids[:, :n].contiguous()


def time_first_token(generator, ids: torch.Tensor) -> float:
    job = Job(input_ids = ids, max_new_tokens = 1, min_new_tokens = 1, sampler = GreedySampler(),
              stop_conditions = [])
    dist.barrier()
    torch.cuda.synchronize()
    t0 = time.time()
    generator.enqueue(job)
    t_first = None
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if "error" in result:
                raise result["error"]
            if "token_ids" in result and t_first is None:
                torch.cuda.synchronize()
                t_first = time.time()
    assert t_first is not None, "no token emitted"
    return t_first - t0


def main():
    torch.cuda.set_device(RANK)
    dist.init_process_group(backend = "nccl", rank = RANK, world_size = WORLD,
                            init_method = os.environ["INIT_METHOD"])
    device = torch.device("cuda", RANK)
    config = Config.from_directory(os.path.join(OVERLAY, f"rank{RANK}"))
    assert config.tp_size == WORLD and config.tp_rank == RANK
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = CACHE_TOKENS, max_history = 5)
    model.load(device = device, max_chunk_size = MAX_CHUNK, max_output_size = 8, max_batch_size = 1,
               progressbar = False)
    if COMM in ("p2p", "hybrid"):
        p2p = TPBackendP2P(P2PComm(RANK, world = WORLD, rendezvous_path = os.environ["P2P_RDV"], device = RANK))
        backend = p2p if COMM == "p2p" else TPBackendHybrid(TPBackendNCCL(), p2p)
    else:
        backend = TPBackendNCCL()
    set_tp_reduce(model, backend)
    wrap_vocab_split(model, RANK, WORLD, backend)
    tokenizer = Tokenizer.from_config(config)
    log(f"LOADED MAX_CHUNK={MAX_CHUNK} CACHE_TOKENS={CACHE_TOKENS} COMM={COMM} {gpu_mem_line()}")

    trial = 0
    for chunk in CHUNKS:
        for n in PROMPT_LENS:
            c = chunk if chunk else n
            if c > MAX_CHUNK:
                log(f"SKIP len={n} chunk={c} > MAX_CHUNK={MAX_CHUNK}")
                continue
            generator = Generator(model, cache, tokenizer, max_batch_size = 1, max_chunk_size = c)
            # warm pass (graph capture / allocator) on a different unique prompt of the same length
            trial += 1
            time_first_token(generator, make_ids(tokenizer, n, trial))
            times = []
            for rep in range(REPS):
                trial += 1
                dt = time_first_token(generator, make_ids(tokenizer, n, trial))
                times.append(dt)
                log(f"BENCH len={n} chunk={c} rep={rep} ms={dt * 1e3:.1f} tok_s={n / dt:.1f}")
            best = min(times)
            log(f"SUMMARY len={n} chunk={c} best_ms={best * 1e3:.1f} best_tok_s={n / best:.1f} "
                f"median_ms={sorted(times)[len(times) // 2] * 1e3:.1f}")
            generator.close()
            del generator
            torch.cuda.empty_cache()
            if RANK == 0:
                log(f"MEM len={n} chunk={c} {gpu_mem_line()}")

    if PROFILE_LEN:
        from torch.profiler import profile, ProfilerActivity
        generator = Generator(model, cache, tokenizer, max_batch_size = 1, max_chunk_size = PROFILE_CHUNK or PROFILE_LEN)
        time_first_token(generator, make_ids(tokenizer, PROFILE_LEN, 10_000))
        trial_ids = make_ids(tokenizer, PROFILE_LEN, 10_001)
        with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            dt = time_first_token(generator, trial_ids)
        log(f"PROFILE len={PROFILE_LEN} chunk={PROFILE_CHUNK or PROFILE_LEN} ttft_ms={dt * 1e3:.1f}")
        if RANK == 0:
            prof.export_chrome_trace(os.environ.get("PROFILE_TRACE", "/artifacts/tp2logs/prefill_trace.json"))
            print(prof.key_averages().table(sort_by = "self_cuda_time_total", row_limit = 40), flush = True)
        generator.close()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
