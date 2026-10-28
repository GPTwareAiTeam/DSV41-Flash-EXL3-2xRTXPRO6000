"""Compare cold and recurrent prefix-cache generation in one TP2 process group.

Launch with the same rank environment as bench_prefill (RANK, WORLD_SIZE,
INIT_METHOD, and P2P_RDV for p2p/hybrid). Defaults: PROMPT_LENS=2048,8192,
CHUNK=512, NEW_TOKENS=32, CACHE_TOKENS=max(PROMPT_LENS)+512. OVERLAY and
COMM have the same meanings as in bench_prefill. EXL3_PREFILL_RC_SPLIT defaults
on so the last complete prompt page can receive a recurrent checkpoint.
TTFT runs from enqueue to the first emitted token, synchronized on each rank;
effective prompt tok/s uses the entire prompt length, including cached tokens.
"""
from __future__ import annotations

import os
import random
import subprocess
import threading
import time

os.environ.setdefault("EXL3_PREFILL_RC_SPLIT", "1")
os.environ.setdefault("EXL3_PREFIX_CACHE", "1")
import torch
import torch.distributed as dist

from qmesh.tp2.bench_prefill import (
    Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer, P2PComm,
    KV_CACHE_KWARGS, TPBackendNCCL, TPBackendP2P, TPBackendHybrid,
    set_tp_reduce, wrap_vocab_split, make_ids, gpu_mem_line, OVERLAY, COMM, RANK, WORLD,
)
from qmesh.tp2.run_tp2 import PREFIX_CACHE, PREFIX_CACHE_CAPACITY, PREFIX_CACHE_MIN_MATCH, PREFIX_CACHE_INTERVAL

PROMPT_LENS = [int(x) for x in os.environ.get("PROMPT_LENS", "2048,8192").split(",")]
CHUNK = int(os.environ.get("CHUNK", "512"))
NEW_TOKENS = int(os.environ.get("NEW_TOKENS", "32"))
CACHE_TOKENS = int(os.environ.get("CACHE_TOKENS", str(max(PROMPT_LENS) + 512)))


def clocks():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,clocks.sm,temperature.gpu,power.draw,clocks_throttle_reasons.active",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20, check=True
        )
        return ";".join(line.replace(" ", "") for line in result.stdout.strip().splitlines())
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable({exc})"


def run(generator, ids, label):
    job = Job(input_ids=ids, max_new_tokens=NEW_TOKENS, min_new_tokens=NEW_TOKENS,
              sampler=GreedySampler(), stop_conditions=[])
    dist.barrier()
    torch.cuda.synchronize()
    start = time.perf_counter()
    generator.enqueue(job)
    telemetry = []
    thread = threading.Thread(target=lambda: telemetry.append(clocks()))
    thread.start()
    first = None
    tokens = []
    final = None
    while generator.num_remaining_jobs():
        for result in generator.iterate():
            if "error" in result:
                raise result["error"]
            if "token_ids" in result:
                if first is None:
                    torch.cuda.synchronize()
                    first = time.perf_counter()
                tokens.extend(result["token_ids"].reshape(-1).tolist())
                last = time.perf_counter()
            if result.get("eos"):
                final = result
    thread.join()
    assert first is not None and final is not None and len(tokens) == NEW_TOKENS
    elapsed = first - start
    rc = generator.recurrent_cache
    rc_bytes = rc.update_total_size() if rc is not None else 0
    print(f"[PREFIX{RANK}] {label} len={ids.shape[-1]} ttft_ms={elapsed * 1000:.2f} "
          f"decode_ms_per_tok={(last - first) * 1000 / (len(tokens) - 1):.2f} effective_prompt_tok_s={ids.shape[-1] / elapsed:.1f} "
          f"cached_pages={final['cached_pages']} cached_tokens={final['cached_tokens']} "
          f"recurrent_checkpoint_bytes={rc_bytes} recurrent_checkpoints={len(rc) if rc is not None else 0} "
          f"gpu_telemetry[index,sm_mhz,temp_c,power_w,throttle_active]={telemetry[0]} tokens={tokens}", flush=True)
    return tokens


def main():
    torch.cuda.set_device(RANK)
    dist.init_process_group(backend="nccl", rank=RANK, world_size=WORLD,
                            init_method=os.environ["INIT_METHOD"])
    config = Config.from_directory(os.path.join(OVERLAY, f"rank{RANK}"))
    assert config.tp_size == WORLD and config.tp_rank == RANK
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=CACHE_TOKENS, max_history=5, **KV_CACHE_KWARGS)
    model.load(device=torch.device("cuda", RANK), max_chunk_size=CHUNK, max_output_size=8,
               max_batch_size=1, progressbar=False)
    if COMM in ("p2p", "hybrid"):
        p2p = TPBackendP2P(P2PComm(RANK, world=WORLD, rendezvous_path=os.environ["P2P_RDV"], device=RANK))
        backend = p2p if COMM == "p2p" else TPBackendHybrid(TPBackendNCCL(), p2p)
    else:
        backend = TPBackendNCCL()
    set_tp_reduce(model, backend)
    wrap_vocab_split(model, RANK, WORLD, backend)
    tokenizer = Tokenizer.from_config(config)
    kv_bytes = sum(layer.storage_size() for layer in cache.layers.values())
    print(f"[PREFIX{RANK}] loaded kv_cache_storage_bytes={kv_bytes} "
          f"chunk={CHUNK} cache_tokens={CACHE_TOKENS} comm={COMM} {gpu_mem_line()}", flush=True)

    def new_generator():
        generator = Generator(model, cache, tokenizer, max_batch_size=1,
                              max_chunk_size=CHUNK,
                              recurrent_cache_size=PREFIX_CACHE_CAPACITY if PREFIX_CACHE else 0,
                              recurrent_checkpoint_interval=PREFIX_CACHE_INTERVAL,
                              recurrent_checkpoint_interval_pp=PREFIX_CACHE_INTERVAL)
        generator.prefix_cache_min_match = PREFIX_CACHE_MIN_MATCH if PREFIX_CACHE else None
        return generator

    random.seed(42)
    for n in PROMPT_LENS:
        for rep in range(2):
            original = make_ids(tokenizer, n, n * 10 + rep)
            alternate = make_ids(tokenizer, n, n * 10 + rep + 100000)
            # A different token at the midpoint breaks the page-hash chain there.
            partial = original.clone()
            partial[:, n // 2:] = alternate[:, n // 2:]
            partial[:, n // 2] = 0 if original[0, n // 2].item() != 0 else 1

            generator = new_generator()
            run(generator, make_ids(tokenizer, n, n * 10 + rep + 200000), f"rep={rep} warm")
            cold = run(generator, original, f"rep={rep} full_cold")
            hit = run(generator, original, f"rep={rep} full_hit")
            print(f"[PREFIX{RANK}] full len={n} rep={rep} token_equal={cold == hit}", flush=True)
            generator.close()
            if cold != hit:
                raise RuntimeError(f"full cache token mismatch for len={n} rep={rep}")

            # The partial reference must be cold, and must not prime the later partial hit.
            generator = new_generator()
            partial_cold = run(generator, partial, f"rep={rep} partial_cold")
            generator.close()
            generator = new_generator()
            run(generator, original, f"rep={rep} partial_prime")
            partial_hit = run(generator, partial, f"rep={rep} partial_hit")
            print(f"[PREFIX{RANK}] partial len={n} rep={rep} shared_tokens={n // 2} "
                  f"token_equal={partial_cold == partial_hit}", flush=True)
            generator.close()
            if partial_cold != partial_hit:
                raise RuntimeError(f"partial cache token mismatch for len={n} rep={rep}")
        generator = new_generator()
        base = make_ids(tokenizer, n, n * 100 + 1)
        other = make_ids(tokenizer, n, n * 100 + 2)
        run(generator, base, "alternate_a_cold")
        run(generator, other, "alternate_b_cold")
        run(generator, base, "alternate_a_hit")
        run(generator, other, "alternate_b_hit")
        generator.close()
        generator = new_generator()
        grown = base
        for turn in range(4):
            run(generator, grown, f"growth_turn={turn}")
            suffix = make_ids(tokenizer, 1000, n * 1000 + turn)
            grown = torch.cat((grown, suffix), dim=1)
        generator.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
