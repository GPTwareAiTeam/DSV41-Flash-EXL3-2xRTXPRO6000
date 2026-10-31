#!/usr/bin/env python3
"""Multi-prompt P7-hot + GX1-cold end-to-end driver over the native RDMA seam.

Generalizes qmesh/e2e_seam.py: reads a prompt list (JSON array of strings, or a text file with
one prompt per record separated by a blank line), runs greedy decoding, and prints one JSON
line per prompt plus a summary. Timing comes from the engine's own job accounting
(time_enqueued / time_prefill / time_generate / new_tokens) and from wall clock.

Env:
  CHECKPOINT   model dir (default /models/dsv41/exl3-serve-final)
  PLACEMENT    placement json (default qmesh/placement.final.json)
  PROMPTS      path to prompt list (JSON array)  [default: built-in 3 prompts]
  PROMPTS_TXT  path to a text file; each non-empty line is one prompt (overrides PROMPTS)
  CORPUS       comma-separated corpus txt files; each file is truncated to CORPUS_TOKENS tokens
  CORPUS_TOKENS default 24000
  MAX_NEW      new tokens per prompt (default 256)
  MAX_CHUNK    prefill chunk (default 2048)
  DSPARK       1 to enable the DSpark draft model
  NUM_DRAFT    draft width K (default 5)
  WARMUP       number of prompts used as warm-up and excluded from the summary (default 0)
  OUT          jsonl output path (also echoed to stdout)
  MAX_PROMPTS  cap the prompt list (default 0 = all)
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("EXL3_MOE_SPLIT_FUSED", "0")
os.environ.setdefault("EXL3_MOE_CPU_SWAP", "0")

import torch

from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer

CHECKPOINT = os.environ.get("CHECKPOINT", "/models/dsv41/exl3-serve-final")
PLACEMENT = os.environ.get("PLACEMENT", os.path.join(os.path.dirname(__file__), "placement.final.json"))
MAX_NEW = int(os.environ.get("MAX_NEW", "256"))
MAX_CHUNK = int(os.environ.get("MAX_CHUNK", "2048"))
DSPARK = os.environ.get("DSPARK") == "1"
NUM_DRAFT = int(os.environ.get("NUM_DRAFT", "5"))
WARMUP = int(os.environ.get("WARMUP", "0"))
OUT = os.environ.get("OUT", "")
CORPUS_TOKENS = int(os.environ.get("CORPUS_TOKENS", "24000"))
DEFAULT_PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the nth Fibonacci number.",
    "List three primary colors and explain why they matter in painting.",
]


def load_prompts():
    """Returns a list of (label, prompt) pairs."""
    txt = os.environ.get("PROMPTS_TXT") or ""
    corpus = os.environ.get("CORPUS") or ""
    if corpus:
        out = []
        for path in corpus.split(","):
            path = path.strip()
            if not path:
                continue
            text = open(path, encoding="utf-8", errors="replace").read()
            out.append((os.path.basename(path), text[: CORPUS_TOKENS * 5]))
        return out
    if txt:
        text = open(txt, encoding="utf-8", errors="replace").read()
        recs = [r.strip() for r in text.split("\n\n") if r.strip()]
        return [(f"line{i}", r) for i, r in enumerate(recs)]
    path = os.environ.get("PROMPTS", "")
    if path:
        data = json.load(open(path))
        if isinstance(data, dict):
            data = [v for k, v in sorted(data.items())]
        return [(f"p{i}", p if isinstance(p, str) else p.get("prompt", ""))
                for i, p in enumerate(data)]
    return [(f"p{i}", p) for i, p in enumerate(DEFAULT_PROMPTS)]


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
    t0 = time.time()
    model.load(device="cuda:0", progressbar=False)
    free, total = torch.cuda.mem_get_info(0)
    print(f"P7_LOAD_OK {time.time() - t0:.1f}s "
          f"vram={torch.cuda.memory_allocated(0) / 2**30:.2f}GiB "
          f"free={free / 2**30:.2f}GiB total={total / 2**30:.2f}GiB", flush=True)

    tokenizer = Tokenizer.from_config(config)
    draft_model = None
    if DSPARK:
        draft_config = Config.from_directory(CHECKPOINT)
        draft_model = Model.from_config(draft_config, component="mtp")
        t1 = time.time()
        draft_model.load(device="cuda:0", progressbar=False)
        print(f"DSPARK_LOADED {time.time() - t1:.1f}s", flush=True)

    prompts = load_prompts()[: int(os.environ.get("MAX_PROMPTS", "0")) or None]
    # Warm-up prompts are synthetic, excluded from the summary, and never touch the prompt list
    warm = ["Write a paragraph about the history of sailing ships and their navigation instruments."] * WARMUP
    prompts = [(f"warm{i}", p) for i, p in enumerate(warm)] + prompts
    fh = open(OUT, "a", encoding="utf-8") if OUT else None
    rows = []
    for idx, (label, prompt) in enumerate(prompts):
        ids = tokenizer.encode(prompt, add_bos=True, encode_special_tokens=True)
        cache = Cache(model, max_num_tokens=len(ids) + MAX_NEW + 64, max_history=2)
        draft_cache = Cache(draft_model, max_num_tokens=len(ids) + MAX_NEW + 64, max_history=2) \
            if draft_model is not None else None
        generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=MAX_CHUNK,
                              draft_model=draft_model, draft_cache=draft_cache,
                              num_draft_tokens=NUM_DRAFT if DSPARK else None)
        job = Job(input_ids=ids, max_new_tokens=MAX_NEW, sampler=GreedySampler())
        generator.enqueue(job)
        t0 = time.time()
        text = ""
        chunks = 0
        eos_result = None
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                chunk = result.get("text", "")
                if chunk:
                    text += chunk
                    chunks += 1
                if result.get("eos"):
                    eos_result = result
        wall = time.time() - t0
        st = eos_result or {}
        new_tokens = st.get("new_tokens") or chunks
        prefill_s = (st.get("time_enqueued", 0.0) + st.get("time_prefill", 0.0)) or float("nan")
        gen_s = st.get("time_generate", float("nan"))
        decode_ms = 1000.0 * gen_s / max(new_tokens - 1, 1)
        acc = st.get("accepted_draft_tokens", 0)
        rej = st.get("rejected_draft_tokens", 0)
        rounds = getattr(job, "verify_rounds", 0)
        toks_per_round = (new_tokens / rounds) if rounds else None
        free, total = torch.cuda.mem_get_info(0)
        row = {
            "i": idx, "label": label, "prompt_tokens": len(ids), "new_tokens": new_tokens,
            "ttft_ms": round(1000.0 * prefill_s, 1),
            "decode_ms_per_tok": round(decode_ms, 2),
            "wall_s": round(wall, 3),
            "accepted": acc, "rejected": rej, "verify_rounds": rounds,
            "accept_rate": round(acc / max(acc + rej, 1), 4) if (acc or rej) else None,
            "tokens_per_round": round(toks_per_round, 3) if toks_per_round else None,
            "gpu0_free_gib": round(free / 2**30, 2),
            "text": text[:400],
        }
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
        if idx >= WARMUP:
            rows.append(row)
        del generator, job, cache, draft_cache
        torch.cuda.empty_cache()

    def med(key):
        vals = sorted(r[key] for r in rows if r.get(key) is not None)
        return vals[len(vals) // 2] if vals else None

    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "n": len(rows), "dspark": DSPARK, "num_draft": NUM_DRAFT if DSPARK else None,
        "max_new": MAX_NEW,
        "decode_ms_per_tok_mean": mean("decode_ms_per_tok"),
        "decode_ms_per_tok_median": med("decode_ms_per_tok"),
        "ttft_ms_mean": mean("ttft_ms"),
        "tokens_per_round_mean": mean("tokens_per_round"),
        "accept_rate_mean": mean("accept_rate"),
        "gpu0_free_gib_min": min((r["gpu0_free_gib"] for r in rows), default=None),
        "tokens": sum(r["new_tokens"] for r in rows),
        "wall_s": round(sum(r["wall_s"] for r in rows), 2),
    }
    print("SUMMARY " + json.dumps(summary), flush=True)
    if fh:
        fh.write("SUMMARY " + json.dumps(summary) + "\n")
        fh.close()


if __name__ == "__main__":
    main()
