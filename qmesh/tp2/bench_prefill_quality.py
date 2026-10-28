"""Long-context TP2 quality capture and offline comparison.

Capture under the same two-rank launcher as bench_prefill.py, once per backend:
  QUALITY_OUT=/artifacts/baseline COMM=hybrid PREFILL_AR=0 P2P_RDV=/tmp/tp2-rdv python -m qmesh.tp2.bench_prefill_quality capture
  QUALITY_OUT=/artifacts/arm COMM=hybrid PREFILL_AR=int8 P2P_RDV=/tmp/tp2-rdv python -m qmesh.tp2.bench_prefill_quality capture
Then (no GPU required):
  python -m qmesh.tp2.bench_prefill_quality compare /artifacts/baseline /artifacts/arm --max-kl 0.001

The launcher must supply RANK, WORLD_SIZE and INIT_METHOD. Backend-specific
configuration is inherited from the environment. QUALITY_OUT is a filename
prefix; each rank writes PREFIX.rankN.pt. PROMPT_LEN=20000, MAX_NEW=128,
SAMPLES=8, CHUNK=2048, CACHE_TOKENS=prompt length + max new + 512 and
OVERLAY=/artifacts/tp2-overlay are configurable. SAMPLES evenly spaced
zero-based input positions include the last prompt token; each row is the full
vocabulary logits predicting the token after that position. Each sampled row
is obtained by an independent prefill of the corresponding prompt prefix, so
its final chunk may be shorter than the corresponding chunk in a full prefill.
The final row comes from the actual full-length greedy run. This avoids a
20k-by-vocabulary allocation. Compare checks both ranks, input identity, all
greedy tokens and KL(baseline || arm); --max-kl gates the worst sampled row.
"""
from __future__ import annotations

import argparse
import os

import torch


TEXTS = (
    "A historian describes the rise of coastal cities, their trade routes, and the decisions of their citizens. ",
    "Review this Python function: def merge(a, b): return sorted(a + b). Explain edge cases and complexity. ",
    "In a garden, a patient botanist observes the colors of leaves after rain and records each small change. ",
    "Given a triangle with sides a, b, c, derive useful bounds and explain every algebraic step carefully. ",
    "A spacecraft receives a sequence of signals; the crew compares timestamps, measurements, and uncertainty. ",
    "Translate the following ideas into clear prose: cooperation, evidence, disagreement, and a practical remedy. ",
)


def capture():
    import torch.distributed as dist
    from exllamav3 import Cache, Config, Generator, GreedySampler, Job, Model, Tokenizer
    from exllamav3.tp2.p2p import P2PComm
    from qmesh.tp2.run_tp2 import KV_CACHE_KWARGS, TPBackendNCCL, TPBackendP2P, TPBackendHybrid, set_tp_reduce, wrap_vocab_split

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    length = int(os.environ.get("PROMPT_LEN", "20000"))
    new = int(os.environ.get("MAX_NEW", "128"))
    samples = int(os.environ.get("SAMPLES", "8"))
    chunk = int(os.environ.get("CHUNK", "2048"))
    cache_tokens = int(os.environ.get("CACHE_TOKENS", str(length + new + 512)))
    if not (length >= 2 and 2 <= samples <= length and new > 0 and chunk > 0 and cache_tokens >= length + new):
        raise ValueError("Require length >= 2, 2 <= samples <= length, new/chunk > 0, and sufficient cache")
    positions = [i * (length - 1) // (samples - 1) for i in range(samples)]
    output = os.environ["QUALITY_OUT"]
    comm = os.environ.get("COMM", "nccl")
    if comm not in ("nccl", "p2p", "hybrid"):
        raise ValueError(f"Unknown COMM: {comm}")
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world, init_method=os.environ["INIT_METHOD"])
    config = Config.from_directory(os.path.join(os.environ.get("OVERLAY", "/artifacts/tp2-overlay"), f"rank{rank}"))
    if config.tp_size != world or config.tp_rank != rank:
        raise ValueError("Overlay rank/size does not match launcher")
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=cache_tokens, max_history=5, **KV_CACHE_KWARGS)
    model.load(device=torch.device("cuda", rank), max_chunk_size=chunk, max_output_size=8, max_batch_size=1, progressbar=False)
    if comm == "nccl":
        backend = TPBackendNCCL()
        vocab_backend = backend
        if os.environ.get("PREFILL_AR", "0") != "0":
            raise ValueError("PREFILL_AR requires COMM=p2p or hybrid")
    else:
        p2p = TPBackendP2P(P2PComm(rank, world=world, rendezvous_path=os.environ["P2P_RDV"], device=rank))
        backend = p2p if comm == "p2p" else TPBackendHybrid(TPBackendNCCL(), p2p)
        vocab_backend = backend
        from qmesh.tp2.prefill_ar import wrap_prefill_ar
        backend = wrap_prefill_ar(backend, p2p)
    set_tp_reduce(model, backend)
    wrap_vocab_split(model, rank, world, vocab_backend)
    tokenizer = Tokenizer.from_config(config)

    def run(ids, count):
        # A fresh generator prevents cached prefixes from bypassing the prefill under test.
        generator = Generator(model, cache, tokenizer, max_batch_size=1, max_chunk_size=chunk)
        job = Job(input_ids=ids, max_new_tokens=count, min_new_tokens=count,
                  sampler=GreedySampler(), stop_conditions=[], return_logits=True)
        generator.enqueue(job)
        tokens, first = [], None
        while generator.num_remaining_jobs():
            for result in generator.iterate():
                if "error" in result:
                    raise result["error"]
                if "token_ids" in result:
                    tokens.extend(result["token_ids"].reshape(-1).tolist())
                if first is None and "logits" in result:
                    first = result["logits"].reshape(-1, result["logits"].shape[-1])[0].float().cpu().clone()
        generator.close()
        if len(tokens) != count or first is None:
            raise RuntimeError(f"Expected {count} tokens and logits, got {len(tokens)}")
        return tokens, first

    records = []
    with torch.inference_mode():
        for index, text in enumerate(TEXTS):
            # Repeated, prompt-specific prose with a unique prefix gives deterministic,
            # independent contexts while retaining exact token length.
            ids = tokenizer.encode(f"Document {index}: " + text * (length // 8 + 8), add_bos=False, encode_special_tokens=True)
            if ids.shape[-1] < length:
                raise RuntimeError("Prompt did not tokenize to the requested length")
            ids = ids[:, :length].contiguous()
            rows = []
            for pos in positions[:-1]:
                _, row = run(ids[:, :pos + 1].contiguous(), 1)
                rows.append(row)
            tokens, row = run(ids, new)
            rows.append(row)
            records.append({"input_ids": ids.cpu(), "tokens": tokens, "logits": torch.stack(rows)})
            print(f"QUALITY rank={rank} prompt={index} positions={positions} tokens={len(tokens)}", flush=True)
    torch.save({"rank": rank, "world": world, "length": length, "new": new, "chunk": chunk,
                "positions": positions, "records": records}, f"{output}.rank{rank}.pt")
    dist.barrier()
    dist.destroy_process_group()


def compare(baseline, arm, max_kl):
    def load(prefix, rank):
        return torch.load(f"{prefix}.rank{rank}.pt", map_location="cpu", weights_only=True)

    base0, arm0 = load(baseline, 0), load(arm, 0)
    world = base0["world"]
    if world != 2 or arm0["world"] != world:
        raise ValueError("Quality gate requires two ranks in both captures")
    captures = [(base0, arm0)] + [(load(baseline, r), load(arm, r)) for r in range(1, world)]
    worst = 0.0
    identity = True
    for rank, (base, candidate) in enumerate(captures):
        for data in (base, candidate):
            if data["rank"] != rank or data["world"] != world or len(data["records"]) != 6:
                raise ValueError("Invalid rank or prompt count")
        for key in ("length", "new", "chunk", "positions"):
            if base[key] != candidate[key] or base[key] != base0[key]:
                raise ValueError(f"Capture mismatch: {key}")
        for i, (a, b) in enumerate(zip(base["records"], candidate["records"])):
            if not torch.equal(a["input_ids"], b["input_ids"]) or not torch.equal(a["input_ids"], base0["records"][i]["input_ids"]):
                raise ValueError(f"Input mismatch rank={rank} prompt={i}")
            if a["logits"].shape != b["logits"].shape or a["logits"].shape[0] != len(base["positions"]):
                raise ValueError("Logit shape mismatch")
            if not torch.isfinite(a["logits"]).all() or not torch.isfinite(b["logits"]).all():
                raise ValueError("Non-finite logits")
            if len(a["tokens"]) != base["new"] or len(b["tokens"]) != base["new"]:
                raise ValueError("Incomplete token sequence")
            same = a["tokens"] == b["tokens"]
            rank_same = (a["tokens"] == base0["records"][i]["tokens"] and
                         b["tokens"] == arm0["records"][i]["tokens"])
            identity &= same and rank_same
            logp = torch.log_softmax(a["logits"].double(), dim=-1)
            logq = torch.log_softmax(b["logits"].double(), dim=-1)
            kl = (logp.exp() * (logp - logq)).sum(dim=-1)
            worst = max(worst, kl.max().item())
            for position, value in zip(base["positions"], kl.tolist()):
                print(f"rank={rank} prompt={i} position={position} kl={value:.9g}")
            print(f"rank={rank} prompt={i} baseline_arm_tokens_equal={same} ranks_equal={rank_same}")
    passed = identity and worst <= max_kl
    print(f"RESULT {'PASS' if passed else 'FAIL'} max_kl={worst:.9g} limit={max_kl} token_identity={identity}")
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("capture")
    cmp = sub.add_parser("compare")
    cmp.add_argument("baseline")
    cmp.add_argument("arm")
    cmp.add_argument("--max-kl", type=float, required=True, help="Maximum permitted per-position KL(baseline || arm)")
    args = parser.parse_args()
    if args.command == "capture":
        capture()
    else:
        if not 0 <= args.max_kl < float("inf"):
            parser.error("--max-kl must be finite and nonnegative")
        raise SystemExit(compare(args.baseline, args.arm, args.max_kl))


if __name__ == "__main__":
    main()
