"""Teacher-forced perplexity for the TP=2 DeepSeek-V4.1 build (qmesh/tp2).

Scores a fixed token stream one position at a time (no sampling, no speculation), on the same
weights/collectives the generator uses. Two protocols share one loaded model:

  wiki2   llama.cpp perplexity.c, as eval/ppl.py's GGUF-equivalent mode implements it: the raw
          wikitext-2-raw-v1 test text, tokenized without special tokens, cut into non-overlapping
          n_ctx (2048) chunks, the first token of each chunk replaced by BOS, and the NLL of logits
          positions [n_ctx/2, n_ctx-1) taken against the following tokens.
  panel   the sudoingX/dgx-spark-deepseek-v41 quality panel, reproduced from its own scripts:
          scripts/quantize/caldata/build_panel.py renders each conversation with the checkpoint's own
          encoder (tools on a system message, thinking/chat mode, reasoning effort), truncates to
          --panel-max tokens and marks assistant-generated positions; scripts/quantize/p4_kld.py then
          scores position t against token t+1 on marked positions only. Only NLL is reproduced: the
          published data/kld/*.json carry summary statistics with "tensors": "", so the reference
          logits needed for KLD are not published, but nll_native is, and perplexity is comparable.

Scoring runs through the production cached path (`--attn cached`, the default): one generator prefill per
sequence with a prefill hook that runs the full stack and the head over every prompt row, i.e. exactly
the kernels a live request uses. `--attn nc` instead uses the stateless flash_attn_nc single pass; on
this TP2 build that path is NOT prefix-stable (the logits for a given token at a given position move
with the trailing chunk length), so its numbers are A/B diagnostics only, never quality numbers.

Both ranks run every forward on identical ids, because the TP attention/MoE/head collectives require
it; rank 0 does the scoring arithmetic and writes the result file.

Usage (launched like the runner, one process per GPU):
  RANK=0 WORLD_SIZE=2 INIT_METHOD=file:///artifacts/tp2logs/ppl_rdv \
  python3 -m qmesh.tp2.panel_ppl --mode both --chunks 4 --convs 4 --out /artifacts/tp2logs/ppl.json

Env: OVERLAY, TORCH_EXTENSIONS_DIR, MAX_CHUNK (=2048), PREFIX_CACHE, as in qmesh/tp2/run_tp2.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tp2 as R  # noqa: E402  (env constants, TP backend, TP reduce hooks, log)
from panel_encode import MODEL_DIR, load_wikitext2_raw, make_encoder  # noqa: E402

from exllamav3 import Cache, Config, Generator, Job, Model, Tokenizer  # noqa: E402

PANEL_FILE = os.environ.get("PANEL_FILE", "/artifacts/quality/conversations_1017_panel.jsonl")
SCORE_CHUNK = 64  # logits rows scored per logsumexp block (memory)



def build_model(ctx: int, cached: bool):
    torch.cuda.set_device(R.RANK)
    dist.init_process_group(backend = "nccl", rank = R.RANK, world_size = R.WORLD,
                            init_method = os.environ["INIT_METHOD"])
    device = torch.device("cuda", R.RANK)
    t0 = time.time()
    config = Config.from_directory(os.path.join(R.OVERLAY, f"rank{R.RANK}"))
    assert config.tp_size == R.WORLD and config.tp_rank == R.RANK, \
        f"checkpoint tp_size/tp_rank {config.tp_size}/{config.tp_rank} != launch {R.WORLD}/{R.RANK}"
    model = Model.from_config(config)
    # The Cache must exist before model.load(): loading is what allocates the tensors of every attached cache
    cache = Cache(model, max_num_tokens = (ctx + 2 + 255) // 256 * 256, max_history = R.MAX_HISTORY,
                  max_batch_size = 1, **R.KV_CACHE_KWARGS) if cached else None   # prompt + the one decode row
    model.load(device = device, max_chunk_size = ctx, max_output_size = ctx,
               max_batch_size = 1, progressbar = False)
    backend = R.TPBackendNCCL()
    n_blocks = R.set_tp_reduce(model, backend)
    local_v, full_v = R.wrap_vocab_split(model, R.RANK, R.WORLD, backend)
    tokenizer = Tokenizer.from_config(config)
    R.log(f"LOADED {time.time() - t0:.1f}s blocks={n_blocks} vocab_local={local_v} vocab_full={full_v} "
          f"actual_vocab={tokenizer.actual_vocab_size} ctx={ctx} "
          f"max_alloc_GiB={torch.cuda.max_memory_allocated(device) / 2**30:.2f}")
    return model, device, tokenizer, backend, cache


def pack_name() -> str:
    """Directory name of the shard set the overlay tensors resolve to (what is actually scored).

    `MODEL_DIR` is the fp8 source checkpoint the encoder config comes from; the weights come from
    `OVERLAY/rank{R}` whose tensors symlink into the served EXL3 pack. Reporting only MODEL_DIR made
    every result file claim the fp8 source.
    """
    import glob
    try:
        f = sorted(glob.glob(os.path.join(R.OVERLAY, f"rank{R.RANK}", "model-*-of-*.safetensors")))
        if f:
            return os.path.basename(os.path.dirname(os.path.dirname(os.path.realpath(f[0]))))
    except Exception:
        pass
    return "unknown"


@torch.inference_mode()
def forward_logits(model, device, ids: list[int]):
    """[1, len(ids), vocab] logits of a single pass over the whole sequence, on the `--attn` path."""
    if CACHED_GEN is not None:
        return cached_logits(model, device, ids)
    t = torch.tensor([ids], dtype = torch.long, device = device)
    return model.forward(t, {"attn_mode": "flash_attn_nc"})


# --- cached (production) path ------------------------------------------------------------------------
# `flash_attn_nc` is a stateless single-pass path and is NOT prefix-stable on this TP2 build: the same
# token at the same absolute position changes NLL when only the trailing chunk length changes
# (measured mean 0.87 / max 7.5 nat and 48 top-1 flips per 202 positions, L=256 vs L=2048, i.e. a
# non-causal path), so it cannot be used to score a panel. Everything below scores through the cached
# path the serving generator uses. `--attn nc` is kept only for A/B diagnosis.
CACHED_GEN = None
CACHED_CAP: list | None = None   # per-prefill-chunk logits captured from the full-stack forward


def _install_prefill_with_logits(model):
    """Replace model.prefill with the same cached forward plus the head over every prompt row. Set on the
    instance: set_tp_reduce already wrapped model.prefill / model.forward there, which shadows any class hook.
    Calls the wrapped model.forward, so the TP backend is still injected into params."""
    def prefill(input_ids, params = None):
        p = dict(params or {})
        p.pop("last_tokens_only", None)
        y = model.forward(input_ids, p)
        lg = y if torch.is_tensor(y) else (y.get("logits") if isinstance(y, dict) else None)
        if lg is not None and CACHED_CAP is not None:
            CACHED_CAP.append(lg)
        return None
    model.prefill = prefill


def build_cached_scorer(model, cache, tokenizer, chunk: int):
    """Install a prefill hook that returns logits for every prompt row, and the Generator to drive it."""
    gen = Generator(model, cache, tokenizer, max_batch_size = 1, max_chunk_size = chunk,
                    recurrent_cache_size = 0)   # no prefix cache: 0 avoids the 4 GiB default alloc
    gen.prefix_cache_min_match = None            # read via getattr() by pagetable.py: disables it
    # Model.prefill stops at the last cache-writing module, so it never runs the head (job.prefill
    # ignores its return value). The full forward is the same kernels plus the head over every row --
    # the call shape job.py already uses for MTP drafting, minus `last_tokens_only`.
    _install_prefill_with_logits(model)
    return gen


@torch.inference_mode()
def cached_logits(model, device, ids: list[int]):
    """[1, len(ids), vocab] logits of one sequence from one cached prefill (production path)."""
    global CACHED_CAP
    gen = CACHED_GEN
    CACHED_CAP = []
    gen.enqueue(Job(input_ids = torch.tensor([ids], dtype = torch.long), max_new_tokens = 1,
                    min_new_tokens = 1, sampler = R.RankAgreedGreedySampler(), stop_conditions = [],
                    return_logits = True))
    # Prefill covers every token but the last; the generator feeds the last one as the first decode step and
    # returns that row's pre-sampling logits in the result (return_logits)
    last = []
    while gen.num_remaining_jobs():
        for result in gen.iterate():
            if "error" in result:
                raise result["error"]
            if "logits" in result:
                last.append(result["logits"])
    cap, CACHED_CAP = CACHED_CAP, None
    if R.RANK != 0:
        return torch.zeros((1, 1, 1), dtype = torch.float, device = device)
    if not cap or not last:
        raise RuntimeError("cached prefill captured no logits")
    tail = last[0].to(device = cap[0].device).reshape(1, -1, last[0].shape[-1])[:, :1, :]
    w = min(tail.shape[-1], *(c.shape[-1] for c in cap))   # scoring reads only the first `vocab` columns
    lg = torch.cat([c[..., :w] for c in cap] + [tail[..., :w].to(cap[0].dtype)], dim = 1)
    assert lg.shape[1] == len(ids), f"captured {lg.shape[1]} logits rows for {len(ids)} tokens"
    return lg


def nll_at(lg: torch.Tensor, rows: torch.Tensor, targets: torch.Tensor, vocab: int) -> torch.Tensor:
    """NLL (nats) of targets given logits rows, over the first `vocab` columns, in row blocks."""
    out = torch.empty(rows.numel(), dtype = torch.float64, device = lg.device)
    for a in range(0, rows.numel(), SCORE_CHUNK):
        b = min(a + SCORE_CHUNK, rows.numel())
        sel = lg.index_select(0, rows[a:b])[:, :vocab].float()
        lse = torch.logsumexp(sel, -1)
        picked = sel.gather(-1, targets[a:b].unsqueeze(-1)).squeeze(-1)
        out[a:b] = (lse - picked).double()
    return out


# --- protocols --------------------------------------------------------------------------------------

def run_wikitext(model, device, raw_tok, args, vocab: int, bos: int | None) -> dict:
    text = load_wikitext2_raw()
    tokens = raw_tok.encode(text, add_special_tokens = False).ids
    n_ctx = args.ctx
    if len(tokens) < 2 * n_ctx:
        raise ValueError(f"wikitext-2 needs >= {2 * n_ctx} tokens, got {len(tokens)}")
    n_chunks_total = len(tokens) // n_ctx
    n_chunks = n_chunks_total if not args.chunks else min(args.chunks, n_chunks_total)
    first = n_ctx // 2
    R.log(f"WIKI2 tokens={len(tokens)} chunks={n_chunks_total} (scoring {n_chunks}) n_ctx={n_ctx} "
          f"positions=[{first},{n_ctx - 2}]")

    total_nll = total_nll2 = 0.0
    total_count = 0
    per_chunk = []
    t0 = time.time()
    for i in range(n_chunks):
        start = i * n_ctx
        chunk = list(tokens[start:start + n_ctx])
        if bos is not None:
            chunk[0] = bos
        lg = forward_logits(model, device, chunk)[0]
        if R.RANK == 0:
            rows = torch.arange(first, n_ctx - 1, device = device, dtype = torch.long)
            targets = torch.tensor(tokens[start + first + 1:start + n_ctx], dtype = torch.long, device = device)
            nll = nll_at(lg, rows, targets, vocab)
            total_nll += nll.sum().item()
            total_nll2 += (nll * nll).sum().item()
            total_count += nll.numel()
            per_chunk.append(round(math.exp(total_nll / total_count), 4))
            R.log(f"WIKI2 chunk {i + 1}/{n_chunks} running_ppl={per_chunk[-1]:.4f} "
                  f"({time.time() - t0:.1f}s)")
        del lg
        torch.cuda.empty_cache()
    res = {"chunks_scored": n_chunks, "chunks_total": n_chunks_total, "n_ctx": n_ctx,
           "positions": total_count, "elapsed_s": round(time.time() - t0, 1)}
    if R.RANK == 0 and total_count:
        mean_nll = total_nll / total_count
        mean_nll2 = total_nll2 / total_count
        var = max(0.0, mean_nll2 - mean_nll * mean_nll)
        stderr_nll = math.sqrt(var / (total_count - 1)) if total_count > 1 else 0.0
        ppl = math.exp(mean_nll)
        res.update({"nll_mean": mean_nll, "ppl": ppl, "ppl_stderr": stderr_nll * ppl,
                    "per_chunk_running_ppl": per_chunk})
        R.log(f"WIKI2 RESULT ppl={ppl:.4f} +/- {stderr_nll * ppl:.5f} nll={mean_nll:.6f} "
              f"positions={total_count}")
    return res


def _ppl_block(nll_sum: float, nll2_sum: float, count: int) -> dict:
    mean = nll_sum / count
    var = max(0.0, nll2_sum / count - mean * mean)
    stderr = math.sqrt(var / (count - 1)) if count > 1 else 0.0
    ppl = math.exp(mean)
    return {"nll_mean": mean, "ppl": ppl, "ppl_stderr": stderr * ppl}


def run_panel(model, device, enc, args, vocab: int) -> dict:
    """Score every --panel-file; returns per-file results plus the aggregate over all of them."""
    tot_nll = tot_nll2 = 0.0
    tot_count = 0
    by_dom: dict[str, list] = {}
    out_files: dict[str, dict] = {}
    t_all = time.time()
    for pf in args.panel_files:
        name = os.path.basename(pf)
        rows = [json.loads(line) for line in open(pf, encoding = "utf-8")]
        rows = [r for r in rows if r.get("split", "panel") == "panel"]
        if args.convs:
            rows = rows[:args.convs]
        R.log(f"PANEL file={pf} conversations={len(rows)} panel_max={args.panel_max}")
        f_nll = f_nll2 = 0.0
        f_count = 0
        f_dom: dict[str, list] = {}
        per_conv = []
        t0 = time.time()
        for i, r in enumerate(rows):
            ids, score = enc(r)
            ids, score = ids[:args.panel_max], score[:args.panel_max]
            assert 2 <= len(ids) <= args.ctx, f"{r['id']}: rendered length {len(ids)} outside 2..{args.ctx}"
            lg = forward_logits(model, device, ids)[0]
            if R.RANK == 0:
                sc = torch.tensor(score, dtype = torch.bool, device = device)
                pos = sc[1:].nonzero().squeeze(-1)      # position t predicts token t+1
                if pos.numel():
                    targets = torch.tensor(ids[1:], dtype = torch.long, device = device)[pos]
                    nll = nll_at(lg, pos, targets, vocab)
                    n_sum = nll.sum().item()
                    n2_sum = (nll * nll).sum().item()
                    f_nll += n_sum
                    f_nll2 += n2_sum
                    f_count += nll.numel()
                    dom = str(r.get("domain", "?")).split("_")[0]
                    f_dom.setdefault(dom, []).append((n_sum, nll.numel()))
                    per_conv.append({"id": r["id"], "domain": r.get("domain"), "length": len(ids),
                                     "scored": int(pos.numel()), "nll_sum": n_sum})
                if (i + 1) % 10 == 0 or i + 1 == len(rows):
                    running = math.exp(f_nll / f_count) if f_count else float("nan")
                    R.log(f"PANEL {name} {i + 1}/{len(rows)} running_ppl={running:.4f} positions={f_count} "
                          f"({time.time() - t0:.1f}s)")
            del lg
            torch.cuda.empty_cache()
        res = {"file": pf, "conversations": len(rows), "panel_max": args.panel_max,
               "positions": f_count, "elapsed_s": round(time.time() - t0, 1)}
        if R.RANK == 0 and f_count:
            res.update(_ppl_block(f_nll, f_nll2, f_count))
            res["by_domain"] = {k: {"nll": sum(v[0] for v in x) / sum(v[1] for v in x),
                                    "positions": sum(v[1] for v in x)}
                                for k, x in sorted(f_dom.items())}
            res["per_conversation"] = per_conv
            R.log(f"PANEL {name} ppl={res['ppl']:.4f} nll={res['nll_mean']:.6f} positions={f_count} "
                  f"({res['elapsed_s']}s)")
            tot_nll += f_nll
            tot_nll2 += f_nll2
            tot_count += f_count
            for k, v in f_dom.items():
                by_dom.setdefault(k, []).extend(v)
        out_files[name] = res
    agg = {"files": list(out_files), "conversations": sum(v["conversations"] for v in out_files.values()),
           "panel_max": args.panel_max, "positions": tot_count, "elapsed_s": round(time.time() - t_all, 1),
           "by_file": out_files}
    if R.RANK == 0 and tot_count:
        agg.update(_ppl_block(tot_nll, tot_nll2, tot_count))
        agg["by_domain"] = {k: {"nll": sum(v[0] for v in x) / sum(v[1] for v in x),
                                "positions": sum(v[1] for v in x)}
                            for k, x in sorted(by_dom.items())}
        R.log(f"PANEL RESULT ppl={agg['ppl']:.4f} nll={agg['nll_mean']:.6f} positions={tot_count} "
              f"files={len(out_files)}")
    return agg


SANITY_TEXTS = {
    "facts": "The capital of France is Paris. The capital of Germany is Berlin. "
             "The capital of Italy is Rome. The capital of Spain is Madrid. "
             "The capital of Portugal is Lisbon. The capital of Austria is Vienna.",
    "repeat": "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
              "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
              "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu. "
              "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.",
    "code": "def add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n\n"
            "x = add(2, 3)\ny = mul(x, 4)\nprint(x, y)\n",
}


def run_sanity(model, device, tokenizer, vocab: int) -> dict:
    """Absolute forward sanity: boilerplate text a healthy model nearly predicts. Both ranks run the
    forward (collectives), rank0 scores. ppl >> 1 on 'repeat' means the forward is corrupt, not the pack."""
    out = {}
    for name, text in SANITY_TEXTS.items():
        ids = tokenizer.encode(text, add_bos = True)[0].tolist()
        lg = forward_logits(model, device, ids)[0]
        if R.RANK != 0:
            del lg
            continue
        pos = torch.arange(len(ids) - 1, device = device)
        targets = torch.tensor(ids[1:], dtype = torch.long, device = device)
        nll = nll_at(lg, pos, targets, vocab)
        top1 = lg.index_select(0, pos)[:, :vocab].float().argmax(-1)
        acc = (top1 == targets).float().mean().item()
        mean = nll.mean().item()
        out[name] = {"tokens": len(ids), "nll_mean": mean, "ppl": math.exp(mean), "top1_acc": acc,
                     "nll_per_token": [round(v, 4) for v in nll.tolist()]}
        R.log(f"SANITY {name} tokens={len(ids)} nll={mean:.4f} ppl={out[name]['ppl']:.3f} top1={acc:.3f}")
        R.log(f"SANITY {name} per_token_nll=" + " ".join(f"{v:.2f}" for v in nll.tolist()))
        del lg
    return out

def run_tokens(model, device, ids: list, gen_ids: list, vocab: int, logits_out: str = "") -> dict:
    """Teacher-force an exact prompt token stream (plus whatever tokens the generator sampled from it) through
    the same stateless pass the panels use, and report the NLL of the generated region. The generator picked
    those tokens as argmax of its own cached-path logits, so a healthy nc path scores them near 0 with top1 == 1:
    any gap is a nc-path/cached-path disagreement inside this build, independent of data or pack quality."""
    all_ids = [int(x) for x in ids] + [int(x) for x in gen_ids]
    n_gen = len(gen_ids)
    lg = forward_logits(model, device, all_ids)[0]
    if R.RANK != 0:
        del lg
        return {}
    pos = torch.arange(len(all_ids) - 1, device = device)
    targets = torch.tensor(all_ids[1:], dtype = torch.long, device = device)
    nll = nll_at(lg, pos, targets, vocab)
    top1 = lg.index_select(0, pos)[:, :vocab].float().argmax(-1)
    first = len(all_ids) - n_gen - 1
    gen_nll = nll[first:]
    gen_top1 = (top1[first:] == targets[first:]).float().mean().item()
    prompt_nll = nll[:first]
    res = {"total": len(all_ids), "prompt": len(ids), "gen": n_gen,
           "prompt_nll_mean": prompt_nll.mean().item() if prompt_nll.numel() else None,
           "gen_nll_mean": gen_nll.mean().item() if gen_nll.numel() else None,
           "gen_nll_max": gen_nll.max().item() if gen_nll.numel() else None,
           "gen_top1_match": gen_top1,
           "gen_nll_per_token": [round(v, 4) for v in gen_nll.tolist()]}
    if logits_out:
        torch.save(lg.index_select(0, pos[first:]).contiguous(), logits_out)
    R.log(f"SELFNLL total={res['total']} prompt={len(ids)} gen={n_gen} "
          f"prompt_nll={res['prompt_nll_mean']} gen_nll={res['gen_nll_mean']} gen_ppl="
          f"{math.exp(res['gen_nll_mean']) if res['gen_nll_mean'] is not None else None} "
          f"gen_nll_max={res['gen_nll_max']} gen_top1={gen_top1:.3f}")
    R.log("SELFNLL gen_per_token_nll=" + " ".join(f"{v:.2f}" for v in gen_nll.tolist()))
    del lg
    return res


def main():
    global CACHED_GEN
    ap = argparse.ArgumentParser(allow_abbrev = False)
    ap.add_argument("--mode", default = "both", choices = ["both", "wiki2", "panel", "sanity", "tokens"])
    ap.add_argument("--ctx", type = int, default = 2048, help = "n_ctx for wiki2 and the max sequence length")
    ap.add_argument("--chunks", type = int, default = 0, help = "limit the number of wiki2 chunks (0 = all)")
    ap.add_argument("--convs", type = int, default = 0, help = "limit the number of panel conversations")
    ap.add_argument("--sweep", default = "", help = "comma list of panel_max values to score in one load")
    ap.add_argument("--sanity", action = "store_true", help = "score boilerplate texts as an absolute check")
    ap.add_argument("--tokens-file", default = "", help = "json with {'prompt': [ids]} (from the runner PROMPT_IDS_OUT dump)")
    ap.add_argument("--tokens-gen", default = "", help = "json with the generated ids (runner TOKENS_OUT dump, key '0')")
    ap.add_argument("--tokens-out", default = "", help = "save the nc-path logit rows of the generated region (.pt)")
    ap.add_argument("--panel-max", type = int, default = 2048)
    ap.add_argument("--attn", default = "cached", choices = ["cached", "nc"],
                    help = "cached = production generator path (default); nc = stateless flash_attn_nc, "
                           "which is not prefix-stable on this build and only valid for A/B diagnosis")
    ap.add_argument("--prefill-chunk", type = int, default = 0,
                    help = "cached path prefill chunk size (0 = --ctx, i.e. one chunk per sequence)")
    ap.add_argument("--panel-file", action = "append", default = None,
                    help = "panel jsonl to score; repeat to score several in one model load")
    ap.add_argument("--out", default = "")
    ap.add_argument("--reference", action = "append", default = None,
                    help = "upstream kld json per panel file (same order); prints its published nll/ppl")
    args = ap.parse_args()
    args.panel_files = args.panel_file or [PANEL_FILE]
    args.sweep = [int(x) for x in args.sweep.replace(",", " ").split()] if args.sweep.strip() else []
    if not (2 * 256 <= args.ctx <= 65536):
        raise ValueError("--ctx out of range")
    if not (2 <= args.panel_max <= args.ctx):
        raise ValueError("--panel-max must be within [2, ctx]")
    for pm in args.sweep:
        if not (2 <= pm <= args.ctx):
            raise ValueError(f"--sweep value {pm} outside [2, ctx]")

    model, device, tokenizer, _backend, cache = build_model(args.ctx, args.attn == "cached")
    vocab = tokenizer.actual_vocab_size
    raw_tok, enc = None, None
    prefill_chunk = args.prefill_chunk or args.ctx
    if args.attn == "cached":
        CACHED_GEN = build_cached_scorer(model, cache, tokenizer, prefill_chunk)
        R.log(f"ATTN cached prefill_chunk={prefill_chunk} (production path: full-stack forward + head)")
    else:
        R.log("ATTN nc WARNING: stateless flash_attn_nc is NOT prefix-stable on this build (the same token "
              "at the same position moves with the trailing chunk length); these numbers are diagnostics, "
              "not quality numbers")
    out = {"model_dir": MODEL_DIR, "overlay": R.OVERLAY, "pack": pack_name(), "ctx": args.ctx,
           "attn": args.attn, "prefill_chunk": prefill_chunk if args.attn == "cached" else 0,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if args.reference and R.RANK == 0:
        refs = {}
        for i, rf in enumerate(args.reference):
            name = os.path.basename(args.panel_files[i]) if i < len(args.panel_files) else f"ref{i}"
            try:
                ref = json.load(open(rf, encoding = "utf-8"))
                r = {k: ref.get(k) for k in ("positions", "nll_native", "ppl_native", "nll_ours")}
                if ref.get("nll_native") is not None:
                    r["ppl_native"] = math.exp(ref["nll_native"])
                refs[name] = r
            except Exception as e:  # a missing/foreign reference file must not fail the run
                R.log(f"reference {rf}: {e!r}")
        out["reference"] = refs

    if args.mode in ("both", "wiki2"):
        raw_tok = make_encoder()[0]
        out["wiki2"] = run_wikitext(model, device, raw_tok, args, vocab, tokenizer.bos_token_id)
    if args.mode in ("both", "panel", "sanity"):
        if enc is None:
            _, enc = make_encoder()
        if args.sweep:
            sweep = {}
            for i, pm in enumerate(args.sweep):
                args.panel_max = pm
                R.log(f"SWEEP panel_max={pm}")
                sweep[f"{i}_{pm}"] = run_panel(model, device, enc, args, vocab)
            out["panel_sweep"] = sweep
            args.panel_max = args.sweep[0]
        else:
            out["panel"] = run_panel(model, device, enc, args, vocab)
    if args.mode == "tokens":
        if not args.tokens_file:
            raise ValueError("--mode tokens needs --tokens-file")
        d = json.load(open(args.tokens_file, encoding = "utf-8"))
        p_ids = [int(x) for x in (d["prompt"] if isinstance(d, dict) else d)]
        g_ids = []
        if args.tokens_gen:
            g = json.load(open(args.tokens_gen, encoding = "utf-8"))
            g_ids = [int(x) for x in (g[min(g, key = int)] if isinstance(g, dict) else g)]
        out["selfnll"] = run_tokens(model, device, p_ids, g_ids, vocab, args.tokens_out)
    if args.sanity:
        out["sanity"] = run_sanity(model, device, tokenizer, vocab)

    dist.barrier()
    if R.RANK == 0:
        def _trim(v):   # keep the RESULT log line readable: drop the 199-row per-conversation tables
            if isinstance(v, dict):
                return {k: _trim(x) for k, x in v.items() if k != "per_conversation"}
            return v
        brief = _trim(out)
        R.log("RESULT " + json.dumps(brief))
        if args.out:
            with open(args.out, "w", encoding = "utf-8") as f:
                json.dump(out, f, indent = 1)
            R.log(f"wrote {args.out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
