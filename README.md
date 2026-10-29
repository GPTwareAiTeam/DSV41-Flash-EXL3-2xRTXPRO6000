<p align="center"><b>QMesh</b> — a tensor-parallel runtime for very large EXL3 models on two consumer-class GPUs</p>

# QMesh EXL3 Runtime

This repository is a downstream fork of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT) that turns it
into a **true TP=2 serving runtime** for one very large mixture-of-experts model: a 2.565-bits-per-weight EXL3
quantisation of **DeepSeek-V4.1-Flash-1017**, mixed across experts so that it fits, resident and fully on-GPU, in
**2 × 96 GB** while running the same weights split across both cards with per-block collectives.

It is not a general-purpose fork: upstream remains intact (`exllamav3/`, `examples/`, `tests/`, `util/`,
`doc/`), the TP=2 work lives in `qmesh/` and is documented in `docs/tp2/`, and upstream's own README is preserved
as [README.exllamav3.md](README.exllamav3.md).

Status: **v1.0.0** — measured, working, with the limitations listed below stated plainly.

## What it does

| | Measured | Conditions |
|---|---|---|
| Single-stream decode | **93.7 tok/s** (10.67 ms/token) | greedy, 128 tokens, `COMM=p2p`, whole-step graphs |
| Speculative decode | **111–215 tok/s** (K=5, AL 2.6–5.3) | short prompts, acceptance .66–.91 |
| Concurrent streams | **203–206 tok/s** aggregate at B4 | short prompts, captured batched path, 86.5 GiB/GPU |
| Prefill | **4.82k tok/s** (`COMM=hybrid`), 4.74–4.78k on NCCL | 32k prompt, chunk 8192 |
| Long context | 16.1–16.6 ms/token at 24k, 16.9 at 60k | `COMM=hybrid`, eager path; the 60k figure predates compaction |
| Prefix cache | **36–40×** TTFT at 20k, **86–88×** at 60k | full hit on a previously seen prefix |
| Footprint | 86.9 GiB resident per rank after load, 90.7 GiB peak at 32k prefill, ~6 GiB free | 2.565 bpw, fp16 KV, 8192 chunk |

Every figure, its configuration, and the run it came from are in [docs/tp2/RESULTS.md](docs/tp2/RESULTS.md).
Read the "how to read these tables" note there before comparing anything: these cards are power-capped and
drift, so only back-to-back A/B numbers are comparable.

## Hardware and software

- 2 × NVIDIA RTX PRO 6000 Blackwell (96 GB), PCIe peer-to-peer between them, both on one NUMA node.
  The runtime needs `sm_120a`; other Blackwell parts should work if they fit the weights.
- Python ≥ 3.10, a PyTorch build with CUDA 12.8+ and `sm_120a` support, the CUDA toolkit, `ninja`, `gcc`.
- `torch.distributed` (NCCL) for rendezvous; a one-shot P2P all-reduce kernel compiled at first import for the
  fast path.
- ~87 GiB of GPU memory per rank for the reference checkpoint resident, plus ~1 GiB per rank of KV for a
  32k-token context (the exact cost scales with context, batch and the graph pool).

Install the way you install upstream — the extension builds on first import:

```sh
git clone https://github.com/GPTwareAiTeam/QMesh-EXL3-Runtime.git qmesh
cd qmesh
pip install -e .
TORCH_CUDA_ARCH_LIST=12.0a python -c "import exllamav3"   # JIT-builds the extension, ~2 min
```

Set `TORCH_EXTENSIONS_DIR` to a persistent path or every restart rebuilds. No container image ships with this
tree: the runtime builds in a plain PyTorch image with the CUDA toolkit and `ninja`, provided
`TORCH_CUDA_ARCH_LIST` includes `12.0`; `scripts/bench_v1.sh` assumes you already have such a container
running on both GPUs.

## Getting the checkpoint

You need a TP=2 sharded EXL3 checkpoint. `qmesh/tp2/shard_tool.py` produces one from a full checkpoint on a
CPU-only host; the expert-mix rationale, the tensor split rules and a three-step reproduction are in
[docs/tp2/RECIPE.md](docs/tp2/RECIPE.md) and [docs/tp2/SHARD_SPEC.md](docs/tp2/SHARD_SPEC.md). The output
directory holds `rank0/` and `rank1/`, each a self-describing EXL3 checkpoint with `tp_size=2, tp_rank=r`.

## Running it

One process per GPU. `scripts/bench_v1.sh` is the reference invocation: it launches both ranks, waits, and
reports ms/token plus a rank-to-rank token comparison.

```sh
export EXL3_ARTIFACTS=$PWD/artifacts        # host side; mounted at /artifacts in the container
export CONTAINER=qmesh-tp2                  # container name
export COMM=p2p                             # decode fast path
# OVERLAY is the directory that holds rank0/ and rank1/
TAG=first_run scripts/bench_v1.sh \
  -e OVERLAY=/artifacts/tp2-overlay \
  -e EXL3_WHOLE_GRAPH=1 -e EXL3_DSV4_B1_BATCH=1
```

For a long-context or multi-stream run, add `-e CTX_TOKENS=131072`, `-e BATCH=4`, or
`-e LONG_PROMPT_TOKENS=24000`. `DSPARK=1` (the script default) enables speculative decoding; use
`DSPARK=0` for a plain baseline. `COMM=hybrid` routes prefill through NCCL and decode through P2P and is the
right default once you care about TTFT more than decode latency.

The launcher is plain `docker exec`; without a container, run the same command twice by hand:

```sh
rm -rf /tmp/rdv && mkdir -p /tmp/rdv/rendezvous /tmp/rdv/p2p
for r in 0 1; do
  RANK=$r WORLD_SIZE=2 INIT_METHOD=file:///tmp/rdv/rendezvous P2P_RDV=/tmp/rdv/p2p \
  TORCH_CUDA_ARCH_LIST=12.0a EXL3_WHOLE_GRAPH=1 EXL3_DSV4_B1_BATCH=1 \
  python -m qmesh.tp2.run_tp2 > /tmp/rdv/rank$r.log 2>&1 &
done
wait
```

Every dial is documented in [docs/tp2/env_vars.md](docs/tp2/env_vars.md); the engine-level switches (capture,
fusion, kernel selection) are in [doc/env_vars.md](doc/env_vars.md).

## How it works

- **Sharding.** Attention heads, KV heads, the vocab embedding, the output head and the MoE expert
  intermediate dimension are split two ways; the router, the shared experts' input projection and the norms
  are replicated. Each rank builds a *single-device* ExLlamaV3 model with `tp_size=2, tp_rank=r`, which keeps
  the upstream loader, cache and sampler untouched. Design and memory budget: [docs/tp2/DESIGN.md](docs/tp2/DESIGN.md).
- **Collectives.** Partial sums from attention and MoE are combined by a one-shot CUDA-IPC all-reduce over
  PCIe: 4.93 µs per call on decode-sized payloads versus 11.84 µs for NCCL. Prefill payloads are large, so
  `COMM=hybrid` switches to NCCL above a byte threshold — that is worth +16.5% prefill. The vocab head is
  all-gathered so greedy sampling is bit-identical on both ranks.
- **Piecewise CUDA graphs.** The decode step is captured as one graph segment per transformer block and
  merged across blocks; the first step of a run is the capture. This is the single largest decode win
  (13.35 → 10.97 ms/token, then 10.67 with the fused launch merges) and it is why capture *eligibility* is
  checked and logged rather than assumed.
- **Speculative decoding (DSpark).** A DFlash drafter head runs sharded on both ranks; rank 0 decides the
  accepted prefix and broadcasts it, so both ranks verify the same width. The drafter is a separate EXL3
  checkpoint; `TORCH_EXTENSIONS_DIR` and the overlay both need room for it.
- **Prefix cache.** Recurrent layers make exact KV reuse non-trivial, so the cache stores recurrent-state
  checkpoints at chunk boundaries and replays from the last page-aligned one. Costs host RAM (~268 MB per rank
  per 60k context), not VRAM.
- **Prefill chunking.** `MAX_CHUNK=auto` sizes the prefill chunk from a measured device-peak table against
  whatever KV budget you gave it, so a 1M-token context costs one step (8192 → 4096) instead of an OOM.

## Repository layout

```
qmesh/tp2/        runner, sharding tools, TP=2 benchmarks, launchers
qmesh/p2p/        P2P all-reduce benchmark
qmesh/*.py        standalone MoE / attention / all-reduce benches
docs/tp2/         design, recipe, results, env dials, shard spec, raw working notes
scripts/          bench_v1.sh (reference run)
exllamav3/        upstream library, plus the TP=2 backend (tp2/) and the P2P kernel
                  (exllamav3_ext/parallel/p2p_oneshot.cu); other changes are additive hooks
doc/, examples/, tests/, util/, science/   upstream, unchanged
```

## Known limitations

These are measured, not speculative:

- **Capture stops at ~1024 positions.** The captured decode body has no top-k indexer selection, so past that
  the runtime falls back to eager attention: 16.1 ms/token at 24k versus 10.7 at short context. Compaction cut
  `aten::nonzero` from 8 to 0 and launches from 585 to 557 per step but did not remove the gate.
- **One long row drops the whole batch to eager** — a B8 batch containing a 1.1k-token prompt measured 81.6
  ms/token against 31.7 for all-short prompts. Fast batching numbers are short-prompt numbers.
- **Batching is invariant across streams, not across batch width.** Rows are bitwise independent of their
  neighbours, but B1/B2/B4/B8 are different arithmetic paths and can diverge on near-ties. Verified in detail
  in [RESULTS.md](docs/tp2/RESULTS.md#6-concurrency).
- **Speculation and batching are substitutes**: at B4, drafting costs more than it saves.
- **FP8 KV** cuts the KV pool by a third (3200 → 2150 bytes/token/rank) with no measurable quality loss, but
  packed pools break capture and the bypassed capture is numerically wrong; `KV_BITS=16` is the default and
  the supported setting.
- 8-bit prefill all-reduce gains 6–7% but fails the coherence gate (a prompt-3 token flip at step 11); it is
  not shipped.
- The 1M-token path exists and sizes correctly, but the eager long-context path is slow (~3.5 min cold prefill
  at that rate). Long-context workloads in the 20k–100k range are the tested ones.

## Roadmap

In the order the measurements justify: capture-safe candidate selection with exact tie order (that is the
long-context gap), per-row capture eligibility so mixed batches do not fall back, then real-transcript
validation of speculative acceptance and of diverse-prompt batching. Details and rejected ideas (so they do
not get retried) are in [docs/tp2/notes/master-plan.md](docs/tp2/notes/master-plan.md).

## Credits and licence

This runtime is a small delta on a lot of other people's work — see [CREDITS.md](CREDITS.md). Upstream ExLlamaV3
is MIT-licensed by Turboderp; the licence text is unchanged in [LICENSE](LICENSE). The checkpoint and the
sharding tools are ours.
