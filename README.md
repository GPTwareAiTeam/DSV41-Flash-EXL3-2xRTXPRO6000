<p align="center"><b>QMesh</b> — a tensor-parallel runtime for very large EXL3 models on two workstation GPUs</p>

# QMesh EXL3 Runtime

A downstream fork of [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (MIT) that runs one very large
mixture-of-experts model as a **true TP=2 runtime** on **2 × 96 GB GPUs**: a mixed-bitrate EXL3 quantisation of
**DeepSeek-V4.1-Flash-1017** (2.565 bpw average — frequently used experts at higher precision, rarely used ones
at lower), fully resident on-GPU, with speculative decoding always on.

Upstream remains intact (`exllamav3/`, `examples/`, `tests/`, `util/`, `doc/`); the TP=2 work lives in `qmesh/`
and is documented in `docs/tp2/`. Upstream's own README is preserved as [README.exllamav3.md](README.exllamav3.md).

Status: **v1.0.0**.

## Performance

Measured on 2 × RTX PRO 6000 Blackwell Max-Q (PCIe, power-capped), greedy decoding, speculative decoding on
(DSpark, K=5, adaptive confidence cutoff), CPU-pinned ranks. Full tables, configurations and logs:
[docs/tp2/RESULTS.md](docs/tp2/RESULTS.md).

**Single-user decode, 24k-token context** (512 generated tokens):

| Content | tok/s |
|---|---|
| Math | 189 |
| Code | 177 |
| Structured data (JSON) | 169 |
| Agentic coding, 24k (private sessions, not distributed) | 142 |
| Multilingual (zh + en) | 100 |
| Prose | 97 |

**Long context, single user** (decode tok/s): 72k agentic 128, 72k code 128, 128k agentic 124.

**4 concurrent users** (12k-token prompts, 16k KV each): 189 tok/s total, 58 tok/s per user.

**Cold prefill** (no prefix cache): 2,614 tok/s at 24k, 2,782 at 72k, 2,746 at 128k (128k prompt → first token
in 48 s). Repeat prompts hit the prefix cache instead: a 128k prompt seen before answers in under 0.1 s.

The agentic prompts are the owner's own coding-assistant sessions; they are private and not included in this
repository. To reproduce those rows, use your own agentic transcripts.

Read the note in RESULTS.md before comparing numbers: these cards are power-capped and drift, so only
back-to-back interleaved A/B numbers are comparable.

## Quality

Perplexity on the two published DeepSeek-V4.1-Flash-1017 panels (199 conversations each; 237,457 and 254,846
scored assistant positions, the same positions the published packs were scored on), measured through this
runtime's production prefill path:

| Build | Expert bpw | 1017 panel (native 1017 replies) | sudoingx panel |
|---|---|---|---|
| DeepSeek-V4.1-Flash-1017, original weights | — | 1.5720 | 1.6005 |
| sudoingx v2.0 | 2.80 | 1.5869 | 1.6117 |
| HeNryous | 2.75 | 1.5871 | 1.6119 |
| object-oriented-architects | 2.7 | 1.5931 | 1.6146 |
| sudoingx v1.2 | 2.70 | 1.5946 | 1.6159 |
| LibertAIDAI | 2.9 | 1.6040 | 1.6155 |
| **This release (abliterated base)** | **2.565** | **1.6886** | **1.7041** |

Read this table with two caveats:

- The other rows were scored by their authors' harness on DGX Spark; this row is scored by
  `qmesh/tp2/panel_ppl.py` through the TP=2 runtime. Same panels and positions, different code path.
- The base checkpoint is [dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8),
  an abliterated (refusal-removed) fine-tune. Every other quantised row is built from the original
  `deepseek-ai/DeepSeek-V4.1-Flash-1017` (per their model cards), and the panels hold the original model's replies.
  This row's gap to the original therefore includes the abliteration edit as well as quantisation; the split
  between the two has not been measured.

Prefill-chunk invariance: the same text scores the same at 256-, 2048- and 4096-row prefill chunks (8k-token
WikiText-2 windows: 2.1820 / 2.1827 / 2.1827, ±0.025).

## What is in v1

- **True TP=2.** Attention heads, KV heads, vocab embedding/head and the MoE intermediate dimension are split two
  ways; each rank is a single-device ExLlamaV3 model (`tp_size=2`). Partial sums are combined by a one-shot
  CUDA-IPC all-reduce over PCIe. Design: [docs/tp2/DESIGN.md](docs/tp2/DESIGN.md).
- **Captured decode.** Each decode step and each speculative verify step replays CUDA graphs instead of
  ~1,600 kernel launches — including long-context steps and the multi-token verify.
- **Speculative decoding, always on, adaptive.** The DSpark drafter head proposes 5 tokens per round, captured
  as GPU graphs. Its confidence head decides each round how many drafts to verify; the confidence cutoff adapts
  to content (bolder on low-acceptance text such as prose and multilingual) on the GPU, with no host cost
  (`EXL3_DSPARK_GATE`).
- **Correctness by construction for captured graphs.** Every graph is sized from fixed capacities, not from the
  current request, so alternating short and long requests never re-captures a graph against stale memory.
  Repeated requests are token-identical; both ranks are token-identical.
- **Exact routing.** The multi-row expert router is bitwise identical to single-token decode.
- **OpenAI-compatible server** (`qmesh/tp2/serve_tp2.sh start|stop|status`): chat and completions, streaming,
  tool calls, prompt-cache reuse with cached-token usage, API key, clean shutdown. See
  [docs/tp2/SERVER.md](docs/tp2/SERVER.md).
- **Prompt cache** with recurrent-state checkpoints, so multi-turn agentic sessions skip the re-prefill.

## Hardware and software

- 2 × NVIDIA RTX PRO 6000 Blackwell (96 GB), PCIe peer-to-peer, one NUMA node. Needs `sm_120a`.
- Python ≥ 3.10, PyTorch with CUDA 12.8+ and `sm_120a`, the CUDA toolkit, `ninja`, `gcc`.
- About 87–91 GiB per GPU for weights, drafter and graphs; the rest is KV cache (FP4 KV by default, ~890 bytes per token per GPU). The KV
  cache costs 3,200 bytes per token per GPU; the drafter keeps only a small fixed window per user, independent of
  context length.

Install the way you install upstream — the extension builds on first import:

```sh
git clone https://github.com/GPTwareAiTeam/QMesh-EXL3-Runtime.git qmesh
cd qmesh
pip install -e .
TORCH_CUDA_ARCH_LIST=12.0a python -c "import exllamav3"   # JIT-builds the extension
```

Set `TORCH_EXTENSIONS_DIR` to a persistent path, or every restart rebuilds.

## Getting the checkpoint

Download the pre-sharded checkpoint from Hugging Face (link on release), or build one: `qmesh/tp2/shard_tool.py`
produces `rank0/` and `rank1/` from a full EXL3 checkpoint; see [docs/tp2/RECIPE.md](docs/tp2/RECIPE.md) and
[docs/tp2/SHARD_SPEC.md](docs/tp2/SHARD_SPEC.md).

## Running it

Server (recommended):

```sh
OVERLAY=/path/to/checkpoint qmesh/tp2/serve_tp2.sh start      # OpenAI-compatible API on port 8000
qmesh/tp2/serve_tp2.sh status
qmesh/tp2/serve_tp2.sh stop
```

Direct runner, one process per GPU (`qmesh/tp2/launch_tp2.sh` does this, with CPU pinning):

```sh
for r in 0 1; do
  RANK=$r WORLD_SIZE=2 INIT_METHOD=file:///tmp/rdv/rendezvous P2P_RDV=/tmp/rdv/p2p \
  TORCH_CUDA_ARCH_LIST=12.0a DSPARK=1 \
  python -m qmesh.tp2.run_tp2 > /tmp/rdv/rank$r.log 2>&1 &
done; wait
```

Add `CTX_TOKENS=131072` for long context or `BATCH=4` for concurrent streams. Every dial is documented in
[docs/tp2/env_vars.md](docs/tp2/env_vars.md) and [doc/env_vars.md](doc/env_vars.md).

## Opt-in settings (lossy or experimental)

- `PREFILL_AR=int8` — 8-bit prefill all-reduce, ~+13% prefill; changes outputs slightly.
- `KV_BITS=8` — 8-bit KV cache, ~1.5× context capacity; fails our quality gate at long context, so off by default.
- `DSPARK_HEAD=diffbot` — diffbot's drafter head (MIT, credited); ours is faster on every content type we tested.
- `EXL3_DSPARK_GATE=ev` — per-round expected-value draft width; larger wins on some content, larger losses on
  others.

## Known limitations

- Cold prefill is ~2.6–2.8k tokens/s; a 128k prompt takes ~48 s to its first token. Repeated prefixes hit the
  prefix cache and skip that.
- Multi-user throughput scales weakly: with several users, decode attention still runs one user at a time (only
  the expert and dense layers are batched). 4 users reach 189 tok/s total. A whole-batch attention graph is the
  planned fix.
- 8 concurrent users with speculation do not run: out of memory at 16k KV per user, and a drafter graph-capture
  failure at 8k per user. 8 users without speculation run.
- Little VRAM is spare. At the default 4096-row prefill chunk (1 user, 32k context, speculation on) a 24k prompt
  still runs with 1.42 GiB per GPU used by another process and fails at 1.92 GiB. With `MAX_CHUNK=2048` it runs
  with 2.92 GiB used (the most tested), at a cost in prefill speed.
- K=7 speculation needs a block-7 drafter head (only diffbot's exists) and does not beat K=5 on our test set;
  parked for v2 (see `docs/tp2/notes/`).

## Credits and licence

This runtime is a delta on a lot of other people's work — see [CREDITS.md](CREDITS.md): turboderp (ExLlamaV3,
MIT), DeepSeek (model, MIT), dealignai (base checkpoint, MIT), the external MoE-kernel candidates kit (two
kernel changes compiled in), and diffbot (optional drafter head, MIT). The code is MIT; the licence text is
unchanged in [LICENSE](LICENSE).
