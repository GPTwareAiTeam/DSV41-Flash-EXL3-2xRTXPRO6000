<p align="center"><b>DSV41-Flash-EXL3-2xRTXPRO6000</b> — DeepSeek-V4.1-Flash-1017, mixed-bitrate EXL3, on two RTX PRO 6000 Max-Q GPUs, tensor-parallel</p>

# DeepSeek-V4.1-Flash-1017 EXL3 Mixed-Bitrate on 2× RTX PRO 6000 Max-Q (TP=2)

## What this project is, and whose work it builds on

This is an independent runtime for one job: serving DeepSeek-V4.1-Flash-1017 from two RTX PRO 6000 Max-Q cards
over PCIe, with the whole model resident on the GPUs. It starts from turboderp's ExLlamaV3 engine, then adds a
tensor-parallel runtime, a quantisation recipe, a speculative-decoding stack and a set of new kernels written for
this model and these cards. About 22,600 lines across 254 commits are this project's own work.

### Built here

- **The TP=2 runtime.** Sharded checkpoint format and bit-exact shard tool, the per-GPU runner and the
  OpenAI-compatible server, and a one-shot CUDA-IPC all-reduce over PCIe (4.9 µs per call vs NCCL's 11.8).
- **Captured decode and verify.** Piecewise CUDA graphs for every decode and speculative verify step, the captured
  single-token attention body, cross-block graph merging, and capture sized from fixed capacities so mixed request
  lengths never re-capture (13.15 → 10.97 ms per token from graphs alone).
- **The mixed-bitrate checkpoint.** The 2.565 and 2.499 bpw builds: experts ranked by routing frequency and given 2,
  3 or 4 bpw each, with mixed per-expert rates decoded inside one MoE launch (new mixed-rate kernel variants,
  shared-expert fold, 256-thread blocks, prefetch-ring hoist and programmatic dependent launch).
- **FP4 KV cache.** Packs the main cache in DeepSeek's trained FP4 layout and decodes it in-kernel: 3,200 → ~890
  bytes per token per GPU, which is what makes the 5M-token shared cache fit.
- **Speculative decoding tuned for this checkpoint.** Adaptive K: the verify width is chosen every round on the GPU
  from the drafter's confidence, with a cutoff that adapts to the content. Also a per-user drafter ring instead of a
  context-sized drafter cache, the drafter captured as GPU graphs, and the hot/cold drafter experts.
- **Memory and long-context work.** Adaptive prefill chunking from a measured memory model (an 876k-token prompt
  completes), indexer score slabs, long-context candidate compaction, the prefix cache made to work across both
  ranks, an exact multi-row router, and the shared-page prefill fix.
- **Measurement.** Every default here was chosen by back-to-back A/B with token-identity gates; the numbers and the
  ideas that did not pay off are in [docs/tp2/RESULTS.md](docs/tp2/RESULTS.md).

### Built on, with thanks

| Who | What this project uses | What we did with it |
|---|---|---|
| **[turboderp — ExLlamaV3](https://github.com/turboderp-org/exllamav3)** (MIT) | The engine underneath: the EXL3 format, trellis quantiser and decode kernels, the cooperative MoE kernel, DeepSeek-V4 attention and the DSpark drafter component, the paged cache and recurrent checkpoints, loaders and generator. | Everything listed under *Built here* sits on top of it. Upstream's history and authorship are kept in full. |
| **[diffbot](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)** (MIT) | The reference deployment on the same cards; an optional drafter head (`DSPARK_HEAD=diffbot`); the idea of an 8-bit peer-push prefill all-reduce. | Our own all-reduce with int8 and exact modes (`PREFILL_AR`, opt-in). We also tested their flat MoE loop's weight streaming and their row top-k: bit-identical but slower on our mixed-rate checkpoint, and not needed by our indexer ([RESULTS](docs/tp2/RESULTS.md) section 11d). Our drafter head is faster on every content type we tested. Their AGPL vLLM patches were not read or used. |
| **philbert440 — 1Cat-vLLM PR #1243** | The idea of scaling the drafter's input taps by a power of two before the fp16 cast. | Implemented and measured (`EXL3_DSPARK_TAP_SCALE`); neutral here, so the default stays 1.0. |
| **[dealignai](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8)** (MIT, gated) | The base FP8 checkpoint. | Re-quantised into our mixed-bitrate EXL3 builds. |
| **sudoingX, HeNryous** | Published quantisation panels. | sudoingX's panel protocol is reproduced exactly (`qmesh/tp2/panel_ppl.py`) to score our builds against theirs. |
| **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT) | The WikiText-2 perplexity protocol. | Our prefill-chunk invariance gate. |

Reference numbers only, no code taken: sfxnz, MiaAI-Lab, SSHdotCodes, sybil-solutions, acrogenesis, llamaperf.
Full detail: [CREDITS.md](CREDITS.md). The code is MIT; the licence text is unchanged in [LICENSE](LICENSE).

## Model and layout

**DeepSeek-V4.1-Flash-1017**, mixed-bitrate EXL3: 2.499 bpw routed-expert average by default (1,400 hottest experts
at 4 bpw, 4,860 at 3, 9,100 at 2); a 2.565 bpw build is the higher-quality alternative. Fully resident on two 96 GB
GPUs, true TP=2, speculative decoding on.

Upstream's tree is kept intact (`exllamav3/`, `examples/`, `tests/`, `util/`, `doc/`) with this project's engine
changes inside it; the TP=2 runner, server and tools live in `qmesh/tp2/` (the directory name is historical) and are
documented in `docs/tp2/`. Upstream's own README is preserved as [README.exllamav3.md](README.exllamav3.md).

Status: **v1.2.0** (hot/cold drafter experts; final measurements below).

## Features

| Feature | Default | What it does | Measured |
|---|---|---|---|
| **True TP=2** | on | Attention heads, KV heads, vocab and the MoE intermediate dimension split two ways; one process per GPU. | — |
| **One-shot P2P all-reduce** (`COMM=p2p`) | on | CUDA-IPC arena over PCIe, device-side sequence counters, safe inside graphs. | 4.93 µs/call vs NCCL 11.84 µs |
| **Captured decode and verify** | on | Each decode step and each speculative verify step replays CUDA graphs instead of ~1,600 launches; graphs are sized from fixed capacities, so mixed short/long requests never re-capture. | 13.15 → 10.97 ms/token |
| **DSpark speculative decoding** (`DSPARK=1`) | on in the server | The drafter proposes 5 tokens per round as one captured block; the target verifies them in one captured step and keeps the longest correct prefix, so output is decided by the target alone. | 93.7 → 130–185 tok/s by content |
| **Adaptive K (verify width per round)** (`EXL3_DSPARK_GATE=auto`) | on | Each round, the drafter's confidence head scores its 5 positions and only the confident prefix is verified (K = 0–5 rows, chosen on the GPU inside the captured graph, no host round trip). The confidence cutoff itself adapts to the content: an average of accepted drafts per round switches between a bold cutoff on low-acceptance text (prose, multilingual) and a strict one on high-acceptance text (math, structured). Both ranks agree by construction. | 24k prompts, auto vs always verifying 5: prose 113 vs 98, code 139 vs 141, math 176 vs 181 tok/s. Retuned for 2.499: +6.8% on 7 prompts. The best fixed width per content is still 1–9% faster (code: K=3, 152), which the V2 backlog targets. |
| **Hot/cold drafter experts** (new) | on | Drafter routed experts that take 99% of routing picks at 4 bpw, the rest at 2 bpw (`qmesh/tp2/drafter_experts.py`). | 4-bit acceptance for 0.12 GiB/GPU over 3-bit |
| **FP4 KV cache** (`KV_BITS=4`) | on | Main cache in DeepSeek's trained FP4 format, decoded in-kernel. | 3,200 → ~890 bytes/token/GPU; decode within 1% of 16-bit |
| **Adaptive prefill chunk** (`PREFILL_ADAPTIVE=1`) | on | Before every chunk, reads free memory and solves a measured peak model (rows + context depth) for the largest chunk that leaves 1 GiB free. Only ever shortens a chunk; both ranks agree. | 876k-token prompt completes (chunks 6,656 → 3,328 rows); ~0.57 ms per decision |
| **Drafter window cache** | on | The drafter keeps a small per-user ring, not a context-sized cache. | +1.5 GiB free at 512k |
| **Prefix cache** (`EXL3_PREFIX_CACHE=1`) | on | Content-hashed pages plus recurrent-state checkpoints; repeated prefixes skip prefill. | 36–40× faster first token at 20k, 86–88× at 60k |
| **Multi-user batching** (`BATCH=N`) | opt-in | Concurrent streams share one cache and one scheduler; expert and dense layers run batched. | 8 users × 2k: 255 tok/s total |
| **Mixed per-expert rates in one launch** | on | Every expert at its own rate (2, 2.5, 3, 3.5, 4, 6, 8 bpw) inside a single MoE kernel. | — |
| **Exact routing** | on | The multi-row router is bitwise identical to single-token decode. | — |
| **OpenAI-compatible server** | — | `qmesh/tp2/serve_tp2.sh start\|stop\|status`: chat and completions, streaming, tool calls, cached-token usage, API key. | [docs/tp2/SERVER.md](docs/tp2/SERVER.md) |
| **Memory report** (`MEM_BREAKDOWN=1`) | opt-in | Per-GPU resident GiB by owner after load and the peak since load. | — |

## Performance (v1.2.0)

Measured on 2 × RTX PRO 6000 Blackwell Max-Q (PCIe, power-capped), shipped defaults (DSpark K=5 with the adaptive
gate, FP4 KV, graphs, CPU-pinned ranks), greedy, 256 generated tokens, pinned GEMM tuning. Full tables, settings and
logs: [docs/tp2/RESULTS.md](docs/tp2/RESULTS.md) section 12.

**Single user, decode tok/s by content** (24k-token prompts from the corpus; two runs each):

| Content | tok/s | Tokens per verify step |
|---|---|---|
| Structured (JSON, tables) | 185 | 4.18 |
| Math | 162–164 | 4.38 |
| Code | 130 | 2.88 |
| Prose | 113–114 | 2.46 |
| Multilingual | 91–93 | 1.89 |
| 7 short prompts | 157–159 | 3.50 |
| 40 held-out prompts (code, prose, Q&A, mixed) | 157 | 3.26 |

Code, prose and multilingual are slower because the drafter's guesses are accepted less often on them (fewer tokens
per verify step); a verify step costs the same on every content type.

**Concurrent users** (one shared cache, every user prefills, 256 tokens each; decode throughput excludes prefill):

| Users | 2k-token prompts: total / per user tok/s | 24k-token prompts: total / per user tok/s | Last user's first token, 2k / 24k |
|---|---|---|---|
| 1 | 156 / 156 | 131 / 131 | 0.8 s / 8.3 s |
| 2 | 147 / 73 | 147 / 73 | 1.6 s / 16.7 s |
| 4 | 176 / 44 | 163 / 41 | 3.2 s / 33.7 s |
| 8 | 255 / 32 | 230 / 29 | 6.4 s / 66.7 s |
| 16 | 215 / 13 | 254 / 16 | 12.9 s / 133 s |

**Context limits** (FP4 KV, v1.2 drafter):

| Limit | Measured |
|---|---|
| Largest shared KV cache that loads and serves | **5,242,880 tokens** (5M; 4.35 GiB KV per GPU, 2.47 GiB free after load). 6M loads with 1.61 GiB free but runs out of memory in decode; 8M fails to load. |
| 8 users × 131k tokens each in that 5M cache | all 8 answer all 5 planted facts (40/40); peak 92.5 GiB of 95 per GPU |
| 1 user, 131k-token prompt in the 5M cache | 5/5 planted facts; first token 49.9 s; decode 174 tok/s |
| Longest single prompt | 876,328 tokens in a 1M cache (the model's position limit is 1,048,576): both facts at 10% and 90% depth answered; prefill 352 s (2,490 tok/s) with adaptive chunking |

**Cold prefill** (no prefix cache): ~2.5–3.0k tok/s (24k prompt → first token in 8.3 s; 131k in 50 s; 876k in 352 s).

These cards are power-capped and drift, so only back-to-back interleaved A/B numbers are comparable; read the note
in RESULTS.md before comparing.

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
| This build, 2.565 checkpoint (abliterated base) | 2.565 | 1.6886 | 1.7041 |
| **This build, 2.499 checkpoint (default, abliterated base)** | **2.499** | **1.6998** | **1.7138** |

Two caveats: the other rows were scored by their authors' harness on DGX Spark, this row by
`qmesh/tp2/panel_ppl.py` through the TP=2 runtime (same panels and positions, different code path); and our base is
an abliterated fine-tune, so this row's gap to the original includes that edit as well as quantisation. The drafter
does not affect these numbers: speculative decoding only changes speed, the target model decides every token.

Prefill-chunk invariance: the same text scores the same at 256-, 2048- and 4096-row prefill chunks (8k-token
WikiText-2 windows: 2.1820 / 2.1827 / 2.1827, ±0.025).

## Hardware and software

- 2 × NVIDIA RTX PRO 6000 Blackwell (96 GB), PCIe peer-to-peer, one NUMA node. Needs `sm_120a`.
- Python ≥ 3.10, PyTorch with CUDA 12.8+ and `sm_120a`, the CUDA toolkit, `ninja`, `gcc`.
- About 88 GiB per GPU for weights, drafter and graphs (2.499 checkpoint, 7.0 GiB free after load); the rest is KV
  cache. FP4 KV (default) costs ~890 bytes per token per GPU, 16-bit (`KV_BITS=16`) 3,200.

Install the way you install upstream — the extension builds on first import:

```sh
git clone https://github.com/GPTwareAiTeam/DSV41-Flash-EXL3-2xRTXPRO6000.git
cd DSV41-Flash-EXL3-2xRTXPRO6000
pip install -e .
TORCH_CUDA_ARCH_LIST=12.0a python -c "import exllamav3"   # JIT-builds the extension
```

Set `TORCH_EXTENSIONS_DIR` to a persistent path, or every restart rebuilds.

## Getting the checkpoint

Download the pre-sharded checkpoint from Hugging Face (link on release), or build one: `qmesh/tp2/shard_tool.py`
produces `rank0/` and `rank1/` from a full EXL3 checkpoint; see [docs/tp2/RECIPE.md](docs/tp2/RECIPE.md) and
[docs/tp2/SHARD_SPEC.md](docs/tp2/SHARD_SPEC.md). The v1.2 drafter experts are built with
`qmesh/tp2/quant_drafter_experts.py` (2- and 4-bit sets) and `qmesh/tp2/drafter_experts.py --tiers n99:4,rest:2`
using the shipped usage table `qmesh/tp2/data/drafter_expert_usage.json`.

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

- Cold prefill is ~2.5–3.0k tokens/s; a 131k prompt takes 50 s to its first token, and with 8 users each
  prefilling 131k the last first token arrives after ~7 minutes. Repeated prefixes hit the prefix cache and skip that.
- Multi-user throughput scales weakly: decode attention runs one user at a time (only the expert and dense layers
  are batched), so total decode peaks around 250 tok/s at 8–16 users. The batched attention path
  (`EXL3_DSV41_BATCH=1`) is slower on this model and not token-identical, so it is off.
- Greedy output depends on the batch width (number of rows verified together) and on the drafter: the same prompt
  can produce a different continuation alone than alongside other users. Flips sit on near-tied tokens; see
  RESULTS.md section 6.
- The drafter was trained on the original model; on the 2.499 checkpoint it is accepted less often on code, prose
  and multilingual text. Its precision is not the limit (6- and 8-bit drafter attention moved acceptance ≤2%);
  retraining it on the 2.499 checkpoint is the planned fix. Planned work: [docs/tp2/V2_BACKLOG.md](docs/tp2/V2_BACKLOG.md).
