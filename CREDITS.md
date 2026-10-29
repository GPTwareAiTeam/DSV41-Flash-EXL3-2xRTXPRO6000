# Credits

QMesh is a runtime delta on top of a large amount of other people's work. Every entry below is a dependency of
the shipped runtime or of the published checkpoint, or a source the results are compared against.

## Upstream engine

**Turboderp — [ExLlamaV3](https://github.com/turboderp-org/exllamav3)** (MIT, see [LICENSE](LICENSE)).
The engine, the EXL3 quantisation format and trellis/codebook decode, the mixed-bitrate MoE kernels, the
sampler, the cache design and the model loaders are upstream work. This repository is a derivative work
throughout: it adds a tensor-parallel (TP=2) backend, a pre-sharded checkpoint format, the runner and server, and
the decode/speculation changes described in [docs/tp2/](docs/tp2/). Upstream's README is preserved as
[README.exllamav3.md](README.exllamav3.md) and its copyright notice is retained as MIT requires.

## Model

**DeepSeek — [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (MIT).
The model architecture (DSA sparse attention with indexer and compressed KV, hyper-connections, MoE) and the
DSpark / multi-token-prediction drafter design that this runtime's speculative path implements.

**dealignai — [DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8)**
(MIT; gated: 18+, lawful use, user accepts responsibility for outputs). The base checkpoint our mixed-bitrate
EXL3 quantisation was made from. It is an abliterated (refusal-removed) fine-tune of DeepSeek-V4.1-Flash-1017;
quality figures we publish are measured against reference outputs of the original model and therefore include
the effect of that edit, not only of quantisation.

## Third-party code compiled into the runtime

- **MoE decode candidates kit (`decode_moe_candidates`)** — produced by the project owner's external agent and
  provided to this project. Two of its candidates ship in the MoE cooperative decode kernel
  (`exllamav3/exllamav3_ext/quant/exl3_moe_coop*.cu[h]`): **C1, the prefetch-ring hoist** (`EXL3_MOE_HOIST`:
  the weight prefetch is issued before the block prologue) and **C5, programmatic dependent launch of kernel B**
  (`EXL3_MOE_PDL`, `EXL3_MOE_COOP_PDL`). Both are bitwise identical to the baseline arithmetic. The kit's CPU
  emulator, `bench_moe_coop.py` benchmark harness and its memory-latency classification rule were used to
  evaluate the MoE kernel.
- **Diffbot drafter head** (MIT) — optional only (`DSPARK_HEAD=diffbot`). The converter
  `qmesh/tp2/drafter_overlay.py` reads their published drafter pack; no diffbot weights are redistributed.
  Licence and attribution: [docs/tp2/diffbot-drafter-k7.LICENSE.txt](docs/tp2/diffbot-drafter-k7.LICENSE.txt),
  [docs/tp2/diffbot-drafter-k7.CREDITS.md](docs/tp2/diffbot-drafter-k7.CREDITS.md). Their AGPL vLLM patches
  were not used.
- **Flash Linear Attention (FLA)** — Songlin Yang, Yu Zhang, Zhiyuan Li — MIT, vendored under
  `exllamav3/vendor/fla/` with its licence intact.
- **ROCm WMMA GEMM** — Adel Johar — MIT, vendored under `exllamav3/exllamav3_ext/rocm/vendor/rocm_wmma_gemm/`
  with its licence intact.

## Reference data (no code taken)

Used for targets, comparisons and cross-checks only:

- **sudoingX** and **HeNryous** — published DeepSeek-V4.1-Flash quantisation panels (perplexity and KL against
  the original model); our perplexity scorer reproduces sudoingX's panel protocol exactly.
- **diffbot** — the primary comparable deployment (same GPU class, TP=2, PCIe) and its published throughput and
  acceptance figures.
- **sybil-solutions**, **philbert440 / 1Cat-vLLM PR #1243**, **sfxnz**, **MiaAI-Lab**, **SSHdotCodes**,
  **acrogenesis**, **llamaperf** — public performance reports used as reference points.

## Tools

**PyTorch** (runtime and CUDA graph capture), **NCCL** (rendezvous and prefill reduction), **CUDA** (P2P
all-reduce kernel, graph API), **Triton**, **safetensors**, **tokenizers** and **Ninja**.

## Licence

MIT, inherited from upstream; the text in [LICENSE](LICENSE) is unchanged. Model weights are not part of this
repository; the original model (DeepSeek) and the base checkpoint (dealignai) are both published under MIT.
