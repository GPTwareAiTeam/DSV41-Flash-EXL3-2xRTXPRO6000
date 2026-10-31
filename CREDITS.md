# Credits

This project is a TP=2 runtime, quantisation recipe and kernel set for DeepSeek-V4.1-Flash-1017 on two RTX PRO 6000
Max-Q cards, built on turboderp's ExLlamaV3. This file says what comes from where: what upstream provides, what other
people's code or ideas we used, and what was written here. [README.md](README.md) lists this project's own work in
detail; [docs/tp2/RESULTS.md](docs/tp2/RESULTS.md) has the measurements behind it.

## Upstream engine

**turboderp — [ExLlamaV3](https://github.com/turboderp-org/exllamav3)** (MIT, see [LICENSE](LICENSE)).
The engine this runtime builds on: the EXL3 quantisation format, the trellis quantiser and decode kernels, the
cooperative MoE kernel, the DeepSeek-V4 attention and DSpark drafter component, the paged cache, recurrent
checkpoints, sampler, loaders and generator. Upstream's full history and authorship are kept in this repository, its
README is preserved as [README.exllamav3.md](README.exllamav3.md), and its copyright notice is retained as MIT
requires. Upstream contributors' commits (turboderp and others) are upstream work; this project's commits begin on
2026-10-25.

## Model and base checkpoint

- **DeepSeek-V4.1-Flash-1017** (MIT): the model and the DSpark multi-token-prediction drafter design.
- **dealignai — [DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8](https://huggingface.co/dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8)**
  (MIT; gated: 18+, lawful use, user accepts responsibility for outputs): the base checkpoint our mixed-bitrate EXL3
  builds were quantised from. It is an abliterated (refusal-removed) fine-tune, so the quality figures we publish,
  measured against the original model's outputs, include the effect of that edit as well as quantisation.

## Other people's code or ideas used here

- **[diffbot — DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000)** (MIT):
  - **Drafter head, optional** (`DSPARK_HEAD=diffbot`). Our converter `qmesh/tp2/drafter_overlay.py` reads their
    published drafter pack; no diffbot weights are redistributed. Licence and attribution:
    [docs/tp2/diffbot-drafter-k7.LICENSE.txt](docs/tp2/diffbot-drafter-k7.LICENSE.txt),
    [docs/tp2/diffbot-drafter-k7.CREDITS.md](docs/tp2/diffbot-drafter-k7.CREDITS.md). Our own drafter head is the
    default and is faster on every content type we tested.
  - **Idea: an 8-bit peer-push prefill all-reduce** (their `fp8ar`). `PREFILL_AR` (`qmesh/tp2/prefill_ar.py`,
    `exllamav3_ext/parallel/prefill_ar.cu`) is our own implementation, with int8 and exact modes; opt-in.
  - **Studied and measured, not shipped:** their flat decode MoE loop (cp.async weight ring, 128-wide K tiles,
    swapped MMA) and length-aware row top-k (`kernels/xmoe`, `kernels/rowtopk`, MIT). The weight-streaming idea, ported
    to our mixed-rate kernel, was bit-identical but slower (branch `tp2/moeflat-1031`, not merged); our indexer
    already scans only the live width that row top-k targets ([RESULTS](docs/tp2/RESULTS.md) section 11d).
  - Their deployment on the same cards is the reference point our numbers are compared against.
  - Their AGPL vLLM patches were not read or used.
- **philbert440 — 1Cat-vLLM PR #1243** — idea only: scale the drafter's input taps by a power of two before the fp16
  cast. Implemented and measured as `EXL3_DSPARK_TAP_SCALE` (`exllamav3/modules/transformer.py`); neutral here, so the
  default stays 1.0.
- **sudoingX** and **HeNryous** — published DeepSeek-V4.1-Flash quantisation panels. `qmesh/tp2/panel_ppl.py`
  reproduces sudoingX's panel protocol exactly, so our builds are scored on the same positions as theirs.
- **[llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT) — the WikiText-2 perplexity protocol (`perplexity.c`),
  used by `qmesh/tp2/panel_ppl.py` as our prefill-chunk invariance gate.

## Written for this project

All of the TP=2 runtime and the items under *Built here* in [README.md](README.md): the sharded checkpoint and shard
tool, the one-shot P2P all-reduce, captured decode and verify graphs, the mixed per-expert-rate MoE kernels, FP4 KV,
the adaptive drafter gate and drafter ring, hot/cold drafter experts, adaptive prefill chunking, the prefix cache
across ranks, the exact multi-row router, the runner, the server, and the measurement harness.

Two of the MoE kernel changes, the prefetch-ring hoist (`EXL3_MOE_HOIST`) and programmatic dependent launch of the
second MoE kernel (`EXL3_MOE_PDL`, `EXL3_MOE_COOP_PDL`), came from a candidate kit produced for this project (with a
CPU emulator, a benchmark harness and a memory-latency classification rule used to evaluate them); both are bitwise
identical to the previous arithmetic.

## Reference data only (no code taken)

**sybil-solutions**, **sfxnz**, **MiaAI-Lab**, **SSHdotCodes**, **acrogenesis**, **llamaperf** — public performance
reports used as reference points.

## Tools

**PyTorch** (runtime and CUDA graph capture), **NCCL** (rendezvous and prefill reduction), **CUDA**, **Triton**,
**safetensors**, **tokenizers** and **Ninja**.

## Licence

MIT, inherited from upstream; the text in [LICENSE](LICENSE) is unchanged. Model weights are not part of this
repository; the original model and the base checkpoint (dealignai) are both published under MIT.
