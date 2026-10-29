# Credits

QMesh is a small runtime delta on top of a large amount of other people's work. Nothing below is incidental —
each entry is a dependency of the shipped runtime.

## Upstream

**Turboderp — [ExLlamaV3](https://github.com/turboderp-org/exllamav3)** (MIT, see [LICENSE](LICENSE)).
The engine, the EXL3 quantisation kernels, the trellis/mixed-bitrate MoE kernels, the sampler, the cache design
and the model loaders are upstream work. This fork adds a tensor-parallel backend, a TP=2 sharded checkpoint
format and the runner; upstream files are otherwise unchanged and upstream's README is preserved as
[README.exllamav3.md](README.exllamav3.md).

## Model and architecture

**DeepSeek — DeepSeek-V4.1-Flash.** The model, its hybrid attention design (DSA indexer, compressed KV),
the hyper-connections and the multi-token-prediction drafter architecture that this runtime's speculative path
implements. The checkpoint in this repository is a quantisation of their released weights; no model weights
are distributed here.

**Diffbot.** Their MIT-licensed drafter-head converter is the reference this runtime's drafter path follows —
the conversion itself happens outside this tree and no diffbot code is redistributed here. Their published
TP=2 numbers on the same hardware class are the external comparison point cited in the results.

## Third-party code vendored in the tree

- **Flash Linear Attention (FLA)** — Songlin Yang, Yu Zhang, Zhiyuan Li — MIT, vendored under
  `exllamav3/vendor/fla/` with its licence intact at `exllamav3/vendor/fla/LICENSE`.
- **ROCm WMMA GEMM** — Adel Johar — MIT, vendored under `exllamav3/exllamav3_ext/rocm/vendor/rocm_wmma_gemm/`
  with its licence intact.

## Tools

**PyTorch** for the runtime and for CUDA graph capture, **NCCL** for the rendezvous and the prefill reduction,
**CUDA** for the P2P all-reduce kernel and the graph API, **safetensors**, **tokenizers** and **Ninja** on the
build and loading paths.

## Licence

MIT, inherited from upstream; the text in [LICENSE](LICENSE) is unchanged, and the copyright notice for
Turboderp's work is retained as MIT requires.
