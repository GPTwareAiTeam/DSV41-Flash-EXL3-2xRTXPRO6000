# What others did for speed on THIS model (researched 2026-10-27)

Sources: philbert440/DeepSeek-V4.1-Flash-1017-EXL3-franken-v100 (HF card, 8×V100, 1Cat-vLLM SM70 port,
PR #1243), acrogenesis/deepseek-v41-exl3-recipe-kit (single 5090 + 128 GB RAM, SSD streaming),
llamaperf entries, Atomic Chat EXL3-vs-GGUF guide, X posts (MiaAI_lab TensorFold/DFlash2, Tech2Wild,
ViC305 SAGE-EXL3, mr_r0b0t buun-llama).

## Their measured numbers on the SAME model (8×V100, PP2×TP4, 1M ctx)
- plain decode 1 stream: **30.5 tok/s** (we: 76 tok/s on 2×RTX PRO 6000)
- DSpark k=5, 1 stream: 64.9-65.9 tok/s code/JSON, 34.5-35 prose/chat; **AL 4.35-4.38 tokens/verify step**
  (we: 101-135 tok/s, AL 2.6-3.3)
- prefill: 2,985 / 3,231 / 3,194 tok/s at 8K / 32K / 128K (we: 3,527 @2048, 4,141 @4096)
- decode at long context: 51 tok/s at 8K and at 128K
- So: we are ahead on raw decode and prefill; **they are well ahead on tokens-per-verify-step (AL 4.35 vs our 3.0)**
  which multiplies straight into throughput.

## Their engine work (all gated against an fp32 oracle, behind VLLM_DS41_* knobs)
1. **DSpark fp16 tap fix** — the single biggest acceptance bug they found: taps were captured in fp16 and
   overflowed to inf on attention-sink tokens at the start of a chat, leaving NaN rows in the drafter window →
   zero drafts for ~100 steps, **42% of steps**; first-draft acceptance on code 0.49 → **0.90** by scaling the
   taps down before the fp16 cast. Their decode went 43.5 → 62.7 tok/s (code), 29.4 → 35.3 (chat).
   Relevance: our export path (`transformer.py`, export_state branch) clamps to ±65504 instead of scaling, so
   saturation is plausible. The drafter's first op on taps is `main_norm` (RMSNorm) ⇒ a fixed power-of-2 scale
   is mathematically invariant and only removes overflow.
2. **Per-layer CUDA-graph decode attention over a packed FP4 compressed KV cache** (their ATTN_GRAPH=1). In
   vLLM one layer's decode attention is a single fused kernel, so capture is cheap; in ExLlamaV3 it is ~11 ops
   per layer, which is why our capture keeps failing. Their KV is packed FP4 (vs our fp16).
3. **Decoder replay in prefill**: layers 21-39 run only the request's last 3,072 rows (exact) — prefill win.
4. **Tiled EXL3 prefill MoE kernel** (our prefill MoE is 30.6% of prefill time).
5. **Fused hyper-connection kernels** (HC is 8.5% of our prefill; their HC_FUSED/HC_NORM_EXACT).
6. **Graph-capturable fused decode indexer** + fused indexer kernels.
7. **Engram rows served from host RAM with hugepages; gather is a memcpy**; they warn that Engram reads
   256-byte rows at random and a 1 MB dataset record size amplifies reads ~500× (use 16K records).
8. **DSpark drafter in per-batch-size CUDA graphs**; `--max-num-seqs 2`, `num_speculative_tokens 5`.
9. **Batch-invariant dense GEMMs** (reproducibility, not speed).
10. **Layer-sliced prefill between decode steps** (interleaving for multi-stream serving).

## Community context (other EXL3 setups)
- 2×RTX 5090, GLM-5.3-Flash 321B MoE 2.05bpw EXL3: 64.8 tok/s decode, 1,973 tok/s prefill.
- EXL3 3.0 bpw on a single modern GPU (Atomic Chat): 119 tok/s decode, 4,471 tok/s prefill, TTFT 0.46 s.
- MiaAI TensorFold + DFlash2 k=7 on 2×DGX Spark: 63 tok/s single stream, 155 tok/s on 4 sessions.
- Single 5090 + 128 GB RAM with SSD streaming (this model): 9.5-30.7 tok/s — i.e. our 100+ tok/s is far ahead of
  streaming setups; the competitive reference is the 8×V100 box above.

## Action list derived (priority order)
A. **DSpark tap scale fix** (small, testable, potentially +40% throughput via AL).
B. Packed FP4/FP8 KV cache (memory + attention bandwidth; prerequisite for a cheap attention graph).
C. Prefill: decoder replay for late layers, tiled prefill MoE, fused HC.
D. Fused/merged decode attention kernels so the attention core becomes 1-2 launches per layer (then capture).

## Findings after implementing the community items (2026-10-27)
- **DSpark tap saturation: REAL but not the throughput bug here.** Our taps reach ~271k at sequence positions 0/2
  (fp16 max 65504 → 9 saturated values per tap layer). Scaling by 1/64 removes the saturation completely
  (`EXL3_DSPARK_TAP_SCALE`, commit 4a02976e, default off) but AL was mixed: 3.09/3.30/2.58 → 2.84/3.49/2.68.
  Our clamp already prevented the inf/NaN cascade the community hit (they got zero drafts for 42% of steps; we don't).
- **Acceptance decomposition** (77 verify rounds): first draft 85.7%, then 57.6 / 57.9 / 63.6 / 57.1 % conditional.
  So the drafter chain beyond the first draft is the limiter. Community chat/prose AL 2.25–2.27 ≈ ours (2.58–2.68);
  their code AL 4.38 vs our 3.30 — comparable content, so their drafter chain is stronger on code.
- **Prefill fused HC** (tiled `hc_mix`, 4 rows/block, R>=512): bit-exact (maxabs 0.0), commit cf6946a1:
  2048 3502.6 → 3542.2 tok/s, 4096 4131.5 → 4218.5 tok/s.
