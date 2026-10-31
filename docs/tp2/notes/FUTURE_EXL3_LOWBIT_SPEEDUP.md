# Future work: make lower-bpw EXL3 experts actually run faster

Owner note (2026-10-26): real opportunity for an EXL3 modification — park it, do not work on it now.
Applies to any EXL3 model (both recipes), most valuable for low-bpw / cold experts.

## Observation (measured)
At bsz 1 the EXL3 MoE GEMV time does NOT scale with bitrate:
- Same expert count, all 2.0 bpw vs all 4.0 bpw: 51.4 µs vs 51.8 µs (2× bytes, +0.7% time). (CoopPerf.md)
- Mixed 2.56 bpw avg (70.8 MB) vs uniform 3.0 (79.6 MB): 118 vs 119 µs. (MixedK.md)
- Effective bandwidth ~600–800 GB/s vs measured 1.65 TB/s ceiling on RTX PRO 6000 Max-Q.
So 2.0-bpw experts buy memory but no speed today. A bandwidth-bound kernel would make 2.0 bpw ~1.5–1.7× faster than 3.5.

## Why (from code: exl3_gemv_kernel.cuh, codebook.cuh)
Every weight is expanded to FP16 in registers before the tensor core:
16-bit trellis window extract (SHF/BFE) → IMAD ×0x83DCD12D (mul1 hash) → IDP.4A byte-sum → HFMA2 scale/bias → HMMA m16n8k16 (FP16).
- Decode work per weight is identical at any bitrate (bitrate only changes how far the window slides).
- Kernel is latency-bound (dependent load→decode→MMA chain per warp, fixed per-block costs: input Hadamard rotation, cross-warp smem reduce, completion counters), not ALU- or bandwidth-bound (~25% of INT throughput).
- At bsz 1, 15/16 of each m16n8k16 MMA is wasted (only row 0 real).

## Directions to explore later
1. Latency hiding: async smem pipeline (cp.async / TMA if sm_120 supports it), warp specialization (producer loads, consumer decode+MMA), more tiles decoded per MMA. (Two-chain register ILP was tried and regressed ~4.7 µs — see MoEKernelTP2.md.)
2. Cheaper decode: amortize the window extract across adjacent weights; table-free decode tuned per K; packed multi-weight decode for K=2.
3. Fewer fixed costs: rotate input once per (expert, proj); fewer, larger blocks; persistent single launch (work queue).
4. bsz-1 specific path: avoid m16 MMA waste (FMA/dot-product path or pack multiple experts/rows into one MMA).
5. Get hardware counters first (ncu needs NVreg_RestrictProfilingToAdminUsers=0 or CAP_PERFMON) to confirm the dominant stall before redesigning.
6. Longer term: a format with native low-bit tensor-core support (NVFP4 on Blackwell) avoids software decode entirely, but requires requantizing.
