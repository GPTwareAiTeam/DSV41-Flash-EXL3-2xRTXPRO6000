/* qmesh_moe_fused: the cold-expert MoE forward as one launch per projection.
 *
 * The shipped path is four kernels per bitrate group:
 *
 *     had_in(x)  -> a13 (rows x H fp16)          exl3_moe_had_in
 *     gemm13(a13)                                exl3_moe_gemm   (w13 = 2*I columns)
 *     glu(inter) -> a2 (rows x I fp16)           exl3_moe_glu_had_in
 *     gemm2(a2, accumulate into the token rows)  exl3_moe_gemm   (H columns)
 *
 * Every arrow is a full write and read of an activation tensor whose only consumer is the next
 * kernel in the list, and the first one is written *twice* (the w13 trellis is one tensor with two
 * shard groups, and the vendor kernel indexes its A slice as `A + grp*m*k`, so had_in stores a copy
 * per group even though the groups differ only in the suh scale they are transformed with).
 *
 * These kernels fold the Hadamard transforms into the GEMMs that consume them:
 *
 *     stage 1: a13 is never built. Each block transforms the 128-wide k tile it is about to read
 *              (one warp per A row, `had128_warp_in`, straight into the shared A tile), runs the
 *              w13 GEMM, and writes the fp16 *output* transform to `inter` -- still needed there,
 *              because the down projection's k extent (I) is 18 blocks wide and no single block of
 *              the first stage owns a whole row of it.
 *     stage 2: a2 is never built. The A tile of each k step is `had128_warp_glu_in(inter_gate,
 *              inter_up)` over that k tile, then the down GEMM, then the routing-weighted combine.
 *
 * Numerics: bit for bit the same sums. The A values handed to the mma are the same fp16 values the
 * separate transform kernels store (the same `had128_warp_*` helper, the same rounding to fp16
 * before the tensor core sees it), the k accumulation order is ascending 16-element steps either
 * way, and the accumulator is fp32 (cuda-exl3's default; CUDA_EXL3_FP16_ACC is refused here because
 * a fused kernel has no second pass to fall back on). What differs is only the shared-memory tile
 * shape, which cannot change a sum.
 *
 * Tile: BM rows x BN=128 columns x BK=128, one warp row (WARPS_M = 1) and eight warps across n, so
 * each 16x16 trellis tile is decoded exactly once per block -- the vendor's own shape for
 * block_m=16. BK=128 rather than the vendor's 64 is what makes the fused transform possible: the
 * Hadamard is defined on 128 contiguous elements, which is exactly one A row of one k tile, and a
 * warp holds one such row as 4 halfs per lane, in lane order. So the A tile is *not* XOR-swizzled
 * (it cannot be: the transform writes it); its row stride is padded by one int4 instead (17 int4 =
 * 272 B), which puts the eight rows of an ldmatrix.x4 on eight distinct 4-bank groups.
 */
#include "qmesh_moe_cpp.h"

#include <cstdint>

#include <cuda_fp16.h>

/* The vendored device-side primitives: fragments, mma/cp.async wrappers, ldsm4, the trellis
 * decoders and the 128-point Hadamard. Read-only, shared with the vendor's own kernels. */
#include "../../vendor/cuda-exl3/src/cuda_exl3/csrc/exl3_dq.cuh"
#include "../../vendor/cuda-exl3/src/cuda_exl3/csrc/exl3_had.cuh"

namespace qmf {

constexpr int BN = 128;      /* == HAD_N: a block owns whole Hadamard blocks */
constexpr int BK = 128;      /* == HAD_N: one warp transforms one A row segment */
constexpr int NWARPS = 8;
constexpr int NTHREADS = NWARPS * 32;
constexpr int WARP_N = 16;   /* 16x16 trellis tile -> one warp, decoded once */
constexpr int WARPS_N = BN / WARP_N;
constexpr int NBLK = WARP_N / 8;   /* n8 fragments per warp */
constexpr int A_COLS = BK / 8;     /* int4 per A row */
constexpr int A_STRIDE = A_COLS + 1;
constexpr int KSTEPS = BK / 16;
constexpr int NTILES = BN / 16;

template <int BITS, int BM>
struct Cfg {
    static constexpr int MBLK = BM / 16;
    static constexpr int TILE_I4 = 2 * BITS;          /* trellis bytes per 16x16 tile, in int4 */
    static constexpr int TILE_U32 = 8 * BITS;
    static constexpr int SH_A_I4 = BM * A_STRIDE;
    static constexpr int SH_B_I4 = KSTEPS * NTILES * TILE_I4;
    static constexpr int SH_STAGE_I4 = SH_A_I4 + SH_B_I4;
    static constexpr int C_STRIDE = BN + 8;
    static constexpr int C_I4 = (BM * C_STRIDE * 2 + 15) / 16;
    static constexpr int SMEM_I4 = SH_STAGE_I4 * 2 > C_I4 ? SH_STAGE_I4 * 2 : C_I4;
};

/* fp32 accumulator, as cuda-exl3's Acc<false>: the element order of an mma D fragment is regs
 * (0,1),(2,3) for rows lane>>2 and (lane>>2)+8. */
struct AccF {
    using T = FragC;
    __device__ __forceinline__ static void zero(T& c) {
#pragma unroll
        for (int i = 0; i < 4; ++i) c[i] = 0.0f;
    }
    __device__ __forceinline__ static void mma(const FragA& a, const FragB& b, T& c) {
        mma_m16n8k16(a, b, c);
    }
    __device__ __forceinline__ static half geth(const T& c, int i) { return __float2half(c[i]); }
};

/* Trellis staging: KSTEPS x NTILES packed 16x16 tiles, same layout and indexing as the vendor's
 * load_stage, so a block reads exactly the bytes it always read. */
template <int BITS, int BM>
__device__ __forceinline__ void load_b(int4* sh, const uint16_t* __restrict__ Bq, int kt, int n0,
                                       int n_tiles_full, int t) {
    using C = Cfg<BITS, BM>;
    int4* sh_b = sh + A_STRIDE * BM;
    const int k0 = kt * BK;
#pragma unroll
    for (int i = t; i < C::SH_B_I4; i += NTHREADS) {
        const int chunk = i % C::TILE_I4;
        const int tile = i / C::TILE_I4;
        const int ks = tile / NTILES;
        const int nt = tile % NTILES;
        const int4* src = ((const int4*)Bq) +
                          ((size_t)(k0 / 16 + ks) * n_tiles_full + n0 / 16 + nt) * C::TILE_I4 +
                          chunk;
        cp_async16(sh_b + tile * C::TILE_I4 + chunk, src);
    }
}

/* MMA over one staged k tile. Ascending ks (and so ascending k) is what keeps the accumulation
 * order identical to the unfused path. */
template <int BITS, int CB, int BM>
__device__ __forceinline__ void compute_tile(const int4* sh, FragC (&frag_c)[Cfg<BITS, BM>::MBLK][NBLK],
                                             int warp, int lane) {
    using C = Cfg<BITS, BM>;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
    const int4* sh_a = sh;
    const uint32_t* sh_b = (const uint32_t*)(sh + A_STRIDE * BM);
#pragma unroll
    for (int ks = 0; ks < KSTEPS; ++ks) {
        FragA frag_a[C::MBLK];
#pragma unroll
        for (int mb = 0; mb < C::MBLK; ++mb) {
            const int r = (lane & 7) + 8 * ((lane >> 3) & 1);
            const int R = warp_m * BM + mb * 16 + r;
            const int c = ks * 2 + (lane >> 4);
            ldsm4(frag_a[mb], sh_a + R * A_STRIDE + c);
        }
        FragB frag_b[NBLK];
#pragma unroll
        for (int nb = 0; nb < NBLK; nb += 2) {
            const int nt = (warp_n * WARP_N + nb * 8) / 16;
            const uint32_t* tile = sh_b + (ks * NTILES + nt) * C::TILE_U32;
            dq_dispatch<BITS, CB, false>(tile, lane << 3, frag_b[nb], frag_b[nb + 1]);
        }
#pragma unroll
        for (int mb = 0; mb < C::MBLK; ++mb)
#pragma unroll
            for (int nb = 0; nb < NBLK; ++nb) AccF::mma(frag_a[mb], frag_b[nb], frag_c[mb][nb]);
    }
}

/* Shared C tile -> the Hadamard output transform's input. Same writer the vendor epilogue uses. */
template <int BITS, int BM>
__device__ __forceinline__ void store_c(half* sh_c, const FragC (&frag_c)[Cfg<BITS, BM>::MBLK][NBLK],
                                        int warp, int lane) {
    using C = Cfg<BITS, BM>;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;
#pragma unroll
    for (int mb = 0; mb < C::MBLK; ++mb) {
#pragma unroll
        for (int nb = 0; nb < NBLK; ++nb) {
            const int col = warp_n * WARP_N + nb * 8 + 2 * (lane & 3);
            half* p0 = sh_c + (warp_m * BM + mb * 16 + (lane >> 2)) * C::C_STRIDE + col;
            half* p1 = p0 + 8 * C::C_STRIDE;
            p0[0] = AccF::geth(frag_c[mb][nb], 0);
            p0[1] = AccF::geth(frag_c[mb][nb], 1);
            p1[0] = AccF::geth(frag_c[mb][nb], 2);
            p1[1] = AccF::geth(frag_c[mb][nb], 3);
        }
    }
}

__device__ __forceinline__ void zero_row(half* dst, int lane) {
    half4 z;
    z.x = __float2half2_rn(0.0f);
    z.y = __float2half2_rn(0.0f);
    ((half4*)dst)[lane] = z;
}

/* ---- stage 1: had_in + w13 GEMM + out-had ----------------------------------
 *
 * grid (n_total/BN, n_blocks): one block row per expert block of the alignment (block_m rows, all
 * the same expert), one block column per 128 output columns. The w13 trellis holds both shard
 * groups (gate, up), so the block column picks the group's suh scale -- the same two A copies the
 * unfused path built, without building them.
 */
template <int BITS, int CB, int BM>
__global__ __launch_bounds__(NTHREADS) void k_stage_w13(
    const half* __restrict__ x, const int* __restrict__ sorted_ids,
    const half* __restrict__ suh, half* __restrict__ inter, const uint16_t* __restrict__ bq,
    const half* __restrict__ svh, const int* __restrict__ eids, const int* __restrict__ n_rows,
    int k, int n_total, int shard_w, int n_shards, int64_t b_expert_stride, int m_valid, int top_k)
{
    using C = Cfg<BITS, BM>;
    extern __shared__ __align__(16) int4 smem[];
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;

    const int nlv = n_rows ? *n_rows : 0x7fffffff;
    const int m0 = blockIdx.y * BM;
    if (m0 >= nlv) return;   /* the alignment pads eids for the worst case; retire the surplus */
    const int e = eids[blockIdx.y];
    if (e < 0) return;

    const int n0 = blockIdx.x * BN;
    const int grp = n0 / shard_w;
    const uint16_t* Bq = bq + (size_t)e * b_expert_stride;
    const half* S = svh + (size_t)e * n_total + n0;
    const half* As = suh + ((size_t)e * n_shards + grp) * k;
    const int n_tiles_full = n_total / 16;
    const int kt_total = k / BK;

    /* The A tile: one warp per row, one Hadamard per row, straight into shared. Padding rows and
     * rows of a dead block carry zeros -- the same values cp.async's zero-fill used to supply. */
    auto load_a = [&](int stage, int kt) {
        half* sh_a = (half*)(smem + stage * C::SH_STAGE_I4);
        const int k0 = kt * BK;
        for (int r = warp; r < BM; r += NWARPS) {
            const int grow = m0 + r;
            const int idx = grow < nlv ? sorted_ids[grow] : -1;
            half* dst = sh_a + r * (A_STRIDE * 8);
            if (idx >= 0 && idx < m_valid)
                had128_warp_in<half>(x + (size_t)(idx / top_k) * k + k0, dst, As + k0, lane);
            else
                zero_row(dst, lane);
        }
    };

    FragC frag_c[C::MBLK][NBLK];
#pragma unroll
    for (int i = 0; i < C::MBLK; ++i)
#pragma unroll
        for (int j = 0; j < NBLK; ++j) AccF::zero(frag_c[i][j]);

    if (kt_total > 0) {
        load_a(0, 0);
        load_b<BITS, BM>(smem, Bq, 0, n0, n_tiles_full, t);
        cp_async_fence();
    }

    for (int kt = 0; kt < kt_total; ++kt) {
        const int slot = kt & 1;
        cp_async_wait<0>();   /* the B tiles of this slot have landed */
        __syncthreads();      /* ... and every warp is done with the slot's previous use */
        if (kt + 1 < kt_total) {
            load_b<BITS, BM>(smem + (slot ^ 1) * C::SH_STAGE_I4, Bq, kt + 1, n0, n_tiles_full, t);
            cp_async_fence();
            load_a(slot ^ 1, kt + 1);
        }
        compute_tile<BITS, CB, BM>(smem + slot * C::SH_STAGE_I4, frag_c, warp, lane);
    }

    /* Out transform, and with it the fp16 rounding the down projection's A tile must see. */
    __syncthreads();
    half* sh_c = (half*)smem;
    store_c<BITS, BM>(sh_c, frag_c, warp, lane);
    __syncthreads();
    for (int r = warp; r < BM; r += NWARPS) {
        const int grow = m0 + r;
        if (grow >= nlv) continue;
        const int idx = sorted_ids[grow];
        if (idx >= m_valid) continue;   /* padding row: stage 2 zeroes its own A tile */
        had128_warp_out<half>(sh_c + r * C::C_STRIDE, inter + (size_t)grow * n_total + n0, S,
                              lane);
    }
}

/* ---- stage 2: glu + had_in + down GEMM + combine ---------------------------
 *
 * grid (H/BN, n_blocks). The rows are already in routed order (stage 1's output), so the A tile is
 * a straight read of `inter`'s two halves; the epilogue is the fused combine, which accumulates
 * each routed row into its token's row of the fp32 answer.
 */
template <int BITS, int CB, int BM>
__global__ __launch_bounds__(NTHREADS) void k_stage_down(
    const half* __restrict__ inter, const int* __restrict__ sorted_ids,
    const half* __restrict__ suh, float* __restrict__ out, const uint16_t* __restrict__ bq,
    const half* __restrict__ svh, const int* __restrict__ eids, const int* __restrict__ n_rows,
    const float* __restrict__ w, int k, int n_total, int64_t b_expert_stride, int m_valid,
    int top_k)
{
    using C = Cfg<BITS, BM>;
    extern __shared__ __align__(16) int4 smem[];
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;

    const int nlv = n_rows ? *n_rows : 0x7fffffff;
    const int m0 = blockIdx.y * BM;
    if (m0 >= nlv) return;
    const int e = eids[blockIdx.y];
    if (e < 0) return;

    const int n0 = blockIdx.x * BN;
    const uint16_t* Bq = bq + (size_t)e * b_expert_stride;
    const half* S = svh + (size_t)e * n_total;
    const half* As = suh + (size_t)e * k;
    const int n_tiles_full = n_total / 16;
    const int kt_total = k / BK;

    auto load_a = [&](int stage, int kt) {
        half* sh_a = (half*)(smem + stage * C::SH_STAGE_I4);
        const int k0 = kt * BK;
        for (int r = warp; r < BM; r += NWARPS) {
            const int grow = m0 + r;
            const int idx = grow < nlv ? sorted_ids[grow] : -1;
            half* dst = sh_a + r * (A_STRIDE * 8);
            if (idx >= 0 && idx < m_valid) {
                const half* gate = inter + (size_t)grow * 2 * k + k0;
                had128_warp_glu_in<half>(gate, gate + k, dst, As + k0, lane);
            } else {
                zero_row(dst, lane);
            }
        }
    };

    FragC frag_c[C::MBLK][NBLK];
#pragma unroll
    for (int i = 0; i < C::MBLK; ++i)
#pragma unroll
        for (int j = 0; j < NBLK; ++j) AccF::zero(frag_c[i][j]);

    if (kt_total > 0) {
        load_a(0, 0);
        load_b<BITS, BM>(smem, Bq, 0, n0, n_tiles_full, t);
        cp_async_fence();
    }

    for (int kt = 0; kt < kt_total; ++kt) {
        const int slot = kt & 1;
        cp_async_wait<0>();
        __syncthreads();
        if (kt + 1 < kt_total) {
            load_b<BITS, BM>(smem + (slot ^ 1) * C::SH_STAGE_I4, Bq, kt + 1, n0, n_tiles_full, t);
            cp_async_fence();
            load_a(slot ^ 1, kt + 1);
        }
        compute_tile<BITS, CB, BM>(smem + slot * C::SH_STAGE_I4, frag_c, warp, lane);
    }

    __syncthreads();
    half* sh_c = (half*)smem;
    store_c<BITS, BM>(sh_c, frag_c, warp, lane);
    __syncthreads();
    for (int r = warp; r < BM; r += NWARPS) {
        const int grow = m0 + r;
        if (grow >= nlv) continue;
        const int idx = sorted_ids[grow];
        if (idx >= m_valid) continue;   /* padding row belongs to no token */
        had128_warp_out_acc<float>(sh_c + r * C::C_STRIDE, out + (size_t)(idx / top_k) * n_total + n0,
                                   S + n0, lane, w[idx]);
    }
}

/* A tile shape wider than 48 KiB needs the dynamic-shared attribute raised first; block_m=16 (the
 * only instantiated shape here) fits under the static limit, so this is a no-op it does not have to
 * know about. The call is kept so a future tile cannot silently fail to launch. */
template <class K>
cudaError_t raise_smem(K fn, int bytes) {
    if (bytes <= 48 * 1024) return cudaSuccess;
    return cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
}

#define QMF_ARGS_W13                                                                          \
    x, sorted_ids, suh, inter, bq, svh, eids, n_rows, k, n_total, shard_w, n_shards,           \
        b_expert_stride, m_valid, top_k
#define QMF_ARGS_DOWN                                                                         \
    inter, sorted_ids, suh, out, bq, svh, eids, n_rows, w, k, n_total, b_expert_stride,         \
        m_valid, top_k

template <int BITS, int CB, int BM>
int launch_w13(const half* x, const int* sorted_ids, const half* suh, half* inter,
               const uint16_t* bq, const half* svh, const int* eids, const int* n_rows, int k,
               int n_total, int shard_w, int n_shards, int64_t b_expert_stride, int m_valid,
               int top_k, int m_blocks, cudaStream_t s) {
    using C = Cfg<BITS, BM>;
    const int smem = C::SMEM_I4 * 16;
    const cudaError_t attrs = raise_smem(k_stage_w13<BITS, CB, BM>, smem);
    if (attrs != cudaSuccess) return 2;
    dim3 grid(n_total / BN, m_blocks);
    k_stage_w13<BITS, CB, BM><<<grid, NTHREADS, smem, s>>>(QMF_ARGS_W13);
    /* A refused launch leaves `inter` stale and `acc` untouched; the caller must not report that as a
     * computed layer, so it is an error here, not a silent zero. */
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

template <int BITS, int CB, int BM>
int launch_down(const half* inter, const int* sorted_ids, const half* suh, float* out,
                const uint16_t* bq, const half* svh, const int* eids, const int* n_rows,
                const float* w, int k, int n_total, int64_t b_expert_stride, int m_valid,
                int top_k, int m_blocks, cudaStream_t s) {
    using C = Cfg<BITS, BM>;
    const int smem = C::SMEM_I4 * 16;
    const cudaError_t attrs = raise_smem(k_stage_down<BITS, CB, BM>, smem);
    if (attrs != cudaSuccess) return 2;
    dim3 grid(n_total / BN, m_blocks);
    k_stage_down<BITS, CB, BM><<<grid, NTHREADS, smem, s>>>(QMF_ARGS_DOWN);
    return cudaGetLastError() == cudaSuccess ? 0 : 2;
}

#undef QMF_ARGS_W13
#undef QMF_ARGS_DOWN

/* The pack in service is 2 and 4 bit with no fractional bitrate (SEAM_MOE_NOEXT refuses those), and
 * every codebook id is instantiated: one branch per (bits, cb), one tile shape per block_m. A
 * combination that is not here returns 1 and the caller stays on the vendor path. */
#define QMF_TRY(B_, C_, BM_)                                                                   \
    if (bits == B_ && cb == C_ && block_m == BM_) return launch_w13<B_, C_, BM_>(              \
        x, sorted_ids, su, inth, bq, sv, ei, nr, k, n_total, shard_w, n_shards,                 \
        b_expert_stride, m_valid, top_k, m_blocks, s)
#define QMF_TRY_DOWN(B_, C_, BM_)                                                              \
    if (bits == B_ && cb == C_ && block_m == BM_) return launch_down<B_, C_, BM_>(             \
        ih, sorted_ids, su, oth, bq, sv, ei, nr, w, k, n_total, b_expert_stride,                \
        m_valid, top_k, m_blocks, s)

}  // namespace qmf

extern "C" int qmc_fused_w13(const void* x_h, const void* sorted, const void* suh, void* inter,
                             const void* trellis, const void* svh, const void* eids,
                             const void* n_rows, int k, int n_total, int shard_w, int n_shards,
                             int bits, int cb, int block_m, int64_t b_expert_stride, int m_valid,
                             int top_k, int m_blocks, cudaStream_t s) {
    const half* x = (const half*)x_h;
    const int* sorted_ids = (const int*)sorted;
    const half* su = (const half*)suh;
    const uint16_t* bq = (const uint16_t*)trellis;
    const half* sv = (const half*)svh;
    const int* ei = (const int*)eids;
    const int* nr = (const int*)n_rows;
    half* inth = (half*)inter;
    using namespace qmf;
    QMF_TRY(2, 0, 16);
    QMF_TRY(2, 1, 16);
    QMF_TRY(2, 2, 16);
    QMF_TRY(4, 0, 16);
    QMF_TRY(4, 1, 16);
    QMF_TRY(4, 2, 16);
    return 1;
}

extern "C" int qmc_fused_down(const void* inter, const void* sorted, const void* suh, void* out,
                              const void* trellis, const void* svh, const void* eids,
                              const void* n_rows, const void* w_f32, int k, int n_total, int bits,
                              int cb, int block_m, int64_t b_expert_stride, int m_valid,
                              int top_k, int m_blocks, cudaStream_t s) {
    const half* ih = (const half*)inter;
    const int* sorted_ids = (const int*)sorted;
    const half* su = (const half*)suh;
    const uint16_t* bq = (const uint16_t*)trellis;
    const half* sv = (const half*)svh;
    const int* ei = (const int*)eids;
    const int* nr = (const int*)n_rows;
    const float* w = (const float*)w_f32;
    float* oth = (float*)out;
    using namespace qmf;
    QMF_TRY_DOWN(2, 0, 16);
    QMF_TRY_DOWN(2, 1, 16);
    QMF_TRY_DOWN(2, 2, 16);
    QMF_TRY_DOWN(4, 0, 16);
    QMF_TRY_DOWN(4, 1, 16);
    QMF_TRY_DOWN(4, 2, 16);
    return 1;
}
