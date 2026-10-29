// Fused decode-shaped MoE path (bsz 1..MAX_BSZN): launcher and tensor front end. Kernels in
// exl3_moe_coop_kernel.cuh, instantiated per (K, codebook) in comp_units/exl3_moe_coop_inst_k*.cu

#include <cuda_fp16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <map>
#include <cmath>

#include "../util.h"
#include "../util.cuh"
#include "exl3_devctx.cuh"
#include "exl3_moe_coop.cuh"
#include "comp_units/exl3_moe_coop_instances.cuh"
#include "bits_k.cuh"
#define EXL3_MOE_COOP_DEFINE_ROT
#include "exl3_moe_coop_kernel.cuh"

// Copy half weights as raw bits, including NaN payloads, and append exact half 1.0.
__global__ void exl3_moe_fold_stage_kernel(
    const int64_t* sel, const uint16_t* rw, int64_t* out_sel, uint16_t* out_rw,
    int rows, int topk, int expert, int64_t sel_s0, int64_t sel_s1,
    int64_t rw_s0, int64_t rw_s1)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int width = topk + 1;
    if (i >= rows * width) return;
    int row = i / width, col = i % width;
    if (col == topk)
    {
        out_sel[i] = expert;
        out_rw[i] = 0x3c00;
    }
    else
    {
        out_sel[i] = sel[row * sel_s0 + col * sel_s1];
        out_rw[i] = rw[row * rw_s0 + col * rw_s1];
    }
}

void exl3_moe_fold_stage(const at::Tensor& sel, const at::Tensor& rw,
                         at::Tensor& fold_sel, at::Tensor& fold_rw, int rows, int topk, int expert)
{
    const int n = rows * (topk + 1);
    if (!n) return;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    exl3_moe_fold_stage_kernel<<<(n + 255) / 256, 256, 0, stream>>>(
        (const int64_t*) sel.data_ptr(), (const uint16_t*) rw.data_ptr(),
        (int64_t*) fold_sel.data_ptr(), (uint16_t*) fold_rw.data_ptr(),
        rows, topk, expert, sel.stride(0), sel.stride(1), rw.stride(0), rw.stride(1));
}

// Output rows of a launch without routed slots (top-k 0): the empty sum, plus the shared-expert term
// when there is one. One warp per (row, 128-column chunk)
__global__ __launch_bounds__(32)
void exl3_moe_coop_empty_kernel(const MoeCoopParams p)
{
    exl3_moe_coop_ns::write_empty_row_chunk(p, blockIdx.x, blockIdx.y, threadIdx.x);
}

static MoeCoopKernel moe_coop_kernel_a(float K_, int cb, int Hi, bool wide)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    if (bk.half)
    {
        TORCH_CHECK(cb == 2, "exl3_moe_coop: half-integer bitrates require the mul1 codebook");
        switch (K)
        {
            case 1: return exl3_moe_coop_kernel_a_h1(Hi, wide);
            case 2: return exl3_moe_coop_kernel_a_h2(Hi, wide);
            default: return exl3_moe_coop_kernel_a_h3(Hi, wide);
        }
    }
    switch (K)
    {
        case 1: return exl3_moe_coop_kernel_a_k1(cb, Hi, wide);
        case 2: return exl3_moe_coop_kernel_a_k2(cb, Hi, wide);
        case 3: return exl3_moe_coop_kernel_a_k3(cb, Hi, wide);
        case 4: return exl3_moe_coop_kernel_a_k4(cb, Hi, wide);
        case 5: return exl3_moe_coop_kernel_a_k5(cb, Hi, wide);
        case 6: return exl3_moe_coop_kernel_a_k6(cb, Hi, wide);
        case 7: return exl3_moe_coop_kernel_a_k7(cb, Hi, wide);
        default: return exl3_moe_coop_kernel_a_k8(cb, Hi, wide);
    }
}

static MoeCoopKernel moe_coop_kernel_b(float K_, int cb, bool wide)
{
    const BitsK bk = bits_from_K(K_);
    const int K = bk.bits;
    if (bk.half)
    {
        TORCH_CHECK(cb == 2, "exl3_moe_coop: half-integer bitrates require the mul1 codebook");
        switch (K)
        {
            case 1: return exl3_moe_coop_kernel_b_h1(wide);
            case 2: return exl3_moe_coop_kernel_b_h2(wide);
            default: return exl3_moe_coop_kernel_b_h3(wide);
        }
    }
    switch (K)
    {
        case 1: return exl3_moe_coop_kernel_b_k1(cb, wide);
        case 2: return exl3_moe_coop_kernel_b_k2(cb, wide);
        case 3: return exl3_moe_coop_kernel_b_k3(cb, wide);
        case 4: return exl3_moe_coop_kernel_b_k4(cb, wide);
        case 5: return exl3_moe_coop_kernel_b_k5(cb, wide);
        case 6: return exl3_moe_coop_kernel_b_k6(cb, wide);
        case 7: return exl3_moe_coop_kernel_b_k7(cb, wide);
        default: return exl3_moe_coop_kernel_b_k8(cb, wide);
    }
}

// Split-k factor: blocks per column chunk, partial outputs summed by the last-arriving block.
// Measured ineffective.
// Split-k factor for one stage. The grid is slots x column tiles, which fills the device for a
// routed launch (bsz x top-k slots) but not for a single-slot launch such as the shared expert
// (one expert's tiles on a 100+ SM device): split k until there is a block per SM, keeping at
// least 64 k-slices (1024 columns) per split so the partials and their reduction stay cheap.
// EXL3_MOE_COOP_KSPLIT forces one factor for every stage (testing)
static int moe_coop_pick_ksplit(int base, int kslices, int max_split, int device)
{
    static int forced = -2;
    if (forced == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_KSPLIT");
        forced = env ? atoi(env) : 0;
    }
    if (forced > 0) return std::min(forced, max_split);
    const int want = CEIL_DIVIDE(DevCtx::instance().get_num_sms(device), std::max(base, 1));
    const int kcap = std::max(kslices / 64, 1);
    return std::max(1, std::min(std::min(want, kcap), max_split));
}

// Tile geometry per stage: the wide (128-column, 4-way k-split) tile for long k, the narrow one
// otherwise. EXL3_MOE_COOP_WIDE = 0 / 1 forces one geometry for every stage (testing)
static int moe_coop_wide_mode()
{
    static int mode = -2;
    if (mode == -2)
    {
        const char* env = std::getenv("EXL3_MOE_COOP_WIDE");
        mode = env ? atoi(env) : -1;
    }
    return mode;
}

// Per-stage override from an env var: -1 unset, else 0 / 1 (testing)
static int moe_coop_wide_env(const char* name)
{
    const char* env = std::getenv(name);
    return env ? atoi(env) : -1;
}

static bool moe_coop_pick_wide(int kslices, int slots, int device)
{
    const int mode = moe_coop_wide_mode();
    if (mode >= 0) return mode != 0;
    if (DevCtx::instance().get_cc(device) < CC_BLACKWELL) return true;
    return kslices >= 256 || (kslices >= 128 && slots >= 32);
}

// Dynamic shared memory above the 48 KB default needs an opt-in per kernel (once per process)
static void moe_coop_smem_optin(void* kernel, int smem)
{
    if (smem <= 48 * 1024) return;
    static std::map<void*, int> done;
    auto it = done.find(kernel);
    if (it != done.end() && it->second >= smem) return;
    cuda_check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    done[kernel] = smem;
}

// Kernel launch, optionally with programmatic stream serialization (PDL, CC >= 9.0, EXL3_MOE_PDL builds): the
// grid may start while its predecessor drains; the kernels order every dependent access behind griddepcontrol.wait
static void moe_coop_launch_kernel(void* kernel, int grid, int smem, void** args, cudaStream_t stream, bool pdl)
{
#if EXL3_MOE_PDL
    if (pdl)
    {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(grid);
        cfg.blockDim = dim3(MOE_COOP_THREADS);
        cfg.dynamicSmemBytes = smem;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        cuda_check(cudaLaunchKernelExC(&cfg, kernel, args));
        return;
    }
#endif
    cuda_check(cudaLaunchKernel(kernel, dim3(grid), dim3(MOE_COOP_THREADS), args, smem, stream));
}

void exl3_moe_coop_launch(const MoeCoopParams& p_in, float K_gu, float K_d, int cb, int device, cudaStream_t stream)
{
    MoeCoopParams p = p_in;
    { static int dbg = std::getenv("EXL3_MOE_COOP_DBG") ? atoi(std::getenv("EXL3_MOE_COOP_DBG")) : 0; p.dbg = dbg; }
    const int slots = p.bsz * p.topk;
    if (!slots)
    {
        if (p.bsz && p.H_out)
        {
            exl3_moe_coop_empty_kernel<<<dim3(p.bsz, CEIL_DIVIDE(p.H_out, 128)), 32, 0, stream>>>(p);
            cuda_check(cudaPeekAtLastError());
        }
        return;
    }
    const int nproj = p.gated ? 2 : 1;
    p.a_global = p.bsz > 1;

    // Measured per stage (qmesh/notes/MoEKernelTP2.md): gate/up (A) prefers the narrow tile at bsz > 1 (bsz 6:
    // A 218 -> 181 us), down (B) the wide tile (B 97 -> 103 us, A+B 316 -> 283 us). bsz 1 keeps A narrow, B wide.
    // EXL3_MOE_COOP_WIDE overrides both stages; EXL3_MOE_COOP_WIDE_A / _B override one (testing)
    static const int wide_env_a = moe_coop_wide_env("EXL3_MOE_COOP_WIDE_A");
    static const int wide_env_b = moe_coop_wide_env("EXL3_MOE_COOP_WIDE_B");
    const bool forced = moe_coop_wide_mode() >= 0;
    const bool wide_a = wide_env_a >= 0 ? wide_env_a != 0
        : forced ? moe_coop_pick_wide(p.Hi / 16, slots, device) : false;
    // EXL3_MOE_WIDE_B_BSZ1 (default 0; A/B test with EXL3_MOE_COOP_WIDE_B=1): wide down tile at bsz 1. moe_coop_pick_wide
    // returns false for I / 16 = 72 k-slices on CC >= Blackwell, which put B on the narrow tile at bsz 1 (1120 blocks of
    // 9 k-slices per warp at the DSV4 TP=2 shape) despite the comment above
#ifndef EXL3_MOE_WIDE_B_BSZ1
#define EXL3_MOE_WIDE_B_BSZ1 0
#endif
    const bool wide_b = wide_env_b >= 0 ? wide_env_b != 0
        : forced ? moe_coop_pick_wide(p.I / 16, slots, device)
                 : (p.bsz > 1 || EXL3_MOE_WIDE_B_BSZ1 || moe_coop_pick_wide(p.I / 16, slots, device));
    // Mixed-K (any per-expert rate table set): the *_mixed kernels dispatch each block's tile on its expert's
    // rate, so the uniform K / half flags are not used for the projections that carry a table
    const bool mixed = p.k2_g || p.k2_u || p.k2_d;
    TORCH_CHECK(!mixed || cb == 2, "exl3_moe_coop: mixed per-expert bitrates require the mul1 codebook");
    MoeCoopKernel ka = mixed ? exl3_moe_coop_kernel_a_mixed(p.Hi, wide_a) : moe_coop_kernel_a(K_gu, cb, p.Hi, wide_a);
    MoeCoopKernel kb = mixed ? exl3_moe_coop_kernel_b_mixed(wide_b) : moe_coop_kernel_b(K_d, cb, wide_b);
    moe_coop_smem_optin(ka.kernel, ka.smem);
    moe_coop_smem_optin(kb.kernel, kb.smem);

    const int base_a = slots * nproj * (p.I / (wide_a ? 128 : MOE_COOP_COLS));
    const int base_b = slots * (p.Ho / (wide_b ? 128 : MOE_COOP_COLS));
    const int max_split = std::max(1, p.slots_max / slots);
    p.ksplit_a = moe_coop_pick_ksplit(base_a, p.Hi / 16, max_split, device);
    p.ksplit_b = moe_coop_pick_ksplit(base_b, p.I / 16, max_split, device);
    const int grid_a = base_a * p.ksplit_a;
    const int grid_b = base_b * p.ksplit_b;

    void* args[] = { (void*) &p };
    if (p.a_global)
    {
        const int rot_items = slots * (p.Hi / 128) * nproj;
        const int rot_grid = CEIL_DIVIDE(rot_items, MOE_COOP_THREADS / 32);
        exl3_moe_coop_ns::exl3_moe_coop_rot_kernel<<<rot_grid, MOE_COOP_THREADS, 0, stream>>>(p);
    }
    static const bool single = std::getenv("EXL3_MOE_COOP_SINGLE") && atoi(std::getenv("EXL3_MOE_COOP_SINGLE")) != 0;
    if (single && mixed)
    {
        // One persistent launch for A then B (queue of grid_a + grid_b items). The fused kernel does not zero its
        // counters, so they are cleared here in stream order before the launch: queue head, ready flags, A/B counters
        MoeCoopKernel kf = exl3_moe_coop_kernel_fused_mixed(p.Hi, wide_a, wide_b);
        moe_coop_smem_optin(kf.kernel, kf.smem);
        int per_sm = 0;
        cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kf.kernel, MOE_COOP_THREADS, kf.smem));
        int n_sm = 0;
        cuda_check(cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, device));
        p.fused = 1;
        p.nA = grid_a;
        p.nB = grid_b;
        p.resident = std::max(1, std::min(per_sm * n_sm, grid_a + grid_b));
        cuda_check(cudaMemsetAsync(p.queue_head, 0, sizeof(int), stream));
        cuda_check(cudaMemsetAsync(p.ctr_a, 0, sizeof(int) * (p.ctr_a_len + p.ctr_b_len), stream));
        cuda_check(cudaMemsetAsync(p.ready_a, 0, sizeof(int) * p.ctr_a_len, stream));
        cuda_check(cudaLaunchKernel(kf.kernel, dim3(p.resident), dim3(MOE_COOP_THREADS), args, kf.smem, stream));
        cuda_check(cudaPeekAtLastError());
        return;
    }
    // B may overlap A's drain (PDL): it reads only routing and weights before its griddepcontrol.wait. A is launched
    // normally (its routing reads must follow the router). EXL3_MOE_COOP_PDL=0 disables at run time
    static const bool pdl_env = !std::getenv("EXL3_MOE_COOP_PDL") || atoi(std::getenv("EXL3_MOE_COOP_PDL")) != 0;
    const bool pdl_b = pdl_env && DevCtx::instance().get_cc(device) >= CC_HOPPER;
    moe_coop_launch_kernel(ka.kernel, grid_a, ka.smem, args, stream, false);
    moe_coop_launch_kernel(kb.kernel, grid_b, kb.smem, args, stream, pdl_b);
    cuda_check(cudaPeekAtLastError());
}

MoeCoopParams exl3_moe_coop_prepare
(
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_gate_w,
    float& K_gu, float& K_d, int& cb
)
{
    TORCH_CHECK_DTYPE(ctr, kInt);
    TORCH_CHECK(ctr.is_contiguous(), "exl3_moe_coop: ctr must be contiguous");
    for (const at::Tensor* t : { &g_trellis, &g_suh, &g_svh, &u_trellis, &u_suh, &u_svh, &d_trellis, &d_suh, &d_svh })
        TORCH_CHECK(t->scalar_type() == at::kLong && t->is_contiguous(), "exl3_moe_coop: pointer tables must be contiguous int64");
    TORCH_CHECK(!(mcg && mul1), "exl3_moe_coop: specified both mcg and mul1");
    TORCH_CHECK(Kg >= 1 && Kg <= 8 && Ku >= 1 && Ku <= 8 && Kd >= 1 && Kd <= 8, "exl3_moe_coop: K out of range");
    TORCH_CHECK(!gated || Kg == Ku, "exl3_moe_coop: gate and up must share a bit width");
    TORCH_CHECK(act >= 0 && act <= 3, "exl3_moe_coop: unknown activation");

    MoeCoopParams p = {};
    p.Hi = Hi;
    p.I = (int) gu_u.size(-1);
    p.Ho = (int) d_out.size(-1);
    p.H_out = (int) out.size(-1);
    TORCH_CHECK(p.Hi % 128 == 0 && p.I % 128 == 0 && p.Ho % 128 == 0 && p.H_out <= p.Ho, "exl3_moe_coop: Hi/I/Ho shape");
    TORCH_CHECK(p.Hi > 0 && p.I > 0 && p.Ho > 0, "exl3_moe_coop: empty Hi/I/Ho dimension");

    TORCH_CHECK_DTYPE(had_g, kHalf);
    TORCH_CHECK_DTYPE(had_u, kHalf);
    TORCH_CHECK_DTYPE(act_out, kHalf);
    TORCH_CHECK_DTYPE(d_out, kFloat);
    TORCH_CHECK_DTYPE(out, kFloat);
    p.gu_f32 = gu_u.scalar_type() == at::kFloat;
    if (!p.gu_f32) TORCH_CHECK_DTYPE(gu_u, kHalf);
    TORCH_CHECK(gu_g.scalar_type() == gu_u.scalar_type(), "exl3_moe_coop: gate/up scratch dtype mismatch");
    // Scratch capacity in slots: the smallest of the per-slot buffers
    const int slots_max = (int) std::min({ had_g.numel() / p.Hi, had_u.numel() / p.Hi, gu_g.numel() / p.I,
                                           gu_u.numel() / p.I, act_out.numel() / p.I, d_out.numel() / p.Ho });
    auto check_scratch = [&] (const at::Tensor& t, int width)
    {
        TORCH_CHECK(t.is_contiguous() && t.size(-1) == width, "exl3_moe_coop: scratch shape");
    };
    check_scratch(had_g, p.Hi);
    check_scratch(had_u, p.Hi);
    check_scratch(gu_g, p.I);
    check_scratch(gu_u, p.I);
    check_scratch(act_out, p.I);
    check_scratch(d_out, p.Ho);
    TORCH_CHECK(out.dim() >= 2 && out.stride(-1) == 1, "exl3_moe_coop: out shape");
    TORCH_CHECK(slots_max >= 1 && slots_max <= 256, "exl3_moe_coop: scratch must hold 1..256 slots");

    // Counters: the gate/up stage's block covers every slot the scratch can hold, the down
    // stage's every token row the output can hold
    const int rows_max = (int) out.size(-2);
    p.slots_max = slots_max;
    p.ctr_a_len = slots_max * (p.I / 128);
    p.ctr_b_len = rows_max * (p.Ho / 128);
    TORCH_CHECK(ctr.numel() >= exl3_moe_coop_ctr_len(slots_max, rows_max, p.I, p.Ho), "exl3_moe_coop: counter scratch too small");
    p.ctr_a = (int*) ctr.data_ptr();
    p.ctr_b = p.ctr_a + p.ctr_a_len;
    p.runs = p.ctr_b + p.ctr_b_len;
    p.ready_a = p.runs + (2 + (slots_max + 1) + slots_max);
    p.queue_head = p.ready_a + p.ctr_a_len;
    p.exit_ctr = p.queue_head + 1;
    p.rows_max = rows_max;

    p.g_trellis = (const int64_t*) g_trellis.data_ptr();
    p.g_suh = (const int64_t*) g_suh.data_ptr();
    p.g_svh = (const int64_t*) g_svh.data_ptr();
    p.u_trellis = (const int64_t*) u_trellis.data_ptr();
    p.u_suh = (const int64_t*) u_suh.data_ptr();
    p.u_svh = (const int64_t*) u_svh.data_ptr();
    p.d_trellis = (const int64_t*) d_trellis.data_ptr();
    p.d_suh = (const int64_t*) d_suh.data_ptr();
    p.d_svh = (const int64_t*) d_svh.data_ptr();
    p.g_bias = g_bias ? (const int64_t*) g_bias->data_ptr() : nullptr;
    p.u_bias = u_bias ? (const int64_t*) u_bias->data_ptr() : nullptr;
    p.d_bias = d_bias ? (const int64_t*) d_bias->data_ptr() : nullptr;
    p.n_local = (int) u_trellis.size(0);
    p.act = act;
    p.act_limit = act_limit;
    p.gated = gated;

    p.had_g = (half*) had_g.data_ptr();
    p.had_u = (half*) had_u.data_ptr();
    p.gu_g = gu_g.data_ptr();
    p.gu_u = gu_u.data_ptr();
    p.act_out = (half*) act_out.data_ptr();
    p.d_out = (float*) d_out.data_ptr();
    p.out = (float*) out.data_ptr();
    p.out_stride = (int) out.stride(-2);

    if (sh_gate_w)
    {
        TORCH_CHECK_DTYPE(sh_gate_w.value(), kHalf);
        TORCH_CHECK(sh_gate_w->is_contiguous(), "exl3_moe_coop: shared gate weight must be contiguous");
        p.sh_gate_w = (const half*) sh_gate_w->data_ptr();
        p.sh_gate_n = (int) sh_gate_w->numel();
    }

    K_gu = gated ? Kg : Ku;
    K_d = Kd;
    cb = mcg ? 1 : (mul1 ? 2 : 0);
    return p;
}

void exl3_moe_coop_run
(
    MoeCoopParams p, float K_gu, float K_d, int cb,
    const at::Tensor& x, const at::Tensor& sel, const at::Tensor& rw,
    const c10::optional<at::Tensor>& sh_out
)
{
    const at::cuda::OptionalCUDAGuard device_guard(x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    const int device = x.device().index();

    // Per-call checks kept to what can actually vary between calls
    TORCH_CHECK(x.dim() == 2 && x.stride(1) == 1, "exl3_moe_coop: x must be (bsz, H) with unit column stride");
    p.x = (const half*) x.data_ptr();
    p.x_stride = (int) x.stride(0);
    p.bsz = (int) x.size(0);
    p.H = (int) x.size(1);
    p.topk = (int) sel.size(-1);
    p.sel = (const int64_t*) sel.data_ptr();
    p.rw = (const half*) rw.data_ptr();
    const int slots = p.bsz * p.topk;
    TORCH_CHECK(p.H % 4 == 0 && p.H <= p.Hi, "exl3_moe_coop: H/Hi shape");
    TORCH_CHECK(slots <= p.slots_max && p.bsz <= p.rows_max, "exl3_moe_coop: batch exceeds the scratch");
    TORCH_CHECK(p.min_expert < 0 || p.max_expert - p.min_expert <= p.n_local, "exl3_moe_coop: expert range exceeds tables");
    if (sh_out)
    {
        TORCH_CHECK(p.H_out == p.H, "exl3_moe_coop: shared expert output width must match the routed output");
        p.sh_out = (const float*) sh_out->data_ptr();
        if (p.sh_gate_w) TORCH_CHECK(p.sh_gate_n == p.H && p.H % 2 == 0, "exl3_moe_coop: shared gate width");
    }
    else
        p.sh_gate_w = nullptr;
    exl3_moe_coop_launch(p, K_gu, K_d, cb, device, stream);
}

void exl3_moe_coop
(
    const at::Tensor& x,
    const at::Tensor& sel,
    const at::Tensor& rw,
    int min_expert,
    int max_expert,
    int Hi,
    const at::Tensor& g_trellis, const at::Tensor& g_suh, const at::Tensor& g_svh,
    const at::Tensor& u_trellis, const at::Tensor& u_suh, const at::Tensor& u_svh,
    const at::Tensor& d_trellis, const at::Tensor& d_suh, const at::Tensor& d_svh,
    const c10::optional<at::Tensor>& g_bias,
    const c10::optional<at::Tensor>& u_bias,
    const c10::optional<at::Tensor>& d_bias,
    float Kg, float Ku, float Kd,
    bool mcg, bool mul1,
    int act,
    float act_limit,
    bool gated,
    at::Tensor& had_g,
    at::Tensor& had_u,
    at::Tensor& gu_g,
    at::Tensor& gu_u,
    at::Tensor& act_out,
    at::Tensor& d_out,
    at::Tensor& ctr,
    at::Tensor& out,
    const c10::optional<at::Tensor>& sh_out,
    const c10::optional<at::Tensor>& sh_gate_w,
    const c10::optional<at::Tensor>& k2_g,     // per local expert half-bit rates (uint8, nullable): mixed-K layer
    const c10::optional<at::Tensor>& k2_u,
    const c10::optional<at::Tensor>& k2_d
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(sel, kLong);
    TORCH_CHECK_DTYPE(rw, kHalf);
    TORCH_CHECK(sel.is_contiguous() && rw.is_contiguous(), "exl3_moe_coop: sel/rw must be contiguous");
    TORCH_CHECK(sel.sizes() == rw.sizes() && sel.dim() == 2 && sel.size(0) == x.size(0), "exl3_moe_coop: sel/rw must be (bsz, topk)");
    if (sh_out)
    {
        TORCH_CHECK_DTYPE(sh_out.value(), kFloat);
        TORCH_CHECK(sh_out->is_contiguous() && sh_out->size(-1) == x.size(1) && sh_out->numel() >= x.size(0) * x.size(1),
                    "exl3_moe_coop: shared expert output shape");
    }
    float K_gu, K_d; int cb;
    // Mixed layers: per-expert rates come from the tables (k2_*), and the scalar K is ignored by the kernels. Pass the
    // gate rate for up too so the uniform gate/up guard in prepare holds; the per-expert pairing is checked below.
    const bool mixed = k2_g.has_value() || k2_u.has_value() || k2_d.has_value();
    const float Ku_eff = mixed ? Kg : Ku;
    MoeCoopParams p = exl3_moe_coop_prepare(Hi, g_trellis, g_suh, g_svh, u_trellis, u_suh, u_svh, d_trellis, d_suh, d_svh,
                                            g_bias, u_bias, d_bias, Kg, Ku_eff, Kd, mcg, mul1, act, act_limit, gated,
                                            had_g, had_u, gu_g, gu_u, act_out, d_out, ctr, out, sh_gate_w, K_gu, K_d, cb);
    p.min_expert = min_expert;
    p.max_expert = max_expert;
    if (mixed)
    {
        TORCH_CHECK(k2_g && k2_u && k2_d, "exl3_moe_coop: mixed-K needs all three rate tables");
        for (const at::Tensor* t : { &k2_g.value(), &k2_u.value(), &k2_d.value() })
        {
            TORCH_CHECK(t->scalar_type() == at::kByte && t->is_contiguous() && t->is_cuda(),
                        "exl3_moe_coop: rate tables must be contiguous uint8 CUDA tensors");
        }
        // gate and up of one expert must share a width. Checked by the table builder (host, once at load), NOT here:
        // a device comparison in the launch path would synchronize and break CUDA-graph capture.
        TORCH_CHECK(k2_g->numel() >= 1 && k2_u->numel() == k2_g->numel(), "exl3_moe_coop: gate/up rate table length");
        p.k2_g = k2_g->data_ptr<uint8_t>();
        p.k2_u = k2_u->data_ptr<uint8_t>();
        p.k2_d = k2_d->data_ptr<uint8_t>();
    }
    exl3_moe_coop_run(p, K_gu, K_d, cb, x, sel, rw, sh_out);
}
