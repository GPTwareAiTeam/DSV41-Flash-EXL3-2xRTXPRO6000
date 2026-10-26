// Per-expert mixed-K decode kernels (mul1 only: half-bit rates exist only under mul1). Launchers for the
// generated *_mixed kernels in ../exl3_moe_coop_mixed_kernel.cuh; the per-expert rate is read at run time.
#include "exl3_moe_coop_instances.cuh"
#include "../exl3_moe_coop_mixed_kernel.cuh"

MoeCoopKernel exl3_moe_coop_kernel_a_mixed(int Hi, bool wide)
{
    using namespace exl3_moe_coop_mixed_ns;
    const int smem = mixed_smem_a(Hi);
    if (wide) return { (void*) exl3_moe_coop_a_mixed_kernel<2, true>, smem };
    return { (void*) exl3_moe_coop_a_mixed_kernel<2, false>, smem };
}

MoeCoopKernel exl3_moe_coop_kernel_b_mixed(bool wide)
{
    using namespace exl3_moe_coop_mixed_ns;
    const int smem = mixed_smem_b();
    if (wide) return { (void*) exl3_moe_coop_b_mixed_kernel<2, true>, smem };
    return { (void*) exl3_moe_coop_b_mixed_kernel<2, false>, smem };
}
