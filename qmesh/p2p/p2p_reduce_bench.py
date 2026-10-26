"""Standalone 2-process one-shot P2P all-reduce microbench (fp32, small tensors), cudaIpc peer mapping.

Each rank owns an arena on its GPU: [2 parity slots x MAXB bytes data][2 parity flags (uint64)].
Reduce(x): write x into the PEER's data slot[p] (remote P2P store), fence.sys, store seq into the peer's
flag[p]; spin until our own flag[p] == seq (the peer wrote its part into our arena); out = part_r0 + part_r1
(fixed order, bit-identical on both ranks). seq and parity live in device memory and are advanced by the
kernel itself, so a captured graph of N reduces replays correctly.

Usage (container with both GPUs): python3 p2p_reduce_bench.py
"""
import os, sys, time, tempfile
import torch
import torch.multiprocessing as mp
from torch.utils.cpp_extension import load_inline

MAXB = 64 * 1024

CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define MAXB (64 * 1024)

__device__ __forceinline__ void st_release_sys(uint64_t* p, uint64_t v) {
    asm volatile("st.release.sys.global.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t ld_acquire_sys(const uint64_t* p) {
    uint64_t v; asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory"); return v;
}

// One block; n floats (n % 4 == 0, n * 4 <= MAXB). state[0] = seq counter (device, per rank)
__global__ void __launch_bounds__(1024) p2p_reduce_kernel(
    float* __restrict__ x, int n, int rank,
    uint8_t* __restrict__ local_arena, uint8_t* __restrict__ peer_arena, uint64_t* __restrict__ state)
{
    __shared__ uint64_t s_seq;
    if (threadIdx.x == 0) s_seq = state[0] + 1;
    __syncthreads();
    const uint64_t seq = s_seq;
    const int p = seq & 1;
    float4* peer_data = (float4*) (peer_arena + p * MAXB);
    const float4* local_data = (const float4*) (local_arena + p * MAXB);
    uint64_t* peer_flag = (uint64_t*) (peer_arena + 2 * MAXB) + p;
    const uint64_t* local_flag = (const uint64_t*) (local_arena + 2 * MAXB) + p;
    float4* x4 = (float4*) x;
    const int n4 = n / 4;

    // 1. Remote write of my partial into the peer's slot
    for (int i = threadIdx.x; i < n4; i += blockDim.x) peer_data[i] = x4[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) st_release_sys(peer_flag, seq);

    // 2. Wait for the peer's partial in my slot
    if (threadIdx.x == 0) {
        long long spins = 0;
        while (ld_acquire_sys(local_flag) < seq) { if (++spins > (1ll << 34)) __trap(); }
    }
    __syncthreads();

    // 3. Fixed-order sum: rank0 part + rank1 part
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        float4 mine = x4[i];
        float4 other = local_data[i];
        float4 a = rank == 0 ? mine : other;
        float4 b = rank == 0 ? other : mine;
        x4[i] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
    if (threadIdx.x == 0) state[0] = seq;
}

void p2p_reduce(torch::Tensor x, int64_t rank, int64_t local_arena, int64_t peer_arena, torch::Tensor state) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat && x.is_contiguous());
    int n = x.numel();
    TORCH_CHECK(n % 4 == 0 && n * 4 <= MAXB);
    auto stream = at::cuda::getCurrentCUDAStream();
    p2p_reduce_kernel<<<1, 1024, 0, stream>>>(x.data_ptr<float>(), n, (int) rank,
        (uint8_t*) local_arena, (uint8_t*) peer_arena, (uint64_t*) state.data_ptr<int64_t>());
}

int64_t arena_alloc() {
    void* p; TORCH_CHECK(cudaMalloc(&p, 2 * MAXB + 64) == cudaSuccess);
    TORCH_CHECK(cudaMemset(p, 0, 2 * MAXB + 64) == cudaSuccess);
    return (int64_t) p;
}
torch::Tensor ipc_handle(int64_t ptr) {
    cudaIpcMemHandle_t h; TORCH_CHECK(cudaIpcGetMemHandle(&h, (void*) ptr) == cudaSuccess, "ipc get");
    auto t = torch::empty({(long) sizeof(h)}, torch::kUInt8);
    memcpy(t.data_ptr(), &h, sizeof(h)); return t;
}
int64_t ipc_open(torch::Tensor handle, int64_t peer_dev) {
    cudaIpcMemHandle_t h; memcpy(&h, handle.data_ptr(), sizeof(h));
    int cur; cudaGetDevice(&cur);
    cudaError_t e = cudaDeviceEnablePeerAccess((int) peer_dev, 0);
    TORCH_CHECK(e == cudaSuccess || e == cudaErrorPeerAccessAlreadyEnabled, "peer access ", cudaGetErrorString(e));
    cudaGetLastError();
    void* p; e = cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess);
    TORCH_CHECK(e == cudaSuccess, "ipc open ", cudaGetErrorString(e));
    return (int64_t) p;
}
"""
CPP_SRC = """
void p2p_reduce(torch::Tensor x, int64_t rank, int64_t local_arena, int64_t peer_arena, torch::Tensor state);
int64_t arena_alloc();
torch::Tensor ipc_handle(int64_t ptr);
int64_t ipc_open(torch::Tensor handle, int64_t peer_dev);
"""


def build():
    return load_inline("p2p_reduce_bench", CPP_SRC, cuda_sources = CUDA_SRC,
                       functions = ["p2p_reduce", "arena_alloc", "ipc_handle", "ipc_open"],
                       extra_cuda_cflags = ["-O3", "-arch=sm_120"], verbose = False)


def worker(rank, tmpdir, barrier, results):
    torch.cuda.set_device(rank)
    ext = build()
    local = ext.arena_alloc()
    torch.save(ext.ipc_handle(local), os.path.join(tmpdir, f"h{rank}.pt"))
    barrier.wait()
    peer = ext.ipc_open(torch.load(os.path.join(tmpdir, f"h{1 - rank}.pt")), 1 - rank)
    state = torch.zeros(1, dtype = torch.long, device = rank)
    barrier.wait()

    # Correctness: x_r = deterministic per rank; expected = x0 + x1 (fixed order)
    n = 5120
    g = torch.Generator().manual_seed(1234)
    x0 = torch.randn(n, generator = g); x1 = torch.randn(n, generator = g)
    ok = True
    for it in range(50):
        x = (x0 if rank == 0 else x1).mul(1 + it * 0.01).to(rank)
        ext.p2p_reduce(x, rank, local, peer, state)
        ref = (x0.mul(1 + it * 0.01) + x1.mul(1 + it * 0.01))
        if not torch.equal(x.cpu(), ref):
            ok = False
    torch.cuda.synchronize()

    # Eager latency
    x = torch.randn(n, device = rank)
    for _ in range(200): ext.p2p_reduce(x, rank, local, peer, state)
    torch.cuda.synchronize(); barrier.wait()
    iters = 2000
    t0 = time.perf_counter()
    for _ in range(iters): ext.p2p_reduce(x, rank, local, peer, state)
    torch.cuda.synchronize()
    eager_us = (time.perf_counter() - t0) / iters * 1e6

    # Graph of 80 reduces (one decode token's worth), with a tiny compute op between each
    xs = torch.randn(n, device = rank)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            for _ in range(80):
                xs.mul_(0.5); ext.p2p_reduce(xs, rank, local, peer, state)
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize(); barrier.wait()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        for _ in range(80):
            xs.mul_(0.5); ext.p2p_reduce(xs, rank, local, peer, state)
    torch.cuda.synchronize(); barrier.wait()
    for _ in range(10): gr.replay()
    torch.cuda.synchronize(); barrier.wait()
    reps = 200
    t0 = time.perf_counter()
    for _ in range(reps): gr.replay()
    torch.cuda.synchronize()
    graph_us = (time.perf_counter() - t0) / reps / 80 * 1e6

    # Graph-replay correctness: replay 1 step from known inputs on both ranks
    xs.copy_((x0 if rank == 0 else x1).to(rank))
    barrier.wait()
    gr.replay(); torch.cuda.synchronize()
    ref = (x0 if rank == 0 else x1).clone()
    other = (x1 if rank == 0 else x0).clone()
    a, b = ref, other
    for _ in range(80):
        a = a * 0.5; b = b * 0.5
        s0, s1 = (a, b) if rank == 0 else (b, a)
        a = s0 + s1; b = a.clone()
    graph_ok = torch.equal(xs.cpu(), a)
    results[rank] = (ok, eager_us, graph_us, graph_ok)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    build()  # compile once in the parent
    tmpdir = tempfile.mkdtemp()
    barrier = mp.Barrier(2)
    mgr = mp.Manager(); results = mgr.dict()
    ps = [mp.Process(target = worker, args = (r, tmpdir, barrier, results)) for r in range(2)]
    for p in ps: p.start()
    for p in ps: p.join(timeout = 600)
    for r in range(2):
        ok, eu, gu, gok = results[r]
        print(f"rank {r}: eager_correct={ok} graph_correct={gok} eager_us/reduce={eu:.2f} graph_us/reduce(incl mul)={gu:.2f}")
