#include <cstring>
#include <cstdio>
#include <string>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include "p2p_oneshot.cuh"
#include "../ptx.cuh"

#define P2P_TIMEOUT_NS (10ull * 1000000000ull)

// Every collective is symmetric: each rank pushes its payload into the peer and waits for the peer's push, so
// a rank can never run more than one collective ahead of its peer (see header). Payload stripes are per block:
// block b pushes uint4 units [b*U/nblk, (b+1)*U/nblk) and waits only for the peer's block b, which has the same
// partition on both ranks.
//
// mode 0: all-reduce fp32 in place: x <- x_rank0 + x_rank1 (fixed operand order on both ranks)
// mode 1: all-gather: out <- [x_rank0 | x_rank1]
// mode 2: broadcast: out <- x_root
template <int MODE>
__global__ void __launch_bounds__(256) p2p_oneshot_kernel
(
    uint8_t* __restrict__ src,
    uint8_t* __restrict__ out,
    int64_t units,           // uint4 units of this rank's payload
    int rank,
    int root,
    uint8_t* __restrict__ local_arena,
    uint8_t* __restrict__ peer_arena
)
{
    __shared__ uint64_t s_seq;
    const int b = blockIdx.x;
    uint64_t* local_state = (uint64_t*) (local_arena + P2P_OFF_STATE) + b;
    if (threadIdx.x == 0) s_seq = *local_state + 1;
    __syncthreads();
    const uint64_t seq = s_seq;
    const int p = (int) (seq & 1);

    const int64_t u0 = (units * b) / gridDim.x;
    const int64_t u1 = (units * (b + 1)) / gridDim.x;

    uint4* my_src = (uint4*) src;
    uint4* peer_slot = (uint4*) (peer_arena + p * P2P_MAXB);
    uint4* local_slot = (uint4*) (local_arena + p * P2P_MAXB);
    uint64_t* peer_flag = (uint64_t*) (peer_arena + P2P_OFF_FLAGS) + p * P2P_MAXBLK + b;
    uint64_t* local_flag = (uint64_t*) (local_arena + P2P_OFF_FLAGS) + p * P2P_MAXBLK + b;

    // 1. Push my stripe into the peer's slot[p], then publish it with a system-scope release flag.
    for (int64_t i = u0 + threadIdx.x; i < u1; i += blockDim.x) peer_slot[i] = my_src[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) stg_release_sys_u64(peer_flag, seq);

    // 2. Wait until the peer's stripe for this block has arrived in my slot[p].
    if (threadIdx.x == 0)
    {
        uint64_t deadline = globaltimer_ns() + P2P_TIMEOUT_NS;
        while (ldg_acquire_sys_u64(local_flag) < seq)
        {
            if (globaltimer_ns() > deadline)
            {
                printf("p2p_oneshot: timeout rank %d block %d seq %llu mode %d\n", rank, b,
                       (unsigned long long) seq, MODE);
                __trap();
            }
        }
    }
    __syncthreads();

    // 3. Combine. Peer data is read through L1-bypassing loads: it was written by the peer over P2P.
    uint4* peer_data = local_slot;
    if (MODE == 0)
    {
        float4* x4 = (float4*) src;
        const float4* pd = (const float4*) peer_data;
        for (int64_t i = u0 + threadIdx.x; i < u1; i += blockDim.x)
        {
            float4 mine = __ldcg((const float4*) (my_src + i));
            float4 other = __ldcg(pd + i);
            float4 r0 = rank == 0 ? mine : other;
            float4 r1 = rank == 0 ? other : mine;
            x4[i] = make_float4(r0.x + r1.x, r0.y + r1.y, r0.z + r1.z, r0.w + r1.w);
        }
    }
    else if (MODE == 1)
    {
        // out is [rank0 | rank1], each half has `units` uint4 units
        uint4* mine_out = (uint4*) (out + (int64_t) rank * units * 16);
        uint4* other_out = (uint4*) (out + (int64_t) (1 - rank) * units * 16);
        for (int64_t i = u0 + threadIdx.x; i < u1; i += blockDim.x)
        {
            mine_out[i] = __ldcg(my_src + i);
            other_out[i] = __ldcg(peer_data + i);
        }
    }
    else
    {
        uint4* o = (uint4*) out;
        const uint4* from = rank == root ? (const uint4*) my_src : peer_data;
        for (int64_t i = u0 + threadIdx.x; i < u1; i += blockDim.x) o[i] = __ldcg(from + i);
    }

    // 4. Advance the device-side sequence number (graph replays see the new value).
    __syncthreads();
    if (threadIdx.x == 0) *local_state = seq;
}

static void check_common(torch::Tensor x, int64_t local, int64_t peer, int64_t nblk)
{
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), "p2p: input must be a contiguous CUDA tensor");
    TORCH_CHECK(local != 0 && peer != 0, "p2p: arena not set up");
    TORCH_CHECK(nblk >= 1 && nblk <= P2P_MAXBLK, "p2p: nblk out of range");
}

// Pinned shared-host arena (fallback transport, used when peer SM access is unavailable): a /dev/shm file mapped
// into both processes, pinned and device-mapped for each GPU. Returns the device-visible pointer for this GPU.
// `name` is the shm object, `create` sets it up.
int64_t p2p_shm_arena(const std::string& name, bool create)
{
    int fd = shm_open(name.c_str(), create ? (O_CREAT | O_RDWR) : O_RDWR, 0600);
    TORCH_CHECK(fd >= 0, "p2p: shm_open failed for ", name);
    if (create) TORCH_CHECK(ftruncate(fd, P2P_ARENA_BYTES) == 0, "p2p: ftruncate failed");
    void* host = mmap(nullptr, P2P_ARENA_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    TORCH_CHECK(host != MAP_FAILED, "p2p: mmap failed");
    if (create) memset(host, 0, P2P_ARENA_BYTES);
    TORCH_CHECK(cudaHostRegister(host, P2P_ARENA_BYTES, cudaHostRegisterMapped | cudaHostRegisterPortable) == cudaSuccess,
                "p2p: cudaHostRegister failed");
    void* dev = nullptr;
    TORCH_CHECK(cudaHostGetDevicePointer(&dev, host, 0) == cudaSuccess, "p2p: cudaHostGetDevicePointer failed");
    return (int64_t) dev;
}

static int p2p_hexval(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    TORCH_CHECK(false, "p2p: bad hex digit in IPC handle");
    return 0;
}

// cudaIpc transport: device arena on `device`, zeroed, exported as a hex-encoded IPC handle.
int64_t p2p_dev_arena_alloc(int64_t device)
{
    c10::cuda::CUDAGuard guard((c10::DeviceIndex) device);
    void* p = nullptr;
    TORCH_CHECK(cudaMalloc(&p, P2P_ARENA_BYTES) == cudaSuccess, "p2p: cudaMalloc arena failed");
    TORCH_CHECK(cudaMemset(p, 0, P2P_ARENA_BYTES) == cudaSuccess && cudaDeviceSynchronize() == cudaSuccess,
                "p2p: arena zeroing failed");
    return (int64_t) p;
}

std::string p2p_dev_arena_handle(int64_t ptr)
{
    cudaIpcMemHandle_t h;
    TORCH_CHECK(cudaIpcGetMemHandle(&h, (void*) ptr) == cudaSuccess, "p2p: cudaIpcGetMemHandle failed");
    static const char* digits = "0123456789abcdef";
    std::string s(2 * sizeof(h.reserved), '0');
    for (size_t i = 0; i < sizeof(h.reserved); i++)
    {
        unsigned char c = (unsigned char) h.reserved[i];
        s[2 * i] = digits[c >> 4];
        s[2 * i + 1] = digits[c & 15];
    }
    return s;
}

// Opens a peer's arena on `local_device` (peer access enabled). Returns a device pointer valid on local_device.
int64_t p2p_dev_arena_open(const std::string& hex, int64_t local_device, int64_t peer_device)
{
    cudaIpcMemHandle_t h;
    TORCH_CHECK(hex.size() == 2 * sizeof(h.reserved), "p2p: IPC handle has wrong length");
    for (size_t i = 0; i < sizeof(h.reserved); i++)
        h.reserved[i] = (char) (p2p_hexval(hex[2 * i]) * 16 + p2p_hexval(hex[2 * i + 1]));
    c10::cuda::CUDAGuard guard((c10::DeviceIndex) local_device);
    cudaError_t e = cudaDeviceEnablePeerAccess((int) peer_device, 0);
    if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "p2p: cudaDeviceEnablePeerAccess failed: ", cudaGetErrorString(e));
    }
    cudaGetLastError();
    void* p = nullptr;
    e = cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess);
    if (e != cudaSuccess)
    {
        cudaGetLastError();
        TORCH_CHECK(false, "p2p: cudaIpcOpenMemHandle failed: ", cudaGetErrorString(e));
    }
    return (int64_t) p;
}

void p2p_all_reduce_f32(torch::Tensor x, int64_t rank, int64_t local, int64_t peer, int64_t nblk)
{
    check_common(x, local, peer, nblk);
    TORCH_CHECK(x.dtype() == torch::kFloat, "p2p_all_reduce_f32: fp32 only");
    TORCH_CHECK(x.numel() % 4 == 0 && x.numel() * 4 <= P2P_MAXB, "p2p_all_reduce_f32: size");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t units = x.numel() / 4;
    p2p_oneshot_kernel<0><<<(unsigned) nblk, 256, 0, stream>>>((uint8_t*) x.data_ptr(), (uint8_t*) x.data_ptr(),
        units, (int) rank, 0, (uint8_t*) local, (uint8_t*) peer);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "p2p_all_reduce_f32 launch failed");
}

void p2p_all_gather_f32(torch::Tensor x, torch::Tensor out, int64_t rank, int64_t local, int64_t peer, int64_t nblk)
{
    check_common(x, local, peer, nblk);
    TORCH_CHECK(x.dtype() == torch::kFloat && out.dtype() == torch::kFloat, "p2p_all_gather_f32: fp32 only");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.device() == x.device(), "p2p_all_gather_f32: out");
    TORCH_CHECK(out.numel() == 2 * x.numel(), "p2p_all_gather_f32: out must hold two payloads");
    TORCH_CHECK(x.numel() % 4 == 0 && x.numel() * 4 <= P2P_MAXB, "p2p_all_gather_f32: size");
    TORCH_CHECK(out.data_ptr() != x.data_ptr(), "p2p_all_gather_f32: out must not alias x");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t units = x.numel() / 4;
    p2p_oneshot_kernel<1><<<(unsigned) nblk, 256, 0, stream>>>((uint8_t*) x.data_ptr(), (uint8_t*) out.data_ptr(),
        units, (int) rank, 0, (uint8_t*) local, (uint8_t*) peer);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "p2p_all_gather_f32 launch failed");
}

void p2p_bcast_i64(torch::Tensor x, torch::Tensor out, int64_t rank, int64_t root, int64_t local, int64_t peer,
                   int64_t nblk)
{
    check_common(x, local, peer, nblk);
    TORCH_CHECK(x.dtype() == torch::kInt64 && out.dtype() == torch::kInt64, "p2p_bcast_i64: int64 only");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.device() == x.device(), "p2p_bcast_i64: out");
    TORCH_CHECK(out.numel() == x.numel(), "p2p_bcast_i64: size mismatch");
    TORCH_CHECK(x.numel() % 2 == 0 && x.numel() * 8 <= P2P_MAXB, "p2p_bcast_i64: size");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int64_t units = x.numel() / 2;
    p2p_oneshot_kernel<2><<<(unsigned) nblk, 256, 0, stream>>>((uint8_t*) x.data_ptr(), (uint8_t*) out.data_ptr(),
        units, (int) rank, (int) root, (uint8_t*) local, (uint8_t*) peer);
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "p2p_bcast_i64 launch failed");
}
