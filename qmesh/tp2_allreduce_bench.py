"""
Latency of one all-reduce of a (1, 5120) fp32 tensor across two TP ranks on the real native backend
(TPBackendNative): the CPU-reduce path (ext.pg_all_reduce_cpu, via backend.all_reduce) and the on-device GPU ring
(ext.pg_all_reduce, the P2P ring over the shared B buffer). Run inside exl3-fork0 with CUDA_VISIBLE_DEVICES=0,1:

    PYTHONPATH=/src/exllamav3 python3 qmesh/tp2_allreduce_bench.py

Both paths are timed on the same inputs and their sums are compared bitwise (fp32 ring vs fp32 CPU wire may differ
by bf16 wire rounding, so the max abs difference is reported too).
"""
import sys
import time
import torch

sys.path.insert(0, "/src/exllamav3/tests")
from testlib.tp_native import run_ranks  # noqa: E402

ITERS = 200
WARM = 20
SHAPE = (1, 5120)


def bench_program(ctx):
    from exllamav3.ext import exllamav3_ext as ext
    b = ctx.backend
    dev = ctx.device
    g = torch.Generator(device="cpu").manual_seed(1234 + ctx.rank)
    x0 = torch.randn(SHAPE, generator=g, dtype=torch.float32).to(dev)
    out = {}

    # CPU-reduce path (what the native backend uses today)
    def cpu_once():
        t = x0.clone()
        b.all_reduce(t, True)
        return t

    for _ in range(WARM):
        cpu_once()
    torch.cuda.synchronize(dev)
    ctx.sync()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        cpu_once()
        ctx.end_round()
    torch.cuda.synchronize(dev)
    out["cpu_ms"] = (time.perf_counter() - t0) / ITERS * 1e3
    cpu_res = cpu_once()
    ctx.end_round()

    # On-device GPU ring
    def gpu_once():
        t = x0.clone()
        ext.pg_all_reduce(b.ptr_g, b.dev_g, b.active_devices, b.device, b.active_devices[0], t, b.dev_b,
                          b.shbuf_size, b.abort_flag)
        return t

    for _ in range(WARM):
        gpu_once()
    torch.cuda.synchronize(dev)
    ctx.sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        gpu_once()
    end.record()
    torch.cuda.synchronize(dev)
    out["gpu_ring_ms"] = start.elapsed_time(end) / ITERS
    gpu_res = gpu_once()

    out["cpu_res"] = cpu_res.cpu()
    out["gpu_res"] = gpu_res.cpu()
    return out


if __name__ == "__main__":
    res = run_ranks(bench_program, [0, 1], output_rank=-1)
    r0, r1 = res[0], res[1]
    print(f"CPU-reduce path   (pg_all_reduce_cpu): {r0['cpu_ms']*1e3:.1f} us/op (rank0), {r1['cpu_ms']*1e3:.1f} us/op (rank1)")
    print(f"GPU ring path     (pg_all_reduce):     {r0['gpu_ring_ms']*1e3:.1f} us/op (rank0), {r1['gpu_ring_ms']*1e3:.1f} us/op (rank1)")
    d = (r0["cpu_res"] - r0["gpu_res"]).abs().max().item()
    print(f"max |cpu_path - gpu_ring| on rank0: {d:.3e}")
    print(f"rank0 == rank1 (gpu ring): {torch.equal(r0['gpu_res'], r1['gpu_res'])}")
    print(f"rank0 == rank1 (cpu path): {torch.equal(r0['cpu_res'], r1['cpu_res'])}")
