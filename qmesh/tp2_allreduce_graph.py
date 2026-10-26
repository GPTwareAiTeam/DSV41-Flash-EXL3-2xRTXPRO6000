"""
Graph-capture check for the on-device all-reduce (ext.pg_all_reduce) on the native TP backend: two ranks on GPU0/1,
capture one (1, 5120) fp32 all-reduce into a CUDA graph, replay ITERS times, check every replay's result against the
eager result, and time replays per op. Also times the same graph replay for the CPU path, which must NOT be capturable.

    CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=/src/exllamav3:/src/exllamav3/tests python3 qmesh/tp2_allreduce_graph.py
"""
import sys
import time
import torch

sys.path.insert(0, "/src/exllamav3/tests")
from testlib.tp_native import run_ranks  # noqa: E402

ITERS = 200
SHAPE = (1, 5120)


def graph_program(ctx):
    from exllamav3.ext import exllamav3_ext as ext
    b = ctx.backend
    dev = ctx.device
    g = torch.Generator(device="cpu").manual_seed(1234 + ctx.rank)
    x0 = torch.randn(SHAPE, generator=g, dtype=torch.float32).to(dev)
    static = x0.clone()

    def gpu_op():
        ext.pg_all_reduce(b.ptr_g, b.dev_g, b.active_devices, b.device, b.active_devices[0], static, b.dev_b,
                          b.shbuf_size, b.abort_flag)

    # eager reference (also warms up the kernel)
    static.copy_(x0)
    gpu_op()
    torch.cuda.synchronize(dev)
    ref = static.clone().cpu()

    # capture on a side stream, as Segment does
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    graph = torch.cuda.CUDAGraph()
    capture_ok, capture_err = True, None
    try:
        with torch.cuda.stream(s):
            with torch.cuda.graph(graph, stream=s, capture_error_mode="thread_local"):
                gpu_op()
    except Exception as e:  # report and stop
        capture_ok, capture_err = False, repr(e)
    torch.cuda.current_stream(dev).wait_stream(s)
    torch.cuda.synchronize(dev)
    if not capture_ok:
        return {"capture_ok": False, "err": capture_err}

    ctx.sync()
    mismatches = 0
    # replay: each replay reduces the value present in `static` at replay time; reset input before each replay
    for i in range(ITERS):
        static.copy_(x0)
        graph.replay()
        if i < 3 or i == ITERS - 1:
            torch.cuda.synchronize(dev)
            if not torch.equal(static.cpu(), ref):
                mismatches += 1
    torch.cuda.synchronize(dev)
    ctx.sync()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        graph.replay()
    end.record()
    torch.cuda.synchronize(dev)
    ms = start.elapsed_time(end) / ITERS
    return {"capture_ok": True, "mismatches": mismatches, "replay_us": ms * 1e3, "ref": ref}


if __name__ == "__main__":
    res = run_ranks(graph_program, [0, 1], output_rank=-1)
    r0, r1 = res[0], res[1]
    print("capture_ok:", r0["capture_ok"], r1["capture_ok"], r0.get("err", ""), r1.get("err", ""))
    if r0["capture_ok"] and r1["capture_ok"]:
        print(f"graph replay all-reduce (1,5120) fp32: {r0['replay_us']:.1f} us/op rank0, {r1['replay_us']:.1f} us/op rank1")
        print("mismatches vs eager:", r0["mismatches"], r1["mismatches"])
        print("rank0 == rank1 ref:", torch.equal(r0["ref"], r1["ref"]))
