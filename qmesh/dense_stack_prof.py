"""DenseFuse: find the caller of the layer-20 index_elementwise / vectorized long kernels. Profiles one decode step
with with_stack=True, maps each CUDA kernel to its launching CPU op (by correlation id), and prints the first dsv4.py
frame of the launching stack for the index / long-cast kernels. Writes /tmp/densefuse_stack.txt."""
import sys
sys.path.insert(0, "/src/exllamav3/qmesh"); sys.path.insert(0, "/src/exllamav3")
import torch
from torch.profiler import profile, ProfilerActivity
from dense_prof import load, job

model, tok, gen = load()
for _ in range(2):
    gen.enqueue(job(tok, 8))
    while gen.num_remaining_jobs(): gen.iterate()
torch.cuda.synchronize()
gen.enqueue(job(tok, 6)); gen.iterate(); torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True) as prof:
    gen.iterate(); torch.cuda.synchronize()

prof.export_chrome_trace("/tmp/densefuse_stack_trace.json")
T = prof.key_averages  # keep profiler alive
ev = prof.events()

# map correlation id -> launching CPU op stack (kernels carry the correlation id via their launch runtime event)
launch_stack = {}
for e in ev:
    if e.device_type == torch.autograd.DeviceType.CPU and e.name.startswith("cudaLaunchKernel"):
        pass
cpu_ops = [e for e in ev if e.device_type == torch.autograd.DeviceType.CPU and e.stack]
kernels = [e for e in ev if e.device_type == torch.autograd.DeviceType.CUDA]

lines = []
for k in kernels:
    n = k.name
    if not ("index_elementwise" in n or "BUnaryFunctor<long" in n or "AUnaryFunctor<long" in n
            or "CUDAFunctor_add<long" in n or "CUDAFunctorOnSelf_add<int" in n or "direct_copy" in n):
        continue
    # innermost CPU op whose time window contains the kernel's launch: take the CPU op with the same
    # trace time-window (kernel start falls inside op interval) and a dsv4.py frame
    cand = [c for c in cpu_ops if c.time_range.start <= k.time_range.start <= c.time_range.end
            and any("dsv4.py" in s for s in c.stack)]
    if not cand:
        continue
    c = min(cand, key=lambda c: c.time_range.end - c.time_range.start)
    frame = next(s for s in c.stack if "dsv4.py" in s)
    lines.append(f"{k.time_range.elapsed_us():7.2f} {n[:55]:55s} | {frame}")
open("/tmp/densefuse_stack.txt", "w").write("\n".join(lines))
print("ATTRIBUTED", len(lines), flush=True)
for l in lines[:60]:
    print(l)
