"""Kernel and graph launches per decode step (Generator, bsz 1). Compare EXL3_WHOLE_GRAPH=0/1."""
import sys; sys.path.insert(0, "/src/exllamav3")
import torch
from torch.profiler import profile, ProfilerActivity
from qmesh.graph.load import load, new_job

config, model, cache, tok, gen = load(max_history=5)
gen.enqueue(new_job(tok, 24))
while gen.num_remaining_jobs():
    gen.iterate()
gen.enqueue(new_job(tok, 12))
steps = 0
with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    while gen.num_remaining_jobs():
        gen.iterate()
        steps += 1
torch.cuda.synchronize()
names = [e.name for e in prof.events()]
kern = sum(names.count(n) for n in ("cudaLaunchKernel", "cuLaunchKernel", "cudaLaunchKernelExC", "cudaLaunchCooperativeKernel"))
graph = names.count("cudaGraphLaunch")
print(f"STEPS {steps} kernel_launches/step {kern / steps:.0f} graph_launches/step {graph / steps:.0f}", flush=True)
