"""Eager baseline: warm decode wall time per Generator step (P7 hot-only, null remote)."""
import sys, time, os; sys.path.insert(0, "/src/exllamav3")
import torch
from qmesh.graph.load import load, new_job
config, model, cache, tok, gen = load(max_history=5)
N = int(os.environ.get("NTOK", 64))
for _ in range(2):
    gen.enqueue(new_job(tok, 16))
    while gen.num_remaining_jobs(): gen.iterate()
torch.cuda.synchronize()
gen.enqueue(new_job(tok, N))
marks = []
while gen.num_remaining_jobs():
    gen.iterate(); torch.cuda.synchronize(); marks.append(time.perf_counter())
steps = sorted((b - a) for a, b in zip(marks[:-1], marks[1:])[1:] if False) if False else None
d = [b - a for a, b in zip(marks[:-1], marks[1:])][1:]
print(f"EAGER steps={len(d)} mean={sum(d)/len(d)*1e3:.3f} ms min={min(d)*1e3:.3f} max={max(d)*1e3:.3f}", flush=True)
