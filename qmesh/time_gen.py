"""Warm decode wall time through the real Generator (P7 hot-only, null remote)."""
import sys, time, os; sys.path.insert(0, "/src/exllamav3")
import torch
from qmesh.prof.load import load, new_job
config, model, cache, tok, gen = load(max_history=5)
N = int(os.environ.get("NTOK", 64))
for warm in range(2):
    gen.enqueue(new_job(tok, 16)); 
    while gen.num_remaining_jobs(): gen.iterate()
torch.cuda.synchronize()
job = new_job(tok, N)
gen.enqueue(job)
marks = []
t0 = time.perf_counter()
while gen.num_remaining_jobs():
    r = gen.iterate()
    torch.cuda.synchronize()
    marks.append(time.perf_counter())
t1 = time.perf_counter()
# first iterate includes prefill; drop it
steps = [b - a for a, b in zip(marks[:-1], marks[1:])]
d = steps[1:]
d.sort()
print(f"GEN decode steps={len(d)} mean={sum(d)/len(d)*1e3:.3f} ms median={d[len(d)//2]*1e3:.3f} ms min={d[0]*1e3:.3f} max={d[-1]*1e3:.3f} ms")
print(f"GEN total={t1-t0:.3f}s prefill_iter={steps[0]*1e3:.1f}ms")
