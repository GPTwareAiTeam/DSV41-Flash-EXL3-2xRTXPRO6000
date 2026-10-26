"""torch.profiler over warm Generator decode steps: GPU busy union, kernel count/token, top kernels,
CPU-side op time, and host time inside model.forward vs generator overhead."""
import sys, time, os, collections, json; sys.path.insert(0, "/src/exllamav3")
import torch
from torch.profiler import profile, ProfilerActivity
from qmesh.prof.load import load, new_job
config, model, cache, tok, gen = load(max_history=5)
for _ in range(2):
    gen.enqueue(new_job(tok, 16))
    while gen.num_remaining_jobs(): gen.iterate()
torch.cuda.synchronize()

# host time inside model.forward
fw = {"t": 0.0, "n": 0}
orig_fwd = model.forward
def timed_fwd(*a, **k):
    t = time.perf_counter(); r = orig_fwd(*a, **k); fw["t"] += time.perf_counter() - t; fw["n"] += 1; return r
model.forward = timed_fwd

STEPS = int(os.environ.get("STEPS", 24))
gen.enqueue(new_job(tok, STEPS + 2))
gen.iterate()  # prefill step, outside profile
torch.cuda.synchronize()
fw["t"] = 0.0; fw["n"] = 0
t0 = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
    n = 0
    while gen.num_remaining_jobs() and n < STEPS:
        gen.iterate(); torch.cuda.synchronize(); n += 1
wall = time.perf_counter() - t0
model.forward = orig_fwd
print(f"STEPS={n} wall/step={wall/n*1e3:.3f} ms  model.forward host time/step={fw['t']/max(fw['n'],1)*1e3:.3f} ms calls={fw['n']}")

evs = prof.events()
kern = [e for e in evs if e.device_type == torch.autograd.DeviceType.CUDA]
# GPU busy union from kernel intervals
iv = sorted((e.time_range.start, e.time_range.end) for e in kern)
busy = 0.0; cs, ce = None, None
for s, e in iv:
    if cs is None: cs, ce = s, e
    elif s <= ce: ce = max(ce, e)
    else: busy += ce - cs; cs, ce = s, e
if cs is not None: busy += ce - cs
print(f"KERNELS/step={len(kern)/n:.1f}  GPU_busy_union/step={busy/n/1e3:.3f} ms  sum_kernel_time/step={sum(e.time_range.elapsed_us() for e in kern)/n/1e3:.3f} ms")

agg = collections.defaultdict(lambda: [0, 0.0])
for e in kern:
    agg[e.name][0] += 1; agg[e.name][1] += e.time_range.elapsed_us()
rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
print("TOP KERNELS (per step): count, us, name")
for name, (c, us) in rows[:40]:
    print(f"{c/n:7.1f} {us/n:10.1f} {name[:150]}")
prof.export_chrome_trace("/home/treadall/src/exllamav3-dsv41/qmesh/prof/gen_trace.json")
# CPU ops
cpu = collections.defaultdict(lambda: [0, 0.0])
for e in evs:
    if e.device_type == torch.autograd.DeviceType.CPU and e.name.startswith(("aten::", "exllamav3", "cuda")) and e.cpu_time_total > 0 and e.name.startswith("aten::"):
        cpu[e.name][0] += e.count if hasattr(e,'count') else 1; cpu[e.name][1] += e.cpu_time_total
print("TOP CPU ops (per step, self-inclusive ms):")
for name, (c, us) in sorted(cpu.items(), key=lambda kv: -kv[1][1])[:25]:
    print(f"  {name:40s} {us/n/1e3:8.3f} ms")
