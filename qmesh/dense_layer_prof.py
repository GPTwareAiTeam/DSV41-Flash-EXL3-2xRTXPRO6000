"""DenseFuse: per-layer non-MoE kernel breakdown for one warm decode step (GPU, P7-local, null remote).
Wraps attention/HC/norm submodule forwards with record_function; attributes each kernel to its innermost
range via correlation id. Prints per-layer (layer 2, a non-Engram layer; layer 1 has Engram) kernel list
and aggregate µs/layer for the non-MoE part. Usage: see notes/DenseFuse.md."""
import sys, os, json, collections, re
sys.path.insert(0, "/src/exllamav3/qmesh"); sys.path.insert(0, "/src/exllamav3")
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from dense_prof import load, job

model, tok, gen = load()
for _ in range(2):
    gen.enqueue(job(tok, 8))
    while gen.num_remaining_jobs(): gen.iterate()
torch.cuda.synchronize()
gen.enqueue(job(tok, 6)); gen.iterate(); torch.cuda.synchronize()

def wrap(obj, name, label):
    f = getattr(obj, name)
    def w(*a, **k):
        with record_function(label):
            return f(*a, **k)
    setattr(obj, name, w)

for m in model.modules:
    li = getattr(m, "layer_idx", None)
    if li is None or not hasattr(m, "attn"): continue
    L = f"L{li}"
    if m.attn_hc is not None:
        wrap(m.attn_hc, "mix_norm", f"{L}.attn_hc.mix")
        wrap(m.attn_hc, "apply_", f"{L}.attn_hc.apply")
    if m.mlp_hc is not None:
        wrap(m.mlp_hc, "mix_norm", f"{L}.mlp_hc.mix")
        wrap(m.mlp_hc, "apply_", f"{L}.mlp_hc.apply")
    if m.attn_norm is not None: wrap(m.attn_norm, "forward", f"{L}.attn_norm")
    if m.attn is not None: wrap(m.attn, "forward", f"{L}.ATTN")
    if m.mlp is not None: wrap(m.mlp, "forward", f"{L}.MLP")
    if m.mlp_norm is not None: wrap(m.mlp_norm, "forward", f"{L}.mlp_norm")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    gen.iterate(); torch.cuda.synchronize()
out = "/tmp/densefuse_layer_trace.json"
prof.export_chrome_trace(out)

T = json.load(open(out)); ev = T["traceEvents"] if isinstance(T, dict) else T
ranges = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation"]
cpu_rt = {}
kernels = []
for e in ev:
    if e.get("ph") != "X": continue
    c = e.get("cat")
    if c == "cuda_runtime":
        cid = e.get("args", {}).get("correlation")
        if cid is not None: cpu_rt[cid] = (e["tid"], e["ts"])
    elif c == "kernel":
        kernels.append(e)
byt = collections.defaultdict(list)
for r in ranges: byt[r["tid"]].append(r)
def owner(tid, ts):
    best = None
    for r in byt[tid]:
        if r["ts"] <= ts <= r["ts"] + r["dur"] and (best is None or r["dur"] < best["dur"]): best = r
    return best
per = collections.defaultdict(list)  # label -> [(ts, name, dur)]
for k in kernels:
    cid = k.get("args", {}).get("correlation")
    if cid not in cpu_rt: continue
    tid, ts = cpu_rt[cid]
    r = owner(tid, ts)
    per[r["name"] if r else "(none)"].append((k["ts"], k["name"], k["dur"]))
json.dump({k: v for k, v in per.items()}, open("/tmp/densefuse_per_range.json", "w"))
print("RANGES", len(per))
