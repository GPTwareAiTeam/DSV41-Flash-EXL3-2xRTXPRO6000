"""DenseFuse: per-projection kernel attribution for layers 3 and 20. Wraps each dense projection forward with a
record_function label so each GEMV/mgemm kernel is tagged with the weight it reads. Writes /tmp/densefuse_proj.json
as (label, dur_us, kernel_name) rows, in launch order."""
import sys, json
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
    if li not in (3, 20) or getattr(m, "attn", None) is None:
        continue
    at = m.attn
    L = f"L{li}"
    for n in ("q_a", "q_b", "wkv", "wo_b", "idx_wq_b", "idx_wk"):
        l = getattr(at, n, None)
        if l is not None:
            wrap(l, "forward", f"{L}.P.{n}")
    for i, l in enumerate(at.wo_a):
        wrap(l, "forward", f"{L}.P.wo_a{i}")
    if at.compressor is not None:
        if at.compressor.wkv is not None:
            wrap(at.compressor.wkv, "forward", f"{L}.P.comp.wkv")
        if at.compressor.wgate is not None:
            wrap(at.compressor.wgate, "forward", f"{L}.P.comp.wgate")
    if at.indexer is not None:
        if at.indexer.wkv is not None:
            wrap(at.indexer.wkv, "forward", f"{L}.P.idx.wkv")
        if at.indexer.wgate is not None:
            wrap(at.indexer.wgate, "forward", f"{L}.P.idx.wgate")

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    gen.iterate(); torch.cuda.synchronize()
prof.export_chrome_trace("/tmp/densefuse_proj_trace.json")

T = json.load(open("/tmp/densefuse_proj_trace.json"))
ev = T["traceEvents"] if isinstance(T, dict) else T
ranges = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation" and ".P." in e["name"]]
kernels = sorted([e for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel"], key=lambda e: e["ts"])
rows = []
for k in kernels:
    best = None
    for r in ranges:
        if r["tid"] == k.get("tid") or True:
            if r["ts"] <= k["ts"] <= r["ts"] + r["dur"] and (best is None or r["dur"] < best["dur"]):
                best = r
    rows.append((best["name"] if best else "-", k["dur"], k["name"][:80]))
json.dump(rows, open("/tmp/densefuse_proj.json", "w"))
for lab, d, n in rows:
    if ".P." in lab:
        print(f"{d:8.2f} {lab:22s} {n}", flush=True)
