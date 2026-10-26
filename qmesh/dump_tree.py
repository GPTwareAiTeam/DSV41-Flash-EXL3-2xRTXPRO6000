import sys; sys.path.insert(0, "/src/exllamav3")
from qmesh.prof.load import load, new_job
import torch
config, model, cache, tok, gen = load()
def walk(m, depth, out, name):
    kids = [(n, c) for n, c in m._modules.items()] if hasattr(m, "_modules") else []
    out.append(f"{'  '*depth}{name}: {type(m).__name__} key={getattr(m,'key',None)}")
    if depth < 4:
        for n, c in kids:
            if c is not None: walk(c, depth+1, out, n)
out = []
print("top-level modules:", len(model.modules))
for i, m in enumerate(model.modules[:3]): print(i, type(m).__name__, getattr(m, "key", None))
for L in (0, 1, 2, 14):
    blk = [m for m in model.modules if type(m).__name__ == "TransformerBlock" and m.layer_idx == L]
    if blk: walk(blk[0], 0, out, f"layer{L}")
print("\n".join(out))
gen.enqueue(new_job(tok, 4)); 
while gen.num_remaining_jobs(): gen.iterate()
print("OK")
