"""DenseFuse correctness reference: 8 greedy decode steps; saves argmax tokens and lm_head logits.
Usage: python3 dense_ref.py <out.pt>   (compare with: dense_ref.py <out.pt> <ref.pt>)"""
import sys
sys.path.insert(0, "/src/exllamav3/qmesh"); sys.path.insert(0, "/src/exllamav3")
import torch
from dense_prof import load, job
model, tok, gen = load()
logits = []
lm = [m for m in model.modules if getattr(m, "key", "") == "head"]
assert lm, "lm_head module not found"
_orig = lm[0].forward
def _wrapped(x, params, *a, **k):
    y = _orig(x, params, *a, **k)
    logits.append(y.detach().float().cpu().reshape(-1, y.shape[-1])[-1].clone())
    return y
lm[0].forward = _wrapped
prompts = ["Hi", "The capital of France is"]
res = {}
for p in prompts:
    logits.clear()
    j = job(tok, 8, p)
    gen.enqueue(j)
    while gen.num_remaining_jobs(): gen.iterate()
    res[p] = {"tokens": list(j.full_completion_ids) if hasattr(j, "full_completion_ids") else None,
              "logits": torch.stack(logits).clone()}
    print("STEPS", p, len(logits), flush=True)
out = sys.argv[1]
torch.save(res, out)
print("SAVED", out, {p: (res[p]["logits"].shape, res[p]["tokens"]) for p in prompts})
if len(sys.argv) > 2:
    ref = torch.load(sys.argv[2])
    for p in prompts:
        a, b = res[p]["logits"], ref[p]["logits"]
        am = (a.argmax(-1) == b.argmax(-1)).all().item()
        print(f"CMP {p!r}: argmax_equal={am} max_abs_logit_diff={(a-b).abs().max().item():.6g}")
