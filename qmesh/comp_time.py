"""Per-component GPU time per decode step via CUDA events wrapped around module forwards.
Events are recorded on the current stream; elapsed is summed per component over warm steps.
Event wrappers add a little launch overhead; use the wall/GPU numbers from time_gen/prof_gen for absolute totals."""
import sys, os, collections; sys.path.insert(0, "/src/exllamav3")
import torch
from qmesh.prof.load import load, new_job
from exllamav3.modules.transformer import TransformerBlock
config, model, cache, tok, gen = load(max_history=5)
for _ in range(2):
    gen.enqueue(new_job(tok, 16))
    while gen.num_remaining_jobs(): gen.iterate()
torch.cuda.synchronize()

acc = collections.defaultdict(float)
cnt = collections.defaultdict(int)
pending = []  # (name, start, end)
def wrap(obj, attr, name):
    f = getattr(obj, attr)
    def w(*a, **k):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); r = f(*a, **k); e.record(); pending.append((name, s, e)); return r
    setattr(obj, attr, w)

wrapped = 0
for blk in model.modules:
    if not isinstance(blk, TransformerBlock): continue
    L = blk.layer_idx
    for sub, name in [("attn_hc", "hc_attn"), ("mlp_hc", "hc_mlp"), ("attn", "attn"), ("mlp", "moe_total"),
                      ("attn_norm", "norm"), ("mlp_norm", "norm")]:
        m = getattr(blk, sub, None)
        if m is not None: wrap(m, "forward", name) if sub not in ("attn_hc", "mlp_hc") else None
    if blk.attn_hc is not None:
        wrap(blk.attn_hc, "mix_norm", "hc_attn_mix"); wrap(blk.attn_hc, "apply_", "hc_attn_apply")
    if blk.mlp_hc is not None:
        wrap(blk.mlp_hc, "mix_norm", "hc_mlp_mix"); wrap(blk.mlp_hc, "apply_", "hc_mlp_apply")
    if blk.attn is not None:
        a = blk.attn
        for sub in ("compressor", "indexer"):
            mm = getattr(a, sub, None)
            if mm is not None and hasattr(mm, "forward"): wrap(mm, "forward", f"attn.{sub}")
    if blk.mlp is not None and blk.mlp.shared_experts is not None:
        wrap(blk.mlp.shared_experts, "forward", "shared_expert")
    if blk.mlp is not None:
        wrap(blk.mlp, "cpu_split_submit", "moe_remote_issue")
        wrap(blk.mlp, "cpu_split_combine", "moe_remote_combine")
    wrapped += 1
eng = [m for m in model.modules if type(m).__name__ == "EngramLayer"]
for m in eng: wrap(m, "forward", "engram")
head = [m for m in model.modules if type(m).__name__ in ("HyperHead",)]
for m in head: wrap(m, "forward", "hyper_head")
lm = model.modules[-1]
wrap(lm, "forward", "lm_head_module")
print("wrapped blocks", wrapped, "engram", len(eng), "head", len(head), "lm", type(lm).__name__)

def flush():
    torch.cuda.synchronize()
    for name, s, e in pending:
        acc[name] += s.elapsed_time(e); cnt[name] += 1
    pending.clear()

STEPS = int(os.environ.get("STEPS", 8))
gen.enqueue(new_job(tok, STEPS + 2)); gen.iterate(); torch.cuda.synchronize()
pending.clear()
n = 0
while gen.num_remaining_jobs() and n < STEPS:
    gen.iterate(); flush(); n += 1
print(f"STEPS={n}  (ms per step, sum over all layers; events include ~5us overhead each)")
tot = 0.0
for name in sorted(acc, key=lambda k: -acc[k]):
    print(f"  {name:22s} {acc[name]/n:9.3f} ms/step   calls/step={cnt[name]/n:7.1f}")
