"""Per-operator CUDA-event timing (no profiler) for one decode step, summed over 40 layers.
Event brackets are placed on the real Linear/MultiLinear/RMSNorm/HyperConnection/MoE callables, so the
GPU time of each operator is measured; the launch gap between operators is in 'gap'."""
import sys, os, collections; sys.path.insert(0, "/src/exllamav3")
import torch
from exllamav3.modules.transformer import TransformerBlock
from qmesh.prof.load import load, new_job
config, model, cache, tok, gen = load(max_history=5)
for _ in range(2):
    gen.enqueue(new_job(tok, 16))
    while gen.num_remaining_jobs(): gen.iterate()
torch.cuda.synchronize()
acc = collections.defaultdict(float); cnt = collections.defaultdict(int); pend = []
on = {"v": False}
def wrap(obj, attr, name):
    f = getattr(obj, attr)
    def w(*a, **k):
        if not on["v"]: return f(*a, **k)
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); r = f(*a, **k); e.record(); pend.append((name, s, e)); return r
    setattr(obj, attr, w)
for blk in model.modules:
    if not isinstance(blk, TransformerBlock): continue
    a = blk.attn; m = blk.mlp
    for nm in ("q_a", "q_b", "wkv", "wo_b"):
        if hasattr(a, nm) and getattr(a, nm) is not None: wrap(getattr(a, nm), "forward", f"attn.{nm}")
    if getattr(a, "multi_qkv", None) is not None: wrap(a.multi_qkv, "forward", "attn.multi_qkv")
    for nm in ("q_norm", "kv_norm"): wrap(getattr(a, nm), "forward", "attn.norm")
    wrap(blk.attn_norm, "forward", "norm.attn_in"); wrap(blk.mlp_norm, "forward", "norm.mlp_in")
    wrap(a, "forward", "attn.total")
    wrap(m.routing_gate, "forward", "moe.router_gemm")
    wrap(m, "forward", "moe.total")
    if m.shared_experts is not None: wrap(m.shared_experts, "forward", "moe.shared_expert")
    wrap(blk.attn_hc, "mix_norm", "hc.attn_mix"); wrap(blk.attn_hc, "apply_", "hc.attn_apply")
    wrap(blk.mlp_hc, "mix_norm", "hc.mlp_mix"); wrap(blk.mlp_hc, "apply_", "hc.mlp_apply")
for md in model.modules:
    if type(md).__name__ == "EngramLayer": wrap(md, "forward", "engram")
    if type(md).__name__ == "HyperHead": wrap(md, "forward", "hyper_head")
wrap(model.modules[-1], "forward", "lm_head")
wrap(model.modules[0], "forward", "embed")
STEPS = int(os.environ.get("STEPS", 12))
gen.enqueue(new_job(tok, STEPS + 64)); gen.iterate(); torch.cuda.synchronize()
on["v"] = True
pend.clear()
n = 0
walls = []
import time
while gen.num_remaining_jobs() and n < STEPS:
    t = time.perf_counter(); gen.iterate(); torch.cuda.synchronize(); walls.append(time.perf_counter() - t)
    for name, s, e in pend: acc[name] += s.elapsed_time(e); cnt[name] += 1
    pend.clear(); n += 1
on["v"] = False
tot_ops = 0.0
print(f"STEPS={n} (events add ~1-3 us each; op times are GPU-span incl. launch-to-completion of that op)")
for k in sorted(acc, key=lambda k: -acc[k]):
    print(f"  {k:24s} {acc[k]/n:9.3f} ms/step  calls/step={cnt[k]/n:6.1f}  per-call={acc[k]/max(cnt[k],1)*1e3:8.1f} us")
print(f"  step wall (event-instrumented) mean={sum(walls)/len(walls)*1e3:.3f} ms")
