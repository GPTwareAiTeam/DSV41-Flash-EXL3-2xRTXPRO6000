"""One-shot P7-only decode breakdown (null remote, no GX1).
A) wall per Generator step; model.forward wall (synced) => generator overhead.
B) torch.profiler chrome trace with record_function component ranges; kernels/runtime launches
   attributed to the innermost enclosing component by correlation id.
C) local (GPU-resident) expert selections per layer per step => MoE bytes and GB/s.
Writes qmesh/prof/breakdown.json and prints a table."""
import sys, os, json, time, collections, contextlib; sys.path.insert(0, "/src/exllamav3")
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from exllamav3.modules.transformer import TransformerBlock
from qmesh.prof.load import load, new_job

OUT = "/src/exllamav3/qmesh/prof"
STEPS = int(os.environ.get("STEPS", 16))
config, model, cache, tok, gen = load(max_history=5)
for _ in range(3):
    gen.enqueue(new_job(tok, 16))
    while gen.num_remaining_jobs(): gen.iterate()
torch.cuda.synchronize()

# ---------- C: local selection counting + component record_function ranges ----------
sel_calls = collections.defaultdict(int)
counting = {"on": False}
blk_by_layer = {}
mlp_by_layer = {}
ranges_on = {"on": False}

def rf(name, fn):
    def w(*a, **k):
        if not ranges_on["on"]:
            return fn(*a, **k)
        with record_function(name):
            return fn(*a, **k)
    return w

for blk in model.modules:
    if not isinstance(blk, TransformerBlock): continue
    L = blk.layer_idx
    blk_by_layer[L] = blk
    mlp = blk.mlp
    mlp_by_layer[L] = mlp
    first = mlp.cpu_split_first
    orig_submit = mlp.cpu_split_submit
    # local-pick count is computed on-GPU and accumulated (no sync in the hot path)
    mlp._cnt_local = torch.zeros((), dtype=torch.long, device=mlp.device)
    def submit2(y, bsz, selected_experts, routing_weights, _o=orig_submit, _first=first, _m=mlp):
        if counting["on"]:
            _m._cnt_local += (selected_experts < _first).sum()
            sel_calls[id(_m)] += 1
        return _o(y, bsz, selected_experts, routing_weights)
    mlp.cpu_split_submit = submit2
    # component ranges (record_function only when ranges_on)
    blk.attn.forward = rf(f"L{L}.attn", blk.attn.forward)
    blk.attn_hc.mix_norm = rf(f"L{L}.hc_attn_mix", blk.attn_hc.mix_norm)
    blk.attn_hc.apply_ = rf(f"L{L}.hc_attn_apply", blk.attn_hc.apply_)
    blk.mlp_hc.mix_norm = rf(f"L{L}.hc_mlp_mix", blk.mlp_hc.mix_norm)
    blk.mlp_hc.apply_ = rf(f"L{L}.hc_mlp_apply", blk.mlp_hc.apply_)
    if mlp.shared_experts is not None:
        mlp.shared_experts.forward = rf(f"L{L}.shared_expert", mlp.shared_experts.forward)
    mlp.forward = rf(f"L{L}.moe", mlp.forward)
    blk.forward = rf(f"L{L}.block_total", blk.forward)
for m in model.modules:
    if type(m).__name__ == "EngramLayer":
        m.forward = rf(f"engram.{m.key}", m.forward)
head_mod = model.modules[-1]
head_mod.forward = rf("lm_head", head_mod.forward)
for m in model.modules:
    if type(m).__name__ == "HyperHead":
        m.forward = rf("hyper_head", m.forward)
    if type(m).__name__ == "Embedding":
        m.forward = rf("embed", m.forward)

# ---------- A: wall / forward-only timing ----------
fw = {"t": 0.0, "n": 0}
orig_fwd = model.forward
def timed_fwd(*a, **k):
    torch.cuda.synchronize(); t = time.perf_counter()
    r = orig_fwd(*a, **k)
    torch.cuda.synchronize(); fw["t"] += time.perf_counter() - t; fw["n"] += 1
    return r

JOB_TOKENS = 4 * STEPS + 64  # decode steps counted below; job never finishes early
gen.enqueue(new_job(tok, JOB_TOKENS))
# prefill: iterate until the first generated token is emitted (not timed)
r = gen.iterate(); torch.cuda.synchronize()   # iterate 0 = started+prefill+first token
assert any(x.get("stage") == "streaming" for x in r), r

# A: plain wall per decode step, no profiler, no range hooks
walls = []
model.forward = timed_fwd
while gen.num_remaining_jobs() and len(walls) < STEPS:
    t = time.perf_counter(); gen.iterate(); torch.cuda.synchronize(); walls.append(time.perf_counter() - t)
model.forward = orig_fwd
fwd_ms = fw["t"] / max(fw["n"], 1) * 1e3
w = sorted(walls); med = w[len(w)//2]
print(f"[A] generator step: mean={sum(walls)/len(walls)*1e3:.3f} ms median={med*1e3:.3f} ms  "
      f"model.forward(synced)={fwd_ms:.3f} ms  => generator/host overhead={sum(walls)/len(walls)*1e3-fwd_ms:.3f} ms", flush=True)

# B: profiled steps with component ranges and local-pick counting
counting["on"] = True; ranges_on["on"] = True
for m in mlp_by_layer.values(): m._cnt_local.zero_()
for m in mlp_by_layer.values(): m._cnt_local.zero_()
prof_steps = 0
trace_path = f"{OUT}/breakdown_trace.json"
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    while gen.num_remaining_jobs() and prof_steps < STEPS:
        gen.iterate(); torch.cuda.synchronize(); prof_steps += 1
prof.export_chrome_trace(trace_path)
counting["on"] = False; ranges_on["on"] = False
local_per_layer = {L: int(m._cnt_local.item()) / prof_steps for L, m in mlp_by_layer.items()}
print(f"[C] local (GPU-resident) expert picks per layer per step: mean={sum(local_per_layer.values())/40:.3f} total/step={sum(local_per_layer.values()):.2f}")
json.dump({"walls_ms": [x*1e3 for x in walls], "fwd_ms": fwd_ms, "prof_steps": prof_steps,
           "local_per_layer": local_per_layer}, open(f"{OUT}/breakdown.json", "w"), indent=1)
print("WROTE", trace_path, flush=True)
