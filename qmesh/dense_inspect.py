"""DenseFuse: report attention fan/multi state per layer on the loaded P7 model (no timing)."""
import sys
sys.path.insert(0, "/src/exllamav3/qmesh"); sys.path.insert(0, "/src/exllamav3")
import torch
from dense_prof import load, job
model, tok, gen = load()
gen.enqueue(job(tok, 3)); gen.iterate(); torch.cuda.synchronize()
for m in model.modules:
    if not hasattr(m, "attn") or m.attn is None or not hasattr(m.attn, "x_fan_ready"): continue
    a = m.attn
    if m.layer_idx not in (0, 2, 3, 20):
        continue
    def q(l):
        i = getattr(l, "inner", None)
        return (type(l).__name__, l.quant_type, getattr(i, "K", None), getattr(i, "mcg", None), getattr(i, "mul1", None), l.out_features, getattr(l, "in_features", None))
    print(f"L{m.layer_idx} ready={a.x_fan_ready} x_fan={a.x_fan is not None} q_fan={a.q_fan is not None} qb_multi={a.qb_multi is not None} wob_multi={a.wob_multi is not None} wo_a_multi={getattr(a,'wo_a_multi',None) is not None} q_head_norm={a.q_head_norm is not None} indexer={a.indexer is not None} comp={a.compressor is not None}")
    for name in ["q_a", "wkv", "q_b", "wo_b"]:
        l = getattr(a, name, None)
        if l is not None: print("   ", name, q(l))
    if a.compressor is not None: print("    comp.wkv", q(a.compressor.wkv), "wgate", q(a.compressor.wgate))
    if a.indexer is not None: print("    idx.wkv", q(a.indexer.wkv), "idx_wq_b", q(a.idx_wq_b))
