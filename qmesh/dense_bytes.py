"""DenseFuse: weight bytes per dense projection for layers 3 and 20 (non-MoE), via Linear.storage_size().
Usage: python3 dense_bytes.py -> prints per-layer MB and writes /tmp/densefuse_bytes.json."""
import sys, json
sys.path.insert(0, "/src/exllamav3/qmesh"); sys.path.insert(0, "/src/exllamav3")
from dense_prof import load

model, tok, gen = load()
out = {}
for m in model.modules:
    li = getattr(m, "layer_idx", None)
    if li not in (3, 20):
        continue
    rec = {}
    attn = getattr(m, "attn", None)
    if attn is not None:
        for n in ("q_a", "q_b", "wkv", "wo_b", "idx_wq_b", "idx_wk"):
            l = getattr(attn, n, None)
            if l is not None:
                rec[f"attn.{n}"] = l.storage_size()
        rec["attn.wo_a"] = sum(l.storage_size() for l in attn.wo_a)
        if attn.compressor is not None:
            for n in ("wkv", "wgate"):
                l = getattr(attn.compressor, n, None)
                if l is not None:
                    rec[f"attn.comp.{n}"] = l.storage_size()
        if attn.indexer is not None:
            for n in ("wkv", "wgate"):
                l = getattr(attn.indexer, n, None)
                if l is not None:
                    rec[f"attn.idx.{n}"] = l.storage_size()
    out[str(li)] = rec
    print("L", li, {k: round(v / 1e6, 2) for k, v in rec.items()}, "MB total", round(sum(rec.values()) / 1e6, 2), flush=True)

json.dump(out, open("/tmp/densefuse_bytes.json", "w"), indent=1)
