import sys; sys.path.insert(0, "/src/exllamav3")
from qmesh.prof.load import load, new_job
import torch
config, model, cache, tok, gen = load()
blk = model.modules[2]
print("block attrs:")
for k, v in vars(blk).items():
    if isinstance(v, torch.nn.Module) or hasattr(v, "forward") and not isinstance(v,(int,float,str,bool,type(None),list,dict,tuple)):
        print(" ", k, type(v).__name__)
for sub in ["attn", "mlp", "attn_hc", "mlp_hc", "attn_norm", "mlp_norm"]:
    m = getattr(blk, sub, None)
    if m is None: continue
    print(f"--{sub}: {type(m).__name__}")
    for k, v in vars(m).items():
        if type(v).__name__ in ("BlockSparseMLP","Linear","MultiLinear","RMSNorm","DSV4Attention","DSV4Compressor","QSAIndexer","HyperConnection","Module","ExpandStreams","GatedResidual","HyperHead") or type(v).__module__.startswith("exllamav3.modules"):
            print("   ", k, type(v).__name__)
mlp = blk.mlp
print("mlp vars:", [ (k, type(v).__name__) for k,v in vars(mlp).items() if not isinstance(v,(int,float,str,bool,type(None)))][:60])
print("num_experts", getattr(mlp,"num_experts",None), "hot first", getattr(mlp,"cpu_split_first",None), "per_tok", getattr(mlp,"num_experts_per_tok",None))
print("hc:", type(blk.attn_hc).__name__, getattr(blk.attn_hc,'sinkhorn_iters',None))
print("engram layers:", [ (m.key, type(m).__name__) for m in model.modules if 'engram' in str(getattr(m,'key','')).lower()])
