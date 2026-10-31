"""Re-quantize ONLY the DSpark drafter's routed experts (mtp.{0,1,2}.ffn.experts.*.w{1,2,3}) at K bits, exactly as
convert.py's uncalibrated side-model path does (quantize_side_model -> quantize_linears_single with no capture:
placeholder Hessian, mul1 codebook, seed = module index, apply_out_scales None, no hq). Attention, shared experts,
gates, norms, main_proj and the drafter head are NOT touched. Output: one safetensors file with the new expert
tensors under their checkpoint keys (full, unsharded). qmesh/tp2/drafter_experts.py combines sets at different rates
into the per-expert hot/cold layout and splits them per rank.

Usage (one GPU): BASE=<original HF checkpoint dir> python3 quant_drafter_experts.py <bits> <out.safetensors>"""
import os
import sys
import time
import torch
from safetensors.torch import save_file
from exllamav3 import Config, Model
from exllamav3.modules import Linear
from exllamav3.conversion.convert_model import make_quant_args, get_H_data

BASE = os.environ.get("BASE", "/models/dsv41/exl3-base/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8")
K = float(sys.argv[1])
K = int(K) if K.is_integer() else K   # the tile quantizer takes integer K; 1.5/2.5/3.5 stay float (mul1 half rates)
OUT = sys.argv[2]
dev = torch.device("cuda", 0)
args = {"work_dir": os.path.dirname(OUT), "apply_out_scales": None, "codebook": "mul1", "hessians": "",
        "hessians_reg": 0.025, "scaled_hessian": False}

config = Config.from_directory(BASE)
mtp = Model.from_config(config, component = "mtp")
out = {}
t0 = time.time()
for idx, module in enumerate(mtp.modules):
    linears = [m for m in module if isinstance(m, Linear) and m.qmap and ".ffn.experts." in m.key]
    if not linears:
        continue
    print(f" -- {module.key}: {len(linears)} routed-expert linears at K={K:g}", flush = True)
    defer = module.can_defer_load()
    if defer:
        module.config.stc.begin_deferred_load()
    try:
        module.load(dev, keep_source_weights = True)
    finally:
        if defer:
            module.config.stc.end_deferred_load()
    module.config.stc.close()
    for lin in [m for m in module if isinstance(m, Linear) and m.qmap and m.device is not None]:
        lin.inner.swap_cpu()
    for n, lin in enumerate(linears):
        qa = make_quant_args(args, idx, K, [dev])
        H, qa["H_out"] = get_H_data(args, lin, None, None)
        lin.convert_exl3(H, quant_args = qa)
        lin.inner.swap_cpu()
        out.update({k: v.contiguous().cpu() for k, v in lin.get_tensors().items()})
        if n % 96 == 0:
            print(f"    {n}/{len(linears)}  {time.time() - t0:.0f} s", flush = True)
    module.unload()
    config.stc.close()
save_file(out, OUT)
print(f" -- wrote {len(out)} tensors to {OUT} in {time.time() - t0:.0f} s", flush = True)
