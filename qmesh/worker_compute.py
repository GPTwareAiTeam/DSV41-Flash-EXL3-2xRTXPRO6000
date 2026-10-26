import json, time
import torch
from exllamav3 import Config, Model
from qmesh.worker_server import forward_experts

PLACEMENT = "/cold/placement.exl3.json"
MODEL = "/cold"
LAYER = 0

plan = json.load(open(PLACEMENT))
cold = plan["layers"][str(LAYER)]["warm"]
config = Config.from_directory(MODEL)
config.infer_params.no_reconstruct = True
model = Model.from_config(config)
block = next(m for m in model.modules if m.key == f"layers.{LAYER}")
ffn = block.mlp
ffn.gates = [ffn.gates[e] for e in cold]
ffn.ups = [ffn.ups[e] for e in cold]
ffn.downs = [ffn.downs[e] for e in cold]
keep = set(ffn.gates + ffn.ups + ffn.downs)
ffn.modules = [m for m in ffn.modules if m in keep or m.key == f"layers.{LAYER}.ffn.gate"]
ffn.num_local_experts = len(cold)
ffn.routing_first = 0
ffn.routing_last = len(cold)
ffn.cpu_split_first = None
ffn._split_perm = None
ffn.load(torch.device("cuda:0"))

x = torch.randn(1, 5120, dtype=torch.half, device="cuda")
ids = torch.arange(6, dtype=torch.long, device="cuda").view(1, 6)
w = torch.full((1, 6), 1 / 6, dtype=torch.float, device="cuda")
for _ in range(10):
    forward_experts(ffn, x, ids, w)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(50):
    forward_experts(ffn, x, ids, w)
end.record(); torch.cuda.synchronize()
print("compute_layer_ms", start.elapsed_time(end) / 50, "allocated_gib", torch.cuda.memory_allocated(0) / 2**30)
