import copy, json, time
import torch
from exllamav3 import Config, Model
import csv

MODEL = "/cold"
PLAN = "/cold/placement.exl3.json"
rank = {}
for r in csv.DictReader(open("/cold/ranking.csv")):
    rank[(int(r["layer"]), int(r["expert_id"]))] = int(r["rank"])

def rate(layer, expert):
    r = rank[(layer, expert)]
    return 4.0 if r <= 4100 else 3.0 if r <= 7660 else 2.5 if r <= 11460 else 2.0

plan = json.load(open(PLAN))
layer = 0
cold = plan["layers"][str(layer)]["warm"]
group = [e for e in cold if rate(layer, e) == 3.0]
config = Config.from_directory(MODEL)
model = Model.from_config(config)
block = next(m for m in model.modules if m.key == f"layers.{layer}")
base = block.mlp
ffn = copy.deepcopy(base)
ffn.gates = [ffn.gates[e] for e in group]
ffn.ups = [ffn.ups[e] for e in group]
ffn.downs = [ffn.downs[e] for e in group]
keep = set(ffn.gates + ffn.ups + ffn.downs)
ffn.modules = [m for m in ffn.modules if m in keep]; ffn.load_routing = lambda **kw: None
ffn.num_experts = len(group)
ffn.num_local_experts = len(group)
ffn.routing_first = 0
ffn.routing_last = len(group)
ffn.routing_gate = base.routing_gate
ffn.shared_experts = None
ffn.cpu_split_first = None
ffn._split_perm = None
ffn.load(torch.device("cuda:0"))
print("group", len(group), "loaded", torch.cuda.memory_allocated(0) / 2**30, flush=True)

x = torch.randn(1, 5120, dtype=torch.half, device="cuda")
ids = torch.arange(6, dtype=torch.long, device="cuda").view(1, 6)
w = torch.full((1, 6), 1 / 6, dtype=torch.half, device="cuda")
params = {"external_selected": ids, "external_weights": w}
for _ in range(10):
    ffn.forward(x, params)
torch.cuda.synchronize()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = ffn.forward(x, params)
torch.cuda.synchronize()
s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(50):
    g.replay()
e.record(); torch.cuda.synchronize()
print("graph_group_layer_ms", s.elapsed_time(e) / 50, "out", tuple(y.shape), flush=True)
