import copy, csv, json, time
import torch
from exllamav3 import Config, Model
from qmesh.worker_group_server import build_group, forward_group

MODEL = "/cold"
PLAN = "/cold/placement.exl3.json"
rank = {}
for r in csv.DictReader(open("/cold/ranking.csv")):
    rank[(int(r["layer"]), int(r["expert_id"]))] = int(r["rank"])

def rate(layer, expert):
    r = rank[(layer, expert)]
    return 4.0 if r <= 4100 else 3.0 if r <= 7660 else 2.5 if r <= 10360 else 2.0

plan = json.load(open(PLAN))
layer = 0
cold = plan["layers"][str(layer)]["warm"]
config = Config.from_directory(MODEL)
model = Model.from_config(config)
base = next(m for m in model.modules if m.key == f"layers.{layer}").mlp
groups = {}
mapping = {}
for rate_value in (3.0, 2.5, 2.0):
    positions = [i for i, e in enumerate(cold) if rate(layer, e) == rate_value]
    experts = [cold[i] for i in positions]
    groups[rate_value] = build_group(base, experts, torch.device("cuda:0"), layer)
    for local, pos in enumerate(positions):
        mapping[pos] = (rate_value, local)
layers = {layer: (groups, mapping)}

# one assignment from each K group
ids = torch.tensor([[0, mapping and next(p for p,m in mapping.items() if m[0]==3.0),
                      next(p for p,m in mapping.items() if m[0]==2.5),
                      next(p for p,m in mapping.items() if m[0]==2.0), 0, 0]],
                   dtype=torch.long, device="cuda")
x = torch.randn(1, 5120, dtype=torch.half, device="cuda")
w = torch.full((1, 6), 1 / 6, dtype=torch.half, device="cuda")

for _ in range(10):
    forward_group(layers, layer, x, ids, w)
torch.cuda.synchronize()
stream = torch.cuda.Stream()
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g, stream=stream):
    y = forward_group(layers, layer, x, ids, w)
torch.cuda.synchronize()
s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(50):
    g.replay()
e.record(); torch.cuda.synchronize()
print("graph3_layer_ms", s.elapsed_time(e) / 50, "out", tuple(y.shape), flush=True)
