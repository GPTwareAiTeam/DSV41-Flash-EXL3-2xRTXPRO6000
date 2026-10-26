import csv, json
import torch
from exllamav3 import Config, Model
from qmesh.worker_group_server import build_group

MODEL = "/cold"; PLAN = "/cold/placement.exl3.json"
rank = {}
for r in csv.DictReader(open("/cold/ranking.csv")):
    rank[(int(r["layer"]), int(r["expert_id"]))] = int(r["rank"])

def rate(layer, e):
    r = rank[(layer, e)]
    return 4.0 if r <= 4100 else 3.0 if r <= 7660 else 2.5 if r <= 10360 else 2.0

layer = 0
cold = json.load(open(PLAN))["layers"]["0"]["warm"]
config = Config.from_directory(MODEL); model = Model.from_config(config)
base = next(m for m in model.modules if m.key == "layers.0").mlp
graphs = []; xs = []; idsl = []; wl = []; streams = []; outs = []
for rv in (3.0, 2.5, 2.0):
    pos = [i for i, e in enumerate(cold) if rate(0, e) == rv]
    experts = [cold[i] for i in pos]
    ffn = build_group(base, experts, torch.device("cuda:0"), layer)
    x = torch.randn(1, 5120, dtype=torch.half, device="cuda")
    ids = torch.arange(6, dtype=torch.long, device="cuda").view(1, 6)
    w = torch.full((1, 6), 1 / 6, dtype=torch.half, device="cuda")
    p = {"external_selected": ids, "external_weights": w}
    for _ in range(5):
        ffn.forward(x, p)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = ffn.forward(x, p)
    graphs.append(g); xs.append(x); idsl.append(ids); wl.append(w); outs.append(y)
    streams.append(torch.cuda.Stream())
current = torch.cuda.current_stream()
# warm each stream
for s, g in zip(streams, graphs):
    s.wait_stream(current)
    with torch.cuda.stream(s):
        g.replay()
for s in streams: current.wait_stream(s)
torch.cuda.synchronize()
a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
a.record()
for _ in range(50):
    for s, g in zip(streams, graphs):
        s.wait_stream(current)
        with torch.cuda.stream(s):
            g.replay()
    for s in streams:
        current.wait_stream(s)
    out = outs[0] + outs[1] + outs[2]
b.record(); torch.cuda.synchronize()
print("multigraph3_layer_ms", a.elapsed_time(b) / 50, "out", tuple(out.shape), flush=True)
