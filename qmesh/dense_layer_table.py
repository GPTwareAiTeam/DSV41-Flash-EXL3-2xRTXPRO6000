"""DenseFuse: per-layer kernel table from the layer-profile trace. Kernels are ordered by GPU time and
inherit the range label of the nearest CPU-labelled kernel before them (unlabelled DSA/elementwise kernels
run inside the same ATTN/HC section). Usage: python3 dense_layer_table.py <trace.json> <ranges.json>"""
import json, re, sys, collections
T = json.load(open(sys.argv[1])); ev = T["traceEvents"] if isinstance(T, dict) else T
R = json.load(open(sys.argv[2]))
ks = [e for e in ev if e.get("cat") == "kernel"]
rt = {}
for e in ev:
    if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}):
        rt[e["args"]["correlation"]] = e
lab = {}
for name, items in R.items():
    for ts, kn, d in items: pass
# rebuild correlation->label from range dump (ts-keyed)
label_by_ts = {}
for name, items in R.items():
    for ts, kn, d in items: label_by_ts[(round(ts, 3), kn)] = name
ks.sort(key=lambda e: e["ts"])
cur = "(none)"; rows = []
for e in ks:
    c = e["args"].get("correlation")
    l = label_by_ts.get((round(e["ts"], 3), e["name"]))
    if l is None:
        l = cur
    else:
        cur = l
    rows.append((e["ts"], e["dur"], l, e["name"]))
per = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])  # non-moe us, n, moe us, n
for ts, d, l, n in rows:
    m = re.match(r"L(\d+)\.(.*)", l)
    if not m: continue
    li, part = int(m.group(1)), m.group(2)
    if part == "MLP": per[li][2] += d; per[li][3] += 1
    else: per[li][0] += d; per[li][1] += 1
out = {"rows": [(ts, d, l, re.sub(r"\(.*", "", n)[:90]) for ts, d, l, n in rows]}
json.dump(out, open("/tmp/densefuse_seq.json", "w"))
print("layer  nonMoE_us  nonMoE_kern  MoE_us  MoE_kern")
for li in sorted(per):
    a = per[li]; print(f"{li:5d}  {a[0]:9.1f}  {a[1]:10d}  {a[2]:7.1f}  {a[3]:7d}")
tot = sum(v[0] for v in per.values()); n = sum(v[1] for v in per.values())
print("TOTAL nonMoE", round(tot, 1), "kern", n, "per layer", round(tot / len(per), 1))
