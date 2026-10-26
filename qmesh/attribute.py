"""Attribute GPU kernel time and launch counts from breakdown_trace.json to record_function ranges.
A kernel belongs to a range if its launching runtime call (matched by correlation id) lies inside the
range's CPU interval on the same thread. Component = range name with layer index stripped."""
import json, re, collections, sys
p = "/home/treadall/src/exllamav3-dsv41/qmesh/prof/breakdown_trace.json"
T = json.load(open(p))
ev = T["traceEvents"] if isinstance(T, dict) else T
ranges = [e for e in ev if e.get("ph") == "X" and e.get("cat") == "user_annotation"]
cpu_rt = {}   # correlation -> (tid, ts)
kernels = []
for e in ev:
    if e.get("ph") != "X": continue
    cat = e.get("cat")
    if cat == "cuda_runtime":
        c = e.get("args", {}).get("correlation")
        if c is not None: cpu_rt[c] = (e["tid"], e["ts"])
    elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
        kernels.append(e)
# index ranges per thread
byt = collections.defaultdict(list)
for r in ranges: byt[r["tid"]].append(r)
def owner(tid, ts):
    best = None
    for r in byt[tid]:
        if r["ts"] <= ts <= r["ts"] + r["dur"]:
            if best is None or r["dur"] < best["dur"]: best = r  # innermost
    return best
stats = collections.defaultdict(lambda: [0.0, 0])
unattr = [0.0, 0]
for k in kernels:
    c = k.get("args", {}).get("correlation")
    if c not in cpu_rt: unattr[0] += k["dur"]; unattr[1] += 1; continue
    tid, ts = cpu_rt[c]
    r = owner(tid, ts)
    if r is None: unattr[0] += k["dur"]; unattr[1] += 1; continue
    name = re.sub(r"^L\d+\.", "", r["name"]) if r["name"].startswith("L") else r["name"]
    stats[name][0] += k["dur"]; stats[name][1] += 1
    if r["name"].startswith("L"):
        stats["__layer_" + r["name"].split(".")[0]][0] += 0  # placeholder for totals
N = int(sys.argv[1]) if len(sys.argv) > 1 else 16
print("ms per step (kernel-busy sum; kernels/step):")
tot = sum(v[0] for k, v in stats.items() if not k.startswith("__"))
for k, (us, c) in sorted(((k, v) for k, v in stats.items() if not k.startswith("__")), key=lambda kv: -kv[1][0]):
    print(f"  {k:24s} {us/N/1e3:8.3f} ms   kernels/step={c/N:7.1f}")
print(f"  {'(unattributed)':24s} {unattr[0]/N/1e3:8.3f} ms   kernels/step={unattr[1]/N:7.1f}")
print(f"  {'TOTAL':24s} {(tot+unattr[0])/N/1e3:8.3f} ms")
