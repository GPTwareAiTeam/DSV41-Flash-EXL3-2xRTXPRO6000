"""DenseFuse: join layer-20 index/long kernels to their launching dsv4.py frame via the exported trace
(kernel -> cuda_runtime launch by correlation -> innermost enclosing python_function on the same thread)."""
import json, re, sys
T=json.load(open(sys.argv[1])); ev=T["traceEvents"] if isinstance(T,dict) else T
rt={}
for e in ev:
    if e.get("cat")=="cuda_runtime" and "correlation" in e.get("args",{}): rt[e["args"]["correlation"]]=e
py=[e for e in ev if e.get("cat")=="python_function" and e.get("ph")=="X"]
want=re.compile(r"index_elementwise|BUnaryFunctor<long|AUnaryFunctor<long|CUDAFunctor_add<long|CUDAFunctorOnSelf_add<int|AUnaryFunctor<int|direct_copy")
out=[]
for k in ev:
    if k.get("cat")!="kernel" or not want.search(k["name"]): continue
    r=rt.get(k["args"].get("correlation"))
    if r is None: continue
    t,tid=r["ts"],r["tid"]
    cand=[p for p in py if p["tid"]==tid and p["ts"]<=t<=p["ts"]+p["dur"] and "dsv4.py" in p["name"]]
    if not cand: continue
    c=min(cand,key=lambda p:p["dur"])
    out.append((k["dur"],k["name"][:50],c["name"]))
from collections import Counter
agg=Counter()
for d,n,f in out: agg[f]+=d
print("kernels", len(out))
for f,d in sorted(agg.items(), key=lambda x:-x[1])[:20]: print(f"{d:8.2f} us  {f}")
