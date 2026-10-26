"""
MixedK fused-decode kernel timing (item 4): ext.exl3_moe_coop direct calls at real layer geometry
(H=5120, I=2304, top-6, 384 experts, mul1, silu, gated). Run inside exl3-worker on one GPU.

Variants (identical routing, scratch and shapes; only the rate source differs):
  uniform3   : every expert K=3.0, no k2 tables (templated uniform path)
  mixed      : per-expert K from the real layer-3 distribution (148 K=2, 109 K=2.5, 60 K=3, 67 K=3.5),
               k2 tables passed (mixed path, one launch per projection)
  (b') mixed_seq: NOT implemented in this run (needs per-K-class expert masking the coop ABI lacks).
Modes: eager (synchronize per call) and graph (CUDA graph replay of the same launch).
Output: one JSON line per (variant, bsz, mode) with median us and effective GB/s.
"""
import json
import os
import statistics
import time

import torch

from exllamav3.ext import exllamav3_ext as ext
from testlib.exl3 import rand_scale, rand_trellis
from testlib.moe import ptr_table

H, I, E, TOPK = 5120, 2304, 384, 6
HO = H
ITERS = int(os.environ.get("BENCH_ITERS", "200"))
WARM = 20
DEV = torch.device("cuda:0")
ACT_SILU = 0


def class_counts():
    # real layer-3 K distribution (decoded from trellis shapes, see MixedK.md)
    counts = {2.0: 148, 2.5: 109, 3.0: 60, 3.5: 67}
    ks = []
    for k, n in counts.items():
        ks += [k] * n
    assert len(ks) == E
    # deterministic assignment, shuffled but fixed
    g = torch.Generator().manual_seed(1234)
    perm = torch.randperm(E, generator = g).tolist()
    return [ks[perm[e]] for e in range(E)]


def make_proj(k_in, n_out, Kvec, gen):
    """Per-expert trellis/suh/svh for a (k_in -> n_out) projection, each expert at its own K."""
    tr, su, sv = [], [], []
    for e in range(E):
        K = Kvec[e]
        tr.append(rand_trellis(k_in, n_out, K, gen, DEV))
        su.append(rand_scale(k_in, gen, DEV))
        sv.append(rand_scale(n_out, gen, DEV))
    return {
        "tabs": (ptr_table(tr, DEV), ptr_table(su, DEV), ptr_table(sv, DEV)),
        "bytes": sum(t.numel() * t.element_size() for t in tr),
        "tensors": (tr, su, sv),
        "Kvec": Kvec,
    }


def k2_table(Kvec):
    return torch.tensor([int(round(2 * k)) for k in Kvec], dtype = torch.uint8, device = DEV)


def scratch(bsz):
    slots = bsz * TOPK
    smax = 256  # production max (exl3_moe_coop TORCH_CHECK: 1..256 slots); 4*slots capped ksplit at 4
    d = {}
    d["gu_g"] = torch.empty((smax, 1, I), dtype = torch.half, device = DEV)
    d["gu_u"] = torch.empty((smax, 1, I), dtype = torch.half, device = DEV)
    d["act_out"] = torch.empty((smax, 1, I), dtype = torch.half, device = DEV)
    d["d_out"] = torch.empty((smax, 1, HO), dtype = torch.float, device = DEV)
    d["out"] = torch.full((8, H), float("nan"), dtype = torch.float, device = DEV)
    d["ctr"] = torch.zeros((smax * (I // 128) + 8 * (HO // 128) + 2 * smax + 3,), dtype = torch.int, device = DEV)
    d["had_g"] = torch.empty((smax, H), dtype = torch.half, device = DEV)
    d["had_u"] = torch.empty_like(d["had_g"])
    return d


def launch(x, sel, rw, g, u, dn, bsz, sc, k2g, k2u, k2d, Kg, Uk, Dk, mixed, out_rows = None):
    """One ext.exl3_moe_coop call. mixed=True passes the k2 tables; Kg/Uk/Dk are placeholders then."""
    o = sc["out"][:bsz] if out_rows is None else out_rows
    ext.exl3_moe_coop(
        x, sel, rw, -1, -1, H,
        *g["tabs"], *u["tabs"], *dn["tabs"],
        None, None, None,
        Kg, Uk, Dk, False, True,
        ACT_SILU, 0.0, True,
        sc["had_g"], sc["had_u"], sc["gu_g"], sc["gu_u"], sc["act_out"], sc["d_out"], sc["ctr"], o,
        None, None,
        k2g if mixed else None, k2u if mixed else None, k2d if mixed else None,
    )


def timed(fn, graph):
    if graph:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(WARM):
                fn()
        torch.cuda.current_stream().wait_stream(s)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            fn()
        torch.cuda.synchronize()
        run = gr.replay
    else:
        run = fn
        for _ in range(WARM):
            run()
        torch.cuda.synchronize()
    ts = []
    for _ in range(ITERS):
        t0 = time.perf_counter()
        run()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    return statistics.median(ts)


def bytes_touched(g, u, dn, sel):
    """Packed bytes of the distinct routed experts across all three projections."""
    experts = torch.unique(sel).tolist()
    tot = 0
    for name, p in (("g", g), ("u", u), ("d", dn)):
        tr = p["tensors"][0]
        tot += sum(tr[e].numel() * tr[e].element_size() for e in experts)
    return tot


def main():
    gen = torch.Generator().manual_seed(7)
    mixedK = class_counts()
    uniform = [3.0] * E
    g_m = make_proj(H, I, mixedK, gen)
    u_m = make_proj(H, I, mixedK, gen)
    d_m = make_proj(I, HO, mixedK, gen)
    g_u = make_proj(H, I, uniform, gen)
    u_u = make_proj(H, I, uniform, gen)
    d_u = make_proj(I, HO, uniform, gen)
    k2g, k2u, k2d = k2_table(mixedK), k2_table(mixedK), k2_table(mixedK)
    # per-K class index lists for the sequential (b') variant
    classes = sorted(set(mixedK))

    results = []
    for bsz in (1, 6):
        sel = torch.stack([torch.randperm(E, generator = torch.Generator().manual_seed(100 + b))[:TOPK]
                           for b in range(bsz)]).to(DEV)
        rw = (torch.rand((bsz, TOPK), generator = torch.Generator().manual_seed(200 + bsz)) * 0.9 + 0.1).half().to(DEV)
        x = (torch.randn((bsz, H), generator = torch.Generator().manual_seed(300 + bsz)) * 0.05).half().to(DEV)
        sc = scratch(bsz)
        touched_m = bytes_touched(g_m, u_m, d_m, sel)
        touched_u = bytes_touched(g_u, u_u, d_u, sel)

        for mode in ("eager", "graph"):
            graph = mode == "graph"
            # uniform3: templated path, K=3.0 scalars, no tables
            fu = lambda: launch(x, sel, rw, g_u, u_u, d_u, bsz, sc, None, None, None, 3.0, 3.0, 3.0, False)
            us = timed(fu, graph)
            results.append({"variant": "uniform3", "bsz": bsz, "mode": mode, "us": round(us, 2),
                            "bytes": touched_u, "GBps": round(touched_u / (us * 1e-6) / 1e9, 2)})
            print(json.dumps(results[-1]), flush = True)

            # mixed: one launch per projection, per-expert K tables
            fm = lambda: launch(x, sel, rw, g_m, u_m, d_m, bsz, sc, k2g, k2u, k2d, 2.0, 2.0, 2.0, True)
            us = timed(fm, graph)
            results.append({"variant": "mixed", "bsz": bsz, "mode": mode, "us": round(us, 2),
                            "bytes": touched_m, "GBps": round(touched_m / (us * 1e-6) / 1e9, 2)})
            print(json.dumps(results[-1]), flush = True)

    json.dump(results, open("/tmp/mk/bench_coop_results.json", "w"), indent = 1)


if __name__ == "__main__":
    main()
