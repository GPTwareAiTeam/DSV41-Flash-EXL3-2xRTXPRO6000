"""
TP=2 rank-0 fused MoE decode microbenchmark (real checkpoint slices, one GPU).

Shape per GPU: H = 5120, per-rank I = 1152 (routed and shared experts split on the intermediate width, per
TP2_DESIGN.md shard rules), 384 experts, top-6 routed + 1 shared expert (weight 1.0, its own tables).
Slices (rank 0, read straight from the checkpoint with safetensors, no full-expert load):
  w1/w3 (gate/up): trellis[:, 0:72, :] (output tiles, first 1152 of 2304 columns), svh[0:1152], suh full
  w2   (down):     trellis[0:72, :, :] (input tiles, first 1152 of 2304 rows), suh[0:1152], svh full,
                   bias only on rank 0 (none in this checkpoint, so no bias tables)
The shared expert is run as its own one-expert launch (the current production path, BC sh_coop); the
optimized kernel folds it into the routed launches, so both variants are timed on the same inputs.

Usage: python3 qmesh/tp2/bench_moe_tp2.py [--layers 3,20] [--bsz 1,2,6,8] [--out FILE]
Writes one JSON line per (layer, bsz, mode) and a correctness record per (layer, bsz).
"""
import argparse
import json
import os
import statistics
import time

import torch
from safetensors import safe_open

from exllamav3.ext import exllamav3_ext as ext
from testlib.moe import ptr_table

CKPT = "/models/dsv41/exl3-out/DeepSeek-V4.1-Flash-1017-EXL3-mixed-2.565-tp2"
H, I_FULL, E, TOPK = 5120, 2304, 384, 6
I = I_FULL // 2                    # 1152 per rank
TI = I // 16                       # 72 tiles of 16 per rank (out tiles for w1/w3, in tiles for w2)
TOUT = I_FULL // 16                # 144
DEV = torch.device("cuda:0")
ACT_SILU = 0
ITERS = int(os.environ.get("BENCH_ITERS", "200"))
WARM = 20


def _index():
    return json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]

DEV = torch.device(os.environ.get("BENCH_DEV", "cuda:1"))
_IDX = None
_HANDLES = {}


def _get(name):
    """Tensor by checkpoint key (lazy safe_open per shard, CPU)"""
    global _IDX
    if _IDX is None:
        _IDX = _index()
    fn = _IDX[name]
    if fn not in _HANDLES:
        _HANDLES[fn] = safe_open(os.path.join(CKPT, fn), framework = "pt", device = "cpu")
    return _HANDLES[fn].get_tensor(name)


def _trellis_k(t: torch.Tensor) -> float:
    """Bits per weight of a trellis tensor [in/16, out/16, 16K]"""
    k = t.shape[-1] / 16
    return k


def load_proj(prefix: str, which: str):
    """Rank-0 slice of one projection. which: 'gu' (w1/w3, output split) or 'd' (w2, input split).
    Returns (trellis, suh, svh, K)"""
    tr = _get(prefix + ".trellis")
    su = _get(prefix + ".suh")
    sv = _get(prefix + ".svh")
    if which == "gu":
        tr = tr[:, :TI, :].contiguous()
        sv = sv[:I].contiguous()
    else:
        tr = tr[:TI, :, :].contiguous()
        su = su[:I].contiguous()
    return tr, su, sv, _trellis_k(tr)


def load_layer(layer: int, experts: list[int]):
    """Expert-set dict {g,u,d: [(trellis, suh, svh)]} on DEV for the chosen expert ids (+ shared as last)"""
    out = {"g": [], "u": [], "d": [], "K": {"g": [], "u": [], "d": []}}
    for e in experts:
        base = f"layers.{layer}.ffn.experts.{e}" if e >= 0 else f"layers.{layer}.ffn.shared_experts"
        for p, name, kind in (("g", "w1", "gu"), ("u", "w3", "gu"), ("d", "w2", "d")):
            tr, su, sv, K = load_proj(f"{base}.{name}", kind)
            out[p].append((tr.to(DEV), su.to(DEV), sv.to(DEV)))
            out["K"][p].append(K)
    return out


def scratch(smax):
    d = {}
    d["gu_g"] = torch.empty((smax, 1, I), dtype = torch.half, device = DEV)
    d["gu_u"] = torch.empty((smax, 1, I), dtype = torch.half, device = DEV)
    d["act_out"] = torch.empty((smax, 1, I), dtype = torch.half, device = DEV)
    d["d_out"] = torch.empty((smax, 1, H), dtype = torch.float, device = DEV)
    d["out"] = torch.full((8, H), float("nan"), dtype = torch.float, device = DEV)
    d["ctr"] = torch.zeros((smax * (I // 128) + 8 * (H // 128) + 2 * smax + 3,), dtype = torch.int, device = DEV)
    d["had_g"] = torch.empty((smax, H), dtype = torch.half, device = DEV)
    d["had_u"] = torch.empty_like(d["had_g"])
    return d


def k2_of(Kvec):
    return torch.tensor([int(round(2 * k)) for k in Kvec], dtype = torch.uint8, device = DEV)


def tables(ex):
    return {p: tuple(ptr_table([t[i] for t in ex[p]], DEV) for i in range(3)) for p in ("g", "u", "d")}


def run_routed(x, sel, rw, tb, sc, mixed, Kvec, out, k2=None):
    """One routed launch over E experts (sel indices in [0, E)); k2 tables prebuilt (no host copy in capture)"""
    k2 = k2 or {}
    ext.exl3_moe_coop(
        x, sel, rw, -1, -1, H,
        *tb["g"], *tb["u"], *tb["d"],
        None, None, None,
        Kvec["g"][0] if not mixed else 3.0, Kvec["u"][0] if not mixed else 3.0, Kvec["d"][0] if not mixed else 3.0,
        False, True, ACT_SILU, 0.0, True,
        sc["had_g"], sc["had_u"], sc["gu_g"], sc["gu_u"], sc["act_out"], sc["d_out"], sc["ctr"], out,
        None, None,
        k2["g"] if mixed else None, k2["u"] if mixed else None, k2["d"] if mixed else None,
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


def ref_out(x, sel, rw, ex, Kvec):
    """fp32 dense reference of rank-0 partial output: sum_k rw * down(silu(gate)*up) over routed picks
    plus the shared expert (index E, weight 1). Uses the same EXL3 reconstruct path as testlib.moe"""
    from testlib.moe import exl3_linear_ref
    bsz = x.shape[0]
    res = torch.zeros((bsz, H), dtype = torch.float, device = DEV)
    for b in range(bsz):
        for k in range(TOPK):
            e = int(sel[b, k])
            g = exl3_linear_ref(x[b:b + 1], *ex["g"][e], Kvec["g"][e])
            u = exl3_linear_ref(x[b:b + 1], *ex["u"][e], Kvec["u"][e])
            a = (torch.nn.functional.silu(g.float()) * u.float()).half()
            d = exl3_linear_ref(a, *ex["d"][e], Kvec["d"][e]).float()
            res[b] += float(rw[b, k]) * d[0]
        g = exl3_linear_ref(x[b:b + 1], *ex["g"][E], Kvec["g"][E])
        u = exl3_linear_ref(x[b:b + 1], *ex["u"][E], Kvec["u"][E])
        a = (torch.nn.functional.silu(g.float()) * u.float()).half()
        res[b] += exl3_linear_ref(a, *ex["d"][E], Kvec["d"][E]).float()[0]
    return res


def main():
    torch.cuda.set_device(DEV)
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default = "3,20")
    ap.add_argument("--bsz", default = "1,2,6,8")
    ap.add_argument("--out", default = "")
    ap.add_argument("--no-graph", action = "store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    dump = open(args.out, "a") if args.out else None

    def emit(rec):
        line = json.dumps(rec)
        print(line, flush = True)
        if dump:
            dump.write(line + "\n"); dump.flush()

    for layer in [int(x) for x in args.layers.split(",")]:
        # Real expert set: every routed expert of the layer (rank-0 slice) + shared (index E)
        ex = load_layer(layer, list(range(E)) + [-1])
        Kvec = {p: ex["K"][p] for p in ("g", "u", "d")}
        tb = tables(ex)
        mixed = len(set(Kvec["g"])) > 1 or len(set(Kvec["d"])) > 1
        for bsz in [int(x) for x in args.bsz.split(",")]:
            sc = scratch(256)
            g = torch.Generator().manual_seed(1000 + bsz)
            # Top-6 routed picks over the routed experts only (shared is handled separately)
            sel = torch.stack([torch.randperm(E, generator = g)[:TOPK] for _ in range(bsz)]).to(DEV)
            rw = torch.rand((bsz, TOPK), generator = g).mul(0.9).add(0.1).half().to(DEV)
            x = (torch.randn((bsz, H), generator = g) * 0.05).half().to(DEV)
            # Shared expert as one-expert launch (production path): sel 0 -> index E, weight 1
            sh_tb = {p: tuple(ptr_table([ex[p][E][i]], DEV) for i in range(3)) for p in ("g", "u", "d")}
            sh_sel = torch.zeros((bsz, 1), dtype = torch.long, device = DEV)
            sh_rw = torch.ones((bsz, 1), dtype = torch.half, device = DEV)
            sh_out = torch.full((8, H), float("nan"), dtype = torch.float, device = DEV)
            sh_sc = scratch(256)
            sh_K = {p: [Kvec[p][E]] for p in ("g", "u", "d")}

            k2 = {p: k2_of(Kvec[p]) for p in ("g", "u", "d")} if mixed else None

            def routed():
                run_routed(x, sel, rw, tb, sc, mixed, Kvec, sc["out"][:bsz], k2)

            def shared():
                run_routed(x, sh_sel, sh_rw, {p: sh_tb[p] for p in sh_tb}, sh_sc, False, sh_K, sh_out[:bsz])

            # correctness: routed vs dense reference (rank-0 partial, routed experts only)
            routed()
            torch.cuda.synchronize()
            got = sc["out"][:bsz].clone()
            ref = ref_out(x, sel, rw, ex, Kvec)
            # reference above includes the shared expert; subtract it to compare the routed launch
            shared_ref = torch.zeros_like(ref)
            for b in range(bsz):
                shared_ref[b] = exl3_shared(x[b:b + 1], ex, Kvec)
            routed_ref = ref - shared_ref
            err = (got - routed_ref).abs().max().item() / (routed_ref.abs().max().item() + 1e-9)
            emit({"kind": "correct", "layer": layer, "bsz": bsz, "mixed": mixed, "routed_rel_err": err})

            for mode in ("eager",) if args.no_graph else ("eager", "graph"):
                g_mode = mode == "graph"
                t_r = timed(routed, g_mode)
                t_s = timed(shared, g_mode)
                emit({"kind": "time", "layer": layer, "bsz": bsz, "mode": mode, "mixed": mixed,
                      "routed_us": round(t_r, 2), "shared_us": round(t_s, 2), "total_us": round(t_r + t_s, 2)})
    if dump:
        dump.close()


def exl3_shared(x1, ex, Kvec):
    from testlib.moe import exl3_linear_ref
    g = exl3_linear_ref(x1, *ex["g"][E], Kvec["g"][E])
    u = exl3_linear_ref(x1, *ex["u"][E], Kvec["u"][E])
    a = (torch.nn.functional.silu(g.float()) * u.float()).half()
    return exl3_linear_ref(a, *ex["d"][E], Kvec["d"][E]).float()[0]


if __name__ == "__main__":
    main()
