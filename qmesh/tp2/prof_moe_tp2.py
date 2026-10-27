"""Per-kernel device time of the folded TP=2 MoE decode (kernel A, kernel B, rotation).

Reuses bench_moe_tp2's loaders and routed launch (shared expert folded as table index E, weight 1)
and reports torch.profiler CUDA kernel totals per call, so kernel A and kernel B are attributed
separately. Usage: PROF_LAYER=3 PROF_BSZ=1 PROF_ITERS=50 python3 qmesh/tp2/prof_moe_tp2.py
"""
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_moe_tp2 as b  # noqa: E402


def main():
    torch.cuda.set_device(b.DEV)
    layer = int(os.environ.get("PROF_LAYER", "3"))
    bsz = int(os.environ.get("PROF_BSZ", "1"))
    iters = int(os.environ.get("PROF_ITERS", "50"))

    ex = b.load_layer(layer, list(range(b.E)) + [-1])
    Kvec = {p: ex["K"][p] for p in ("g", "u", "d")}
    tb = b.tables(ex)
    mixed = len(set(Kvec["g"])) > 1 or len(set(Kvec["d"])) > 1
    k2 = {p: b.k2_of(Kvec[p]) for p in ("g", "u", "d")} if mixed else None

    g = torch.Generator().manual_seed(1000 + bsz)
    sel = torch.stack([torch.randperm(b.E, generator = g)[:b.TOPK] for _ in range(bsz)]).to(b.DEV)
    rw = torch.rand((bsz, b.TOPK), generator = g).mul(0.9).add(0.1).half().to(b.DEV)
    x = (torch.randn((bsz, b.H), generator = g) * 0.05).half().to(b.DEV)
    fsel = torch.cat([sel, torch.full((bsz, 1), b.E, dtype = torch.long, device = b.DEV)], 1).contiguous()
    frw = torch.cat([rw, torch.ones((bsz, 1), dtype = torch.half, device = b.DEV)], 1).contiguous()
    fsc = b.scratch(256)

    def folded():
        b.run_routed(x, fsel, frw, tb, fsc, mixed, Kvec, fsc["out"][:bsz], k2)

    if os.environ.get("PROF_NCU"):
        # Single bounded launch for Nsight Compute: no warmup, no torch.profiler
        folded()
        torch.cuda.synchronize()
        return
    for _ in range(10):
        folded()
    torch.cuda.synchronize()
    with profile(activities = [ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            folded()
        torch.cuda.synchronize()

    rows = []
    for ev in prof.key_averages():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.count:
            rows.append((ev.self_device_time_total / ev.count, ev.count, ev.key))
    rows.sort(reverse = True)
    print(f"layer={layer} bsz={bsz} mixed={mixed} (device us per call, per kernel)")
    for us, n, name in rows[:8]:
        print(f"  {us:9.2f} us  x{n // iters if iters else n}  {name[:110]}")
    # Effective bandwidth: trellis bytes of the distinct experts the launch touches (routed + folded shared)
    used = sorted(set(sel.flatten().tolist()) | {b.E})
    nbytes = sum(ex[p][e][0].numel() * ex[p][e][0].element_size() for p in ("g", "u", "d") for e in used)
    tot = sum(us for us, _, name in rows if "_a_" in name or "_b_" in name)
    print(f"  experts={len(used)} trellis_bytes={nbytes / 1e6:.2f} MB  kernel_sum={tot:.2f} us  "
          f"eff={nbytes / (tot * 1e3):.0f} GB/s")


if __name__ == "__main__":
    main()
