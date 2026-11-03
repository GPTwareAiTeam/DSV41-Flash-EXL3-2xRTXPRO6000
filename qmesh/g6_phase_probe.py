#!/usr/bin/env python3
"""G6: what a decode-sized GX MoE call spends, split into align / four kernel stages / syncs.

The supervisor's question after the G4 A/B is not "how long is a row-6 call" (that is
`qmesh/moe_cpp_bench.py`, both arms, on the same ids) but *where* it goes on the stock path, so the
three ways to specialise the 2-group case can be priced:

  (a) one sync-free device-side alignment for both groups,
  (b) one launch per stage covering both bitrates,
  (c) dropping the split-k/acc and h_acc branches GX never takes.

Per (layer, rows) this prints, for the stock `CudaExl3Moe.forward`:

  G6PH_PHASE   rows op=<family> ms=<median per call> pct=<of wall>   -- align, had_in, gemm13, glu,
                                                                        gemm2, end (CUDA_EXL3_MOE_TRACE
                                                                        marks, same reader as
                                                                        qmesh/prof_cx_phases.py)
  G6PH_SYNC    rows per_call=<n> item=<n> tolist=<n> explicit=<n>    -- host syncs the call makes,
                                                                        counted, not inferred
  G6PH_ALIGN   rows group=<bits> us=<median standalone align>        -- alignment alone, its own
                                                                        sync-inclusive wall, so
                                                                        align_ms - align_device_us
                                                                        is the host stall in it
  G6PH_GROUP   rows group=<bits> live=<picks> pad=<x> blocks=<n> weights_mib=<m>

Measurement only; it never installs the cpp hook (`SEAM_MOE_CPP` must be unset) and it takes no
pack: like `prof_cx_phases.py` it restacks the layers named by `--layers` from /cold.

    SEAM_MOE_CPP=0 python3 qmesh/g6_phase_probe.py --layers 5 --rows-list 1,2,6 --iters 30
"""

import argparse
import json
import os
import statistics

import torch

import qmesh.exl3_moe_cuda as cx_mod
from qmesh.exl3_moe_cuda import align_block, pick_block_m
from qmesh.prof_cx_phases import phase_table
from qmesh.prof_layer_call import fill_inputs, histogram, mem_avail
from qmesh.worker_seam_server import build_layer


def api_counts(run, iters):
    """Count the CUDA API calls one forward makes, from kineto's CPU trace.

    `torch.Tensor.item` is a C-extension slot and cannot be wrapped, so the syncs are counted where
    they actually land: the CUDA API. `cudaStreamSynchronize`/`cudaDeviceSynchronize` are the ones
    `int(t)`/`int(t.sum())`/`torch.full`-with-a-device-scalar produce; `cudaMemcpy*` next to them is
    the H2D `torch.tensor([n_rows])` and the D2H of the `.item()` itself. Counts are exact; the CPU
    times under the profiler are inflated and are not reported.
    """
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            run()
        torch.cuda.synchronize()
    counts = {"sync": 0, "memcpy": 0, "launch": 0, "alloc": 0}
    for ev in prof.events():
        n = ev.name
        if n in ("cudaStreamSynchronize", "cudaDeviceSynchronize"):
            counts["sync"] += 1
        elif n.startswith("cudaMemcpy") or n.startswith("cudaMemset"):
            counts["memcpy"] += 1
        elif n in ("cudaLaunchKernel", "cuLaunchKernel"):
            counts["launch"] += 1
        elif n in ("cudaMalloc", "cudaFree"):
            counts["alloc"] += 1
    return {k: v / iters for k, v in counts.items()}


def align_once(cx, ids, grp):
    """The alignment half of forward for one group: remap -> live count -> bucket -> H2D of n_rows.

    Copied line for line from `CudaExl3Moe.forward`'s align block, including the last line: the
    per-group `torch.tensor([n_rows])` is one more H2D and belongs to the alignment's bill, so a
    specialisation that moves the alignment to the device has to pay for that too.
    """
    loc = grp.remap[ids]
    live = int((loc >= 0).sum())
    if live == 0:
        return None
    block_m = pick_block_m(live, grp.n_experts)
    sid, eids, n_rows, block_m = align_block(loc, block_m, grp.n_experts, ids.numel())
    if eids is None:
        return None
    n_rows_t = torch.tensor([n_rows], dtype=torch.int32, device=cx.device)
    return live, block_m, int(eids.numel()), sid, eids, n_rows_t


def time_align(cx, ids, grp, iters):
    """Median wall of align_once with the stream synchronised around it: the host cost, not a trace."""
    for _ in range(3):
        align_once(cx, ids, grp)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        align_once(cx, ids, grp)
        t1.record()
        torch.cuda.synchronize()
        ts.append(t0.elapsed_time(t1) * 1e3)
    return statistics.median(ts)


def align_split(fn, iters):
    """Device part of `fn` and the syncs it costs, so align's wall can be split into work vs stall.

    The alignment is torch ops (masked select, bincount, cumsum, sort) plus two `int(...)` reductions
    and one H2D of `n_rows`. On the GPU the ops are short; what the call pays for is the host waiting
    at those reductions. Timing it with events only gives their sum, so the device share is taken from
    kineto's own per-kernel times over the same iterations: wall - device is the host stall.
    """
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    # Kernel time only: `device_time` is set on the kernel events and left at 0 on the CPU-side api
    # events, which is why the sum is the GPU share of the call without needing key_averages' fields.
    dev_us = sum(getattr(ev, "device_time", 0.0) for ev in prof.events()) / iters
    sync = memcpy = 0
    for ev in prof.events():
        if ev.name in ("cudaStreamSynchronize", "cudaDeviceSynchronize"):
            sync += 1
        elif ev.name.startswith("cudaMemcpy") or ev.name.startswith("cudaMemset"):
            memcpy += 1
    return dev_us, sync / iters, memcpy / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/cold")
    ap.add_argument("--placement", default="/cold/placement.exl3.json")
    ap.add_argument("--layers", default="5")
    ap.add_argument("--rows-list", default="1,2,6")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=3)
    args = ap.parse_args()

    if os.environ.get("SEAM_MOE_CPP", "0") not in ("", "0"):
        raise SystemExit("G6PH_REFUSED SEAM_MOE_CPP is set: this probe measures the stock path")
    device = torch.device("cuda:0")
    cx_mod.TRACE = True
    placement = json.load(open(args.placement))
    from exllamav3 import Config, Model
    model = Model.from_config(Config.from_directory(args.model_dir))

    for layer in [int(v) for v in args.layers.split(",")]:
        cold = placement["layers"][str(layer)]["warm"]
        base = next(m for m in model.modules if m.key == f"layers.{layer}").mlp
        ffn, nc = build_layer(base, cold, device)
        del base
        torch.cuda.empty_cache()
        cx = cx_mod.CudaExl3Moe(ffn, device)
        torch.cuda.empty_cache()
        print(f"G6PH_LAYER layer={layer} n_cold={nc} groups={cx.describe()} "
              f"stacked_mib={cx.nbytes() / 2**20:.0f} hidden={cx.hidden} inter={cx.inter} "
              f"top_k={cx.top_k} avail_mib={mem_avail()[0]:.0f}", flush=True)
        # The group knows its own footprint; the probe does not re-derive it from the slot names.
        per_expert_mib = {id(g): g.nbytes() / max(1, g.n_experts) / 2**20 for g in cx.groups}

        for rows in [int(v) for v in args.rows_list.split(",")]:
            x, ids, w = fill_inputs(rows, nc, seed=layer * 1000 + rows)
            mx, touched = histogram(ids, nc)
            x, ids, w = x.to(device).half(), ids.to(device).long(), w.to(device).half()
            for _ in range(args.warmup):
                cx.forward(x, ids, w)
            torch.cuda.synchronize()

            # Per-call family split, one forward per sample: the trace is rebuilt per forward, so a
            # family's median across iters is a per-call cost and not a growing pile of events.
            fam_samples, lab_samples, wall_samples = {}, {}, []
            for _ in range(args.iters):
                t0 = torch.cuda.Event(enable_timing=True)
                t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                cx.forward(x, ids, w)
                t1.record()
                torch.cuda.synchronize()
                wall_samples.append(t0.elapsed_time(t1))
                per_label, per_family, order = phase_table(cx)
                for fam, ms in per_family.items():
                    fam_samples.setdefault(fam, []).append(ms)
                for lab, ms in per_label.items():
                    lab_samples.setdefault(lab, []).append(ms)
            wall = statistics.median(wall_samples)
            print(f"G6PH_WALL layer={layer} rows={rows} total_ms={wall:.3f} "
                  f"sum_ms={sum(statistics.median(v) for v in fam_samples.values()):.3f} "
                  f"max_rows_per_expert={mx} touched={touched}", flush=True)
            for fam, vs in sorted(fam_samples.items(), key=lambda kv: -statistics.median(kv[1])):
                ms = statistics.median(vs)
                print(f"G6PH_PHASE layer={layer} rows={rows} op={fam} ms={ms:.3f} "
                      f"pct={100 * ms / max(wall, 1e-9):.0f}", flush=True)
            # Per group as well as per family: the two bitrates hold different experts with different
            # trellis widths, so "align" is the same work twice while "gemm2" is not.
            for lab, vs in sorted(lab_samples.items(), key=lambda kv: str(kv[0])):
                ms = statistics.median(vs)
                print(f"G6PH_PHASE_G layer={layer} rows={rows} op={lab} ms={ms:.3f} "
                      f"pct={100 * ms / max(wall, 1e-9):.0f}", flush=True)
            # Counts, not inference: three profiled forwards, divided, after the timed samples so the
            # profiler's own overhead cannot touch the wall numbers above.
            cnt = api_counts(lambda: cx.forward(x, ids, w), 3)
            print(f"G6PH_SYNC layer={layer} rows={rows} sync_per_call={cnt['sync']:.1f} "
                  f"memcpy_per_call={cnt['memcpy']:.1f} launch_per_call={cnt['launch']:.1f} "
                  f"groups={len(cx.groups)}", flush=True)

            for grp in cx.groups:
                got = align_once(cx, ids, grp)
                if got is None:
                    continue
                live, block_m, nblk, sid, eids, n_rows_t = got
                us = time_align(cx, ids, grp, max(args.iters // 3, 5))
                dev_us, a_sync, a_memcpy = align_split(lambda: align_once(cx, ids, grp),
                                                      max(args.iters // 3, 5))
                print(f"G6PH_ALIGN layer={layer} rows={rows} group={grp.bits} "
                      f"{'frac' if grp.frac else 'mul1'} us={us:.1f} dev_us={dev_us:.1f} "
                      f"host_us={us - dev_us:.1f} sync={a_sync:.1f} memcpy={a_memcpy:.1f} "
                      f"block_m={block_m}", flush=True)
                print(f"G6PH_GROUP layer={layer} rows={rows} group={grp.bits} live={live} "
                      f"pad={int(n_rows_t.item()) / max(1, live):.2f}x blocks={nblk} "
                      f"weights_mib={nblk * per_expert_mib[id(grp)]:.0f}", flush=True)
            del x, ids, w
            torch.cuda.empty_cache()
        print(f"G6PH_END layer={layer} avail_mib={mem_avail()[0]:.0f}", flush=True)


if __name__ == "__main__":
    main()
