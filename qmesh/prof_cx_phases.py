#!/usr/bin/env python3
"""Attribute one cuda-exl3 cold-layer forward to its kernels, and price block_m at large M.

`forward` runs four kernels per bitrate group (had_in, gemm to 2*I, glu_had_in, gemm to H) plus
an fp32 fp16->fp32 conversion of the group result; at 8192 rows the layer is no longer weight
bound (every cold expert is read once per tile block), so the open question is where the 80 ms
of device time sits and whether a wider tile pays for the weights it re-reads.

Reads `cx.trace` (CUDA_EXL3_MOE_TRACE), so every number here is one call's own events on the
caller's stream.
"""

import argparse
import json
import os
import statistics

import torch

import qmesh.exl3_moe_cuda as cx_mod
from qmesh.worker_seam_server import HIDDEN, TOPK, build_layer
from qmesh.prof_layer_call import fill_inputs, mem_avail, histogram


def phase_table(cx):
    """Sum the last forward's event deltas, per op and per op family (label minus its tag).

    A mark is recorded *before* the op it names, so the interval ending at mark i is the cost of
    mark i-1's op: a delta carries the earlier mark's label.
    """
    per_label, per_family = {}, {}
    order = []
    trace = cx.trace
    for i in range(1, len(trace)):
        label, blk, nrows, ev0 = trace[i - 1]
        ms = ev0.elapsed_time(trace[i][3])
        per_label[label] = per_label.get(label, 0.0) + ms
        fam = label.split(":")[0]
        per_family[fam] = per_family.get(fam, 0.0) + ms
        order.append((label, blk, nrows, ms))
    return per_label, per_family, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/cold")
    ap.add_argument("--placement", default="/cold/placement.exl3.json")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--rows-list", default="8192")
    ap.add_argument("--block-m-list", default="0,64,128",
                    help="0 = the adapter's row-band rule, otherwise a forced tile (16/32/64/128)")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    device = torch.device("cuda:0")
    cx_mod.TRACE = True
    placement = json.load(open(args.placement))
    cold = placement["layers"][str(args.layer)]["warm"]
    from exllamav3 import Config, Model
    model = Model.from_config(Config.from_directory(args.model_dir))
    base = next(m for m in model.modules if m.key == f"layers.{args.layer}").mlp
    ffn, nc = build_layer(base, cold, device)
    del base
    torch.cuda.empty_cache()

    avail0, swap0 = mem_avail()
    cx = cx_mod.CudaExl3Moe(ffn, device)
    torch.cuda.empty_cache()
    avail1, swap1 = mem_avail()
    print(f"PROF_CXPH_LAYER layer={args.layer} n_cold={nc} groups={cx.describe()} "
          f"stacked_mib={cx.nbytes() / 2**20:.0f} hidden={cx.hidden} inter={cx.inter} top_k={cx.top_k} "
          f"avail_mib={avail0:.0f}->{avail1:.0f} swapfree_mib={swap0:.0f}->{swap1:.0f}", flush=True)
    per_expert_mib = {id(g): (g.trellis13.numel() * 2 + g.trellis2.numel() * 2) / g.n_experts / 2**20
                      for g in cx.groups}

    for rows in [int(v) for v in args.rows_list.split(",")]:
        x, ids, w = fill_inputs(rows, nc, seed=args.layer * 1000 + rows)
        mx, touched = histogram(ids, nc)
        x = x.to(device).half()
        ids = ids.to(device).long()
        w = w.to(device).half()
        print(f"PROF_CXPH_ROWS layer={args.layer} rows={rows} max_rows_per_expert={mx} "
              f"touched={touched}", flush=True)
        for bm in [int(v) for v in args.block_m_list.split(",")]:
            cx_mod.BLOCK_M_OVERRIDE = bm
            for _ in range(args.warmup):
                cx.forward(x, ids, w)
            torch.cuda.synchronize()
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            for _ in range(args.iters):
                cx.forward(x, ids, w)
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            torch.cuda.synchronize()
            wall_ms = ev.elapsed_time(end) / args.iters
            per_label, per_family, order = phase_table(cx)

            # Per-group shape: the mark before "had_in" carries the rows the tile block padded to.
            shape = []
            for label, blk, nrows, _ in order:
                if label.startswith("had_in:"):
                    g = next(g for g in cx.groups if label.endswith(f"{g.bits}{'f' if g.frac else ''}"))
                    live = int((g.remap[ids] >= 0).sum())
                    blocks = nrows // blk
                    wmib = blocks * per_expert_mib[id(g)]
                    shape.append(f"{label[7:]}:blk={blk} live={live} rows={nrows} "
                                 f"pad={nrows / max(1, live):+.0%} blocks={blocks} "
                                 f"weights_mib={wmib:.0f}")
            print(f"PROF_CXPH_BM layer={args.layer} rows={rows} block_m={bm or 'auto'} "
                  f"total_ms={wall_ms:.2f} sum_ms={sum(per_family.values()):.2f}", flush=True)
            # One line per op family, biggest first: at 8192 rows the matmuls are not the top.
            for fam, ms in sorted(per_family.items(), key=lambda kv: -kv[1]):
                print(f"PROF_CXPH_PHASE layer={args.layer} rows={rows} block_m={bm or 'auto'} "
                      f"op={fam} ms={ms:.2f} pct={100 * ms / max(wall_ms, 1e-9):.0f}", flush=True)
            for s in shape:
                print(f"PROF_CXPH_GROUP layer={args.layer} rows={rows} block_m={bm or 'auto'} {s}",
                      flush=True)
            wtotal = sum(float(s.split("weights_mib=")[1]) for s in shape)
            print(f"PROF_CXPH_BW layer={args.layer} rows={rows} block_m={bm or 'auto'} "
                  f"weights_read_mib={wtotal:.0f} "
                  f"gib_per_s={wtotal / 2**10 / (wall_ms / 1e3):.0f}", flush=True)
        del x, ids, w
        torch.cuda.empty_cache()
    cx_mod.BLOCK_M_OVERRIDE = 0
    print(f"PROF_CXPH_END avail_mib={mem_avail()[0]:.0f}", flush=True)


if __name__ == "__main__":
    main()
