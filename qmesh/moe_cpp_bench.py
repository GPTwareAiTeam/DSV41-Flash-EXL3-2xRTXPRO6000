#!/usr/bin/env python3
"""Before/after for one loaded pack: the Python cold-expert forward against the C hook.

Both arms run in this process, on the same resident pack, the same layer and the same pinned
request slab, so the only difference is which code reads the slab:

  before  server._forward   -> _forward_piece: numpy .copy() out of the slab, .to(device).half(),
            int64 ids, .float() weights, align_block with two host read-backs, then
            stream.synchronize() and .float().cpu().numpy() back into the response slot.
  after   qmc_compute       -> cudaMemcpyAsync slab -> device, two conversion kernels, one
            histogram + one scatter, the same four cuda-exl3 ops, one D2H into the response slot.

Reported per row count, two lines: the BEFORE line (p50/p99 total, host share, rows/s) and the
AFTER line (same, from qmc_last_call_ns) plus the difference against the Python answer measured on
the identical inputs - rel_fro, max abs, and the fraction of elements outside 1e-3 relative.

The layer is timed once per arm per row count with the piece size fixed (ws_auto off, like
rows_bench): `--py-ws 0 --cpp-ws 0` is one whole-request piece for both, which is the honest
side-by-side; if a rung does not fit, passing 2048 splits both arms at the deployed picker's
floor rung instead. The C side takes its piece size at open, so `--cpp-ws` names it; the Python
side takes it from server.ws_rows. The C arm is timed twice, fp32 and then bf16, so the wire's
per-call cost is measured on the same buffers as the reference, not inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time

import numpy as np
import torch

from exllamav3 import Config, Model

from qmesh import moe_cpp
from qmesh.worker_seam_server import (HIDDEN, MOE_KERNEL, TOPK, SeamServer, _avail_mib,
                                      _probe_stage, _swap_used_mib, build_layer, view)


def pct(times_us, q):
    t = sorted(times_us)
    return t[min(len(t) - 1, int(len(t) * q))]


def stats(times_us):
    return (pct(times_us, 0.50), pct(times_us, 0.99), min(times_us), max(times_us),
            sum(times_us) / len(times_us))


def run_py(server, layer, rows, ptrs, iters, warmup=2):
    x_ptr, ids_ptr, w_ptr, out_ptr = ptrs
    dev = server.device
    for _ in range(warmup):
        server._forward(layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
    torch.cuda.synchronize(dev)
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    times, dev_ms = [], []
    for _ in range(iters):
        ev0.record(server.stream)
        t0 = time.perf_counter_ns()
        server._forward(layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
        torch.cuda.synchronize(dev)
        times.append((time.perf_counter_ns() - t0) / 1e3)
        ev1.record(server.stream)
        ev1.synchronize()
        dev_ms.append(ev0.elapsed_time(ev1))
    return times, dev_ms


def run_cpp(moe, layer, rows, ptrs, iters, warmup=2):
    x_ptr, ids_ptr, w_ptr, out_ptr = ptrs
    rc = moe.compute(layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
    if rc:
        raise RuntimeError(f"qmc_compute warmup rc={rc}: {moe_cpp.last_error()}")
    torch.cuda.synchronize()
    times, wall_ns, dev_ns = [], [], []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        rc = moe.compute(layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
        times.append((time.perf_counter_ns() - t0) / 1e3)
        if rc:
            raise RuntimeError(f"qmc_compute rc={rc}: {moe_cpp.last_error()}")
        w, d = moe.last_call_ns()
        wall_ns.append(w)
        dev_ns.append(d)
    return times, wall_ns, dev_ns


def compare(out_before, out_after):
    a = out_before.astype(np.float64)
    b = out_after.astype(np.float64)
    d = np.abs(a - b)
    denom = np.maximum(np.abs(a), 1e-12)
    rel = d / denom
    return (float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30)),
            float(d.max()), float(np.median(rel)),
            float(np.median(np.abs(a))), float(np.abs(a).max()),
            int((rel > 1e-3).sum()))

def fill_inputs(x_ptr, ids_ptr, rows, ncold, args, seed=0):
    """(Re)fill the pinned slab: activation samples at a chosen magnitude, and routing ids.

    The probe's all-ones activation is a synthetic case: the layer's own answer was ~8.6e-6 per
    element, three orders of magnitude below where a served request's answer sits, and how a half
    response rounds is a question about that magnitude. --x-mode normal draws N(0,1) (what a
    pre-MoE activation looks like after its layernorm) and --scale multiplies it, so the error bar
    can be read against |out| instead of against the probe's tiny numbers. --ids-mode zipf gives
    every token TOPK distinct cold experts drawn from a skewed popularity over the whole cold set,
    in score order, which is what production routing looks like and what arange%n_cold is not: the
    probe's ids hit every expert equally and in ascending order.
    """
    rng = np.random.default_rng(seed)
    if args.x_mode == "normal":
        a = rng.standard_normal((rows, HIDDEN)).astype(np.float32)
    else:
        a = np.ones((rows, HIDDEN), dtype=np.float32)
    if args.scale != 1.0:
        a *= args.scale
    view(x_ptr, rows * HIDDEN * 2, np.uint16).reshape(rows, HIDDEN)[:] = \
        torch.from_numpy(a).bfloat16().view(torch.uint16).numpy()

    if args.ids_mode == "probe":
        return None
    if args.ids_mode == "file":
        # A .npz is the all-layer form (one capture, forty layers): key L<layer>_r<rows>. A .npy
        # stays the single-layer form, so the existing one-layer command lines keep working.
        if str(args.ids_file).endswith(".npz"):
            key = f"L{args.layer}_r{rows}"
            z = np.load(args.ids_file)
            if key not in z:
                have = ",".join(sorted(z.files)[:6])
                raise SystemExit(f"--ids-file has no {key} (keys like {have}...)")
            ids = z[key]
        else:
            ids = np.load(args.ids_file)
        if ids.ndim == 1:
            ids = ids.reshape(-1, TOPK)
        ids = np.ascontiguousarray(ids.astype(np.int32))
        if ids.shape[0] != rows:
            raise SystemExit(f"--ids-file holds {ids.shape[0]} rows, --rows asks for {rows}")
        if ids.shape[1] != TOPK:
            raise SystemExit(f"--ids-file holds {ids.shape[1]} picks per row, expected {TOPK}")
        if ids.min() < -1 or ids.max() >= ncold:
            raise SystemExit(f"--ids-file ids must be in [-1, {ncold}), got "
                             f"[{ids.min()}, {ids.max()}]")
    else:
        # Gumbel top-k over a Zipf popularity: distinct per row, ordered by score like a router's
        # top-T, and skewed the way a real cold set is (a handful of experts carry most picks).
        rank = np.arange(1, ncold + 1, dtype=np.float64)
        logp = -args.zipf_s * np.log(rank)
        g = rng.gumbel(size=(rows, ncold)) + logp
        top = np.argsort(-g, axis=1)[:, :TOPK]
        ids = rng.permutation(ncold)[top].astype(np.int32)
    view(ids_ptr, rows * TOPK * 4, np.int32).reshape(rows, TOPK)[:] = ids
    hist = np.bincount(ids.reshape(-1).clip(0), minlength=ncold)
    top_n = np.sort(hist)[::-1][:8]
    print(f"SEAM_MOE_CPP_IDS rows={rows} mode={args.ids_mode} n_cold={ncold} "
          f"distinct={int((hist > 0).sum())} top8={top_n.tolist()} "
          f"max_count={int(hist.max())} mean_count={float(hist.mean()):.1f}", flush=True)
    return hist


def traffic(cx, server, layer, ids_np):
    """How many experts a call actually reads, and the bytes those experts' weights occupy.

    This is the denominator of the per-call time: a decode call (1 row, top-6) touches at most six
    experts, and whether that is 6 x 70 MiB or 6 x 15 MiB decides whether the measured 1.2 ms is a
    bandwidth floor or dispatch overhead. The expert bytes come from the group's own stacked tensors
    (EXL3 trellis + suh/svh, so the bitrate in the group name is already in the sum) rather than from
    H*I*3*2, which would overstate a 3.5-bit-packed expert by ~4.5x.
    """
    ids_t = torch.from_numpy(ids_np.astype(np.int64)).to(server.device)
    tot_e = tot_b = 0
    parts = []
    gi = -1
    for grp in cx.groups:
        if grp.n_experts == 0:
            continue
        gi += 1
        loc = grp.remap[ids_t]
        sel = loc[loc >= 0]
        e = int(torch.unique(sel).numel()) if sel.numel() else 0
        per = grp.nbytes() / max(grp.n_experts, 1)
        tot_e += e
        tot_b += int(round(e * per))
        parts.append(f"g{gi}bits{grp.bits}{'+f' if grp.frac else ''}={e}/{grp.n_experts}x"
                     f"{per / (1 << 20):.2f}MiB")
    return tot_e, tot_b, " ".join(parts)


def check_align(moe, server, layer, ffn, ptrs, rows, ids_np):
    """The C alignment against Python's align_block() on the identical ids.

    Python takes block_m from this call's live count and the C side from the previous call's, so
    the tier is frozen to Python's value for the comparison (qmc_set_block_m), which makes
    total/nblk comparable exactly; the automatic tier's stability is reported beside it. sorted_ids
    cannot be element-comparable: both sides place each pick at its expert's block through a
    cursor, so the order *within* a block is arbitrary. What must hold, and is checked, is that the
    live slots are the same slots, the expert of every live slot is the same expert, and each pick
    appears exactly once.
    """
    from qmesh.exl3_moe_cuda import align_block, pick_block_m

    x_ptr, ids_ptr, w_ptr, out_ptr = ptrs
    cx = server._cx_moe(layer, ffn)
    ids_t = torch.from_numpy(ids_np.astype(np.int64)).to(server.device)
    m_valid = rows * TOPK
    gi = -1
    for grp in cx.groups:
        if grp.n_experts == 0:
            continue
        gi += 1
        loc = grp.remap[ids_t]
        live = int((loc >= 0).sum())
        if live == 0:
            continue
        bm_ref = pick_block_m(live, grp.n_experts)
        ref_sid, ref_eids, ref_nrows, bm_ref = align_block(loc, bm_ref, grp.n_experts, m_valid)
        if ref_eids is None:
            continue
        ref_sid = ref_sid.cpu().numpy()
        ref_eids = ref_eids.cpu().numpy()
        flat = loc.reshape(-1).cpu().numpy()
        old = moe.set_block_m(bm_ref)
        rc = moe.compute(layer, rows, x_ptr, ids_ptr, w_ptr, out_ptr)
        moe.set_block_m(old)
        if rc:
            raise RuntimeError(f"qmc_compute during the alignment check rc={rc}: "
                               f"{moe_cpp.last_error()}")
        g_sid, g_eids, g_nblk, g_total, g_bm = moe.dump_align(layer, gi)
        mask_c = g_sid < m_valid
        mask_r = ref_sid < m_valid
        slots_ok = mask_c.shape == mask_r.shape and bool((mask_c == mask_r).all())
        eids_ok = g_nblk == ref_eids.shape[0] and bool((g_eids == ref_eids).all())
        picks_ok = bool(np.array_equal(np.sort(g_sid[mask_c]), np.sort(ref_sid[mask_r])))
        exp_r = flat[ref_sid[mask_r].astype(np.int64)]  # the reference's expert per live slot
        # Expert of every live slot from the C dump's own block map, so a wrong eids cannot hide
        # behind an equally wrong sorted_ids: slot -> block -> expert.
        slot_expert_c = g_eids[np.arange(g_total)[mask_c] // g_bm]
        expert_map_ok = bool((slot_expert_c == exp_r).all())
        lo = pick_block_m(max(1, int(live * 0.9)), grp.n_experts)
        hi = pick_block_m(max(1, int(live * 1.1)), grp.n_experts)
        print(f"SEAM_MOE_CPP_ALIGN rows={rows} layer={layer} g={gi} tag={grp.bits}"
              f"{'f' if grp.frac else ''} live={live} bm={g_bm}/{bm_ref} "
              f"nblk={g_nblk}/{ref_eids.shape[0]} total={g_total}/{ref_sid.size} "
              f"slots_eq={int(slots_ok)} eids_eq={int(eids_ok)} slot_expert_eq={int(expert_map_ok)} "
              f"picks_eq={int(picks_ok)} band_bm={lo}/{bm_ref}/{hi}", flush=True)


# --- all-layer driver ------------------------------------------------------------------------------
# One child process per layer. Two reasons: a layer build pulls ~3.9 GiB (stacked_mib=2576) and
# qmc_open pins its own device buffers for the life of the process, so forty layers inside one
# process is a memory cliff and not a measurement; and a fresh process is a fresh CUDA context and
# a fresh qmc stats block, so no layer can inherit another's alignment tier, block_m or fallback.
# The child is this same script with --layer pinned, so the single-layer path below is the only
# code that measures anything and the two modes cannot drift apart.
_LINE = re.compile(r"^SEAM_MOE_CPP_(AFTER|BEFORE|TRAFFIC|IDS|NULL) rows=(\d+) (.*)$")


def parse_layers(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def child_argv(layer):
    """This process's own command line with --layers dropped and --layer pinned to one layer."""
    argv, skip = [], False
    for a in sys.argv[1:]:
        if skip:
            skip = False
            continue
        if a in ("--layers", "--layer"):
            skip = True
            continue
        if a.startswith(("--layers=", "--layer=")):
            continue
        argv.append(a)
    return [sys.executable, "-u", os.path.abspath(__file__), "--layer", str(layer)] + argv


def run_all_layers(args, layers):
    rows_list = [int(x) for x in args.rows.split(",") if x.strip()]
    per, failed = {}, []
    for layer in layers:
        t0 = time.time()
        p = subprocess.run(child_argv(layer), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        got = {}
        for ln in p.stdout.splitlines():
            m = _LINE.match(ln)
            if m:
                got[(m.group(1), int(m.group(2)))] = dict(
                    kv.split("=", 1) for kv in m.group(3).split() if "=" in kv)
                # TRAFFIC is summarised into the G6L line as five rounded fields; the line itself
                # carries the per-arm device and host times, which is what a bandwidth argument
                # needs, so it goes through untouched.
                if m.group(1) == "TRAFFIC":
                    print(ln, flush=True)
            elif ln.startswith(("SEAM_MOE_CPP_OUT", "SEAM_MOE_CPP_IDS")):
                # Not summarised anywhere, and they must survive the driver: OUT/OUT_PY are the
                # per-arm tensor identity (two arms of one call are a controlled comparison only if
                # their bytes match), IDS is which experts the routing actually picked.
                print(ln, flush=True)
        if p.returncode != 0 or ("AFTER", rows_list[0]) not in got:
            failed.append(layer)
            tail = " | ".join(p.stdout.splitlines()[-6:])[-700:]
            print(f"SEAM_MOE_CPP_G6L_FAIL layer={layer} rc={p.returncode} tail={tail}", flush=True)
            continue
        for rows in rows_list:
            a, b = got.get(("AFTER", rows)), got.get(("BEFORE", rows))
            tr, ids = got.get(("TRAFFIC", rows)), got.get(("IDS", rows))
            if a is None:
                continue
            rec = {"layer": layer, "rows": rows,
                   "cpp_ms": float(a["total_ms_p50"]), "dev_ms": float(a["dev_ms"]),
                   "host_ms": float(a["host_ms"]),
                   "py_ms": float(b["total_ms_p50"]) if b else float("nan"),
                   "experts": int(tr["experts"]) if tr else -1,
                   "w_mib": float(tr["w_mib"]) if tr else float("nan"),
                   "gbs": float(tr["cpp_gbs"]) if tr else float("nan"),
                   "distinct": int(ids["distinct"]) if ids else -1}
            per[(layer, rows)] = rec
            print("SEAM_MOE_CPP_G6L layer={layer} rows={rows} cpp_ms={cpp_ms:.3f} "
                  "py_ms={py_ms:.3f} dev_ms={dev_ms:.3f} host_ms={host_ms:.3f} "
                  "experts={experts} w_mib={w_mib:.2f} gbs={gbs:.1f} distinct={distinct}"
                  .format(**rec), flush=True)
        print(f"SEAM_MOE_CPP_G6L_DONE layer={layer} wall_s={time.time() - t0:.1f}", flush=True)
    for rows in rows_list:
        recs = [per[(l, rows)] for l in layers if (l, rows) in per]
        if not recs:
            continue
        cpp = [r["cpp_ms"] for r in recs]
        py = [r["py_ms"] for r in recs if r["py_ms"] == r["py_ms"]]
        gbs = [r["gbs"] for r in recs if r["gbs"] == r["gbs"]]
        w = [r["w_mib"] for r in recs if r["w_mib"] == r["w_mib"]]
        # The decode budget is the sum over the layer stack, not a mean of the layers' means:
        # one token runs every layer once, so cpp_ms_sum is what the MoE adds to tok/s.
        print(f"SEAM_MOE_CPP_G6L_SUM rows={rows} layers={len(recs)} "
              f"cpp_ms_sum={sum(cpp):.3f} cpp_ms_mean={statistics.fmean(cpp):.4f} "
              f"cpp_ms_median={statistics.median(cpp):.4f} cpp_ms_max={max(cpp):.3f} "
              f"cpp_ms_min={min(cpp):.3f} "
              f"py_ms_sum={sum(py):.3f} py_ms_mean={statistics.fmean(py) if py else -1:.4f} "
              f"speedup_sum={sum(py) / max(sum(cpp), 1e-9):.2f} "
              f"w_mib_sum={sum(w):.1f} "
              f"gbs_mean={statistics.fmean(gbs) if gbs else -1:.1f}", flush=True)
        top = sorted(recs, key=lambda r: -r["cpp_ms"])[:8]
        print(f"SEAM_MOE_CPP_G6L_TOP rows={rows} " + " ".join(
            f"L{r['layer']}:{r['cpp_ms']:.3f}ms/{r['w_mib']:.0f}MiB" for r in top), flush=True)
    if failed:
        print(f"SEAM_MOE_CPP_G6L_FAILED layers={failed}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/cold")
    ap.add_argument("--placement", default="/cold/placement.exl3.json")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--rows", default="64,2048,8192")
    ap.add_argument("--layers", default=None,
                    help="bench several layers, one child process each (e.g. 0-39, or 0,5,39): "
                         "the per-layer line plus an all-layer sum, which is the decode budget. "
                         "Each layer is built, measured and freed in its own process, so the "
                         "resident footprint stays one layer (~3.9 GiB) instead of forty")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--py-ws", type=int, default=0,
                    help="piece size for the Python arm (0 = the whole request in one piece)")
    ap.add_argument("--cpp-ws", type=int, default=0,
                    help="piece size handed to qmc_open (0 = the whole request in one piece)")
    ap.add_argument("--skip-py", action="store_true",
                    help="C arm only (the Python numbers are already in the log)")
    ap.add_argument("--x-mode", choices=["ones", "normal"], default="ones",
                    help="activation samples in the slab: all-ones (the old probe) or N(0,1)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the activation before rounding to bf16 (magnitude sweep)")
    ap.add_argument("--ids-mode", choices=["probe", "zipf", "file"], default="probe",
                    help="probe = arange%%n_cold, zipf = skewed distinct per token, file = --ids-file")
    ap.add_argument("--ids-file", default=None, help=".npy of (rows, TOPK) int32 for --ids-mode file")
    ap.add_argument("--zipf-s", type=float, default=1.0, help="Zipf exponent of the id popularity")
    ap.add_argument("--ids-seed", type=int, default=0, help="seed of the x and id sampling")
    ap.add_argument("--dump-align", action="store_true",
                    help="compare the C alignment with align_block() (needs SEAM_MOE_CPP_DUMP=1)")
    ap.add_argument("--null-ids", action="store_true",
                    help="also time the same call with every pick at -1: no expert is read, so the "
                         "line is the call's fixed cost (dispatch, launches, barrier, response)")
    ap.add_argument("--traffic", action="store_true",
                    help="also print the experts read per call, their weight bytes and the implied GB/s "
                         "(one extra pass; the line is SEAM_MOE_CPP_TRAFFIC)")
    ap.add_argument("--dump-out", default=None,
                    help="save the C arm's fp32 output as DIR/L<layer>_r<rows>.npy and print its "
                         "sha256 and magnitude (SEAM_MOE_CPP_OUT); two builds that print the same "
                         "sha cannot differ downstream. No effect on the measured call")
    args = ap.parse_args()
    if args.layers:
        return run_all_layers(args, parse_layers(args.layers))

    dev = torch.device("cuda:0")
    t0 = time.time()
    # One layer, built the way qmesh/cx_auto_check.py builds one: the Config's modules are lazy, so
    # build_layer's deepcopy is what pulls exactly this layer out of /cold. A full SeamServer would
    # load and restack all forty, which is minutes of a ten-minute gate for one measured layer.
    model = Model.from_config(Config.from_directory(args.model_dir))
    base = next(m for m in model.modules if m.key == f"layers.{args.layer}").mlp
    placement = json.load(open(args.placement))
    ffn, ncold = build_layer(base, placement["layers"][str(args.layer)]["warm"], dev)
    server = object.__new__(SeamServer)
    server.device = dev
    server.stream = torch.cuda.Stream(device=dev)
    server.layers = {args.layer: (ffn, ncold)}
    server.stub = False
    server.ws_rows = args.py_ws
    server.ws_auto = False
    server._cx_layers = {}
    server.fail_after = 0
    server.seen = 0
    # worker_seam_server's _check_ids(layer, rows, ids, self.layer_n[layer]) guards the ids of the
    # piece and the Python arm runs through it; a stub that fills only `layers` dies there on
    # AttributeError before its first measurement (it did, 09:21:33).
    server.layer_n = {args.layer: ncold}
    print(f"SEAM_MOE_CPP_BENCH_LOAD {time.time() - t0:.1f}s kernel={MOE_KERNEL} "
          f"layers=1 layer={args.layer} n_cold={ncold} py_ws={args.py_ws} cpp_ws={args.cpp_ws} "
          f"avail_mib={_avail_mib()} swapped_mib={_swap_used_mib()}", flush=True)

    moe = moe_cpp.CppMoe(server, ws_rows=args.cpp_ws)
    moe.install(layers=[args.layer])
    cx = None
    for rows in [int(x) for x in args.rows.split(",") if x.strip()]:
        stage, x_ptr, ids_ptr, w_ptr, out_ptr = _probe_stage(rows, ncold)
        ptrs = (x_ptr, ids_ptr, w_ptr, out_ptr)
        print(f"SEAM_MOE_CPP_BENCH_HEAD layer={args.layer} rows={rows} n_cold={ncold} "
              f"iters={args.iters} x_mode={args.x_mode} scale={args.scale} "
              f"ids_mode={args.ids_mode} avail_mib={_avail_mib()} swapped_mib={_swap_used_mib()}",
              flush=True)
        hist = fill_inputs(x_ptr, ids_ptr, rows, ncold, args, seed=args.ids_seed)
        ids_np = view(ids_ptr, rows * TOPK * 4, np.int32).reshape(rows, TOPK).copy()

        if args.dump_align:
            check_align(moe, server, args.layer, ffn, ptrs, rows, ids_np)
        before = None
        py_dev_ms = None
        py_host_ms = None
        py_p50 = None
        if not args.skip_py:
            times, dev_ms = run_py(server, args.layer, rows, ptrs, args.iters)
            p50, p99, mn, mx, mean = stats(times)
            py_p50 = p50
            py_dev_ms = sum(dev_ms) / len(dev_ms)
            py_host_ms = p50 / 1e3 - py_dev_ms
            before = view(out_ptr, rows * HIDDEN * 4, np.float32).reshape(rows, HIDDEN).copy()
            torch.cuda.empty_cache()
            print(f"SEAM_MOE_CPP_BEFORE rows={rows} total_ms_p50={p50 / 1e3:.3f} "
                  f"total_ms_p99={p99 / 1e3:.3f} total_ms_min={mn / 1e3:.3f} "
                  f"total_ms_max={mx / 1e3:.3f} dev_ms={py_dev_ms:.3f} "
                  f"host_ms={p50 / 1e3 - py_dev_ms:.3f} us_per_row={p50 / rows:.2f} "
                  f"rows_per_s={rows * 1e6 / p50:.0f} avail_mib={_avail_mib()} "
                  f"swapped_mib={_swap_used_mib()}", flush=True)
            # The python (exllamav3) path is the one a change inside the extension can move, so it
            # gets the same byte-for-byte treatment as the hook: --dump-out saves both answers and
            # two builds of _C.so are then compared by sha, not by their error against each other.
            if args.dump_out:
                os.makedirs(args.dump_out, exist_ok=True)
                path = os.path.join(args.dump_out, f"L{args.layer}_r{rows}_py.npy")
                np.save(path, before)
                b = np.ascontiguousarray(before).tobytes()
                print(f"SEAM_MOE_CPP_OUT_PY layer={args.layer} rows={rows} "
                      f"sha={hashlib.sha256(b).hexdigest()[:16]} bytes={len(b)} "
                      f"sum={float(before.astype(np.float64).sum()):.6e} "
                      f"absmax={float(np.abs(before).max()):.6e} path={path}", flush=True)

        # The fp32 arm is the reference the AFTER line's error is measured on, whatever the process
        # was started with; the bf16 arm sets its own mode and the finally restores this one.
        mode0 = moe.set_out_mode(0)
        times, wall_ns, dev_ns = run_cpp(moe, args.layer, rows, ptrs, args.iters)
        p50, p99, mn, mx, mean = stats(times)
        cpp_dev_ms = sum(dev_ns) / len(dev_ns) / 1e6
        cpp_host_ms = sum(wall_ns) / len(wall_ns) / 1e6 - cpp_dev_ms
        after = view(out_ptr, rows * HIDDEN * 4, np.float32).reshape(rows, HIDDEN).copy()
        err = ""
        if before is not None:
            rel_fro, max_abs, med_rel, med_abs, max_abs_val, n_gt = compare(before, after)
            err = (f" rel_fro={rel_fro:.3e} max_abs={max_abs:.3e} med_rel={med_rel:.3e} "
                   f"med_abs={med_abs:.3e} max_abs_val={max_abs_val:.3e} n_gt_1e-3={n_gt}")
        speed = f" speedup={py_p50 / max(p50, 1e-9):.2f}" if py_p50 else ""
        print(f"SEAM_MOE_CPP_AFTER rows={rows} total_ms_p50={p50 / 1e3:.3f} "
              f"total_ms_p99={p99 / 1e3:.3f} total_ms_min={mn / 1e3:.3f} "
              f"total_ms_max={mx / 1e3:.3f} dev_ms={cpp_dev_ms:.3f} "
              f"host_ms={cpp_host_ms:.3f} us_per_row={p50 / rows:.2f} "
              f"rows_per_s={rows * 1e6 / p50:.0f} fmt=fp32 cpp_ws={args.cpp_ws}{speed}"
              f" avail_mib={_avail_mib()} swapped_mib={_swap_used_mib()}{err}"
              # The layer goes last so nothing that already parses this line has to change: two
              # runs of the same call (bound vs readback, guard vs base) are only comparable per
              # layer, and one bench process per layer means the line itself has to say which.
              f" layer={args.layer}", flush=True)
        # Identity across two builds of the extension (the split-k guard): the fp32 tensor this
        # process just read back is the whole of the layer's answer, so equal bytes here mean no
        # downstream token can differ, at any batch size, without a further argument.
        if args.dump_out:
            os.makedirs(args.dump_out, exist_ok=True)
            path = os.path.join(args.dump_out, f"L{args.layer}_r{rows}.npy")
            np.save(path, after)
            b = np.ascontiguousarray(after).tobytes()
            print(f"SEAM_MOE_CPP_OUT layer={args.layer} rows={rows} "
                  f"sha={hashlib.sha256(b).hexdigest()[:16]} bytes={len(b)} "
                  f"sum={float(after.astype(np.float64).sum()):.6e} "
                  f"absmax={float(np.abs(after).max()):.6e} path={path}", flush=True)
        # What a call actually reads, so its time has a denominator: at 1-6 rows the experts behind
        # the picks decide whether a ~1.2 ms call is a bandwidth floor or dispatch overhead. Taken
        # after the C arm, so the dev_ms/host_ms beside it came from this same routing.
        if args.traffic:
            if cx is None:
                cx = server._cx_moe(args.layer, ffn)
            e, b, detail = traffic(cx, server, args.layer, ids_np)
            print(f"SEAM_MOE_CPP_TRAFFIC rows={rows} experts={e} w_mib={b / (1 << 20):.2f} "
                  f"cpp_dev_ms={cpp_dev_ms:.3f} cpp_gbs={b / max(cpp_dev_ms, 1e-9) / 1e6:.1f} "
                  f"py_dev_ms={-1.0 if py_dev_ms is None else py_dev_ms:.3f} "
                  f"py_host_ms={-1.0 if py_host_ms is None else py_host_ms:.3f} "
                  f"cpp_host_ms={cpp_host_ms:.3f} {detail}", flush=True)

        # Fixed cost of one call, measured instead of inferred: the same call with every pick
        # pointing at -1 touches no expert, so it pays the dispatch, the group launches, the
        # barrier and the response store and nothing else. full - null is what the weights cost,
        # which is the whole question at decode sizes (6 experts x 10.28 MiB ~= 0.23 ms of a
        # measured 1.71 ms call). The routing on the slab is restored before the next arm.
        if args.null_ids:
            try:
                view(ids_ptr, rows * TOPK * 4, np.int32).reshape(rows, TOPK)[:] = -1
                times, wall_ns, dev_ns = run_cpp(moe, args.layer, rows, ptrs, args.iters)
                pn, _, _, _, _ = stats(times)
                null_dev_ms = sum(dev_ns) / len(dev_ns) / 1e6
                null_host_ms = sum(wall_ns) / len(wall_ns) / 1e6 - null_dev_ms
                print(f"SEAM_MOE_CPP_NULL rows={rows} experts=0 total_ms_p50={pn / 1e3:.3f} "
                      f"dev_ms={null_dev_ms:.3f} host_ms={null_host_ms:.3f} "
                      f"full_dev_ms={cpp_dev_ms:.3f} "
                      f"full_minus_null_ms={cpp_dev_ms - null_dev_ms:.3f}", flush=True)
            except Exception as e:  # noqa: BLE001 - an ablation that fails must not kill the run
                print(f"SEAM_MOE_CPP_NULL_FAILED rows={rows} err={type(e).__name__}: "
                      f"{str(e).splitlines()[0][:160]}", flush=True)
            finally:
                view(ids_ptr, rows * TOPK * 4, np.int32).reshape(rows, TOPK)[:] = ids_np
        # The response is 5120 floats per row, two thirds of a served request's payload. This is the
        # same call with the answer rounded on device, so it is the bf16 wire's per-call cost and
        # its error against the Python path's fp32 answer, measured on the identical inputs.
        ref16 = None
        try:
            moe.set_out_mode(1)
            times, wall_ns, dev_ns = run_cpp(moe, args.layer, rows, ptrs, args.iters)
            p16, p99_16, mn16, mx16, _ = stats(times)
            dev16_ms = sum(dev_ns) / len(dev_ns) / 1e6
            host16_ms = sum(wall_ns) / len(wall_ns) / 1e6 - dev16_ms
            ref16 = view(out_ptr, rows * HIDDEN * 2, np.uint16).reshape(rows, HIDDEN).copy()
            got = ref16.view(np.uint16).astype(np.uint32) << 16
            got = got.view(np.float32).reshape(rows, HIDDEN)
            err16 = ""
            if before is not None:
                r, ma, _, _, _, ngt = compare(before, got)
                err16 = f" rel_fro_vs_py={r:.3e} max_abs={ma:.3e} n_gt_1e-3={ngt}"
            print(f"SEAM_MOE_CPP_AFTER16 rows={rows} total_ms_p50={p16 / 1e3:.3f} "
                  f"total_ms_p99={p99_16 / 1e3:.3f} total_ms_min={mn16 / 1e3:.3f} "
                  f"total_ms_max={mx16 / 1e3:.3f} dev_ms={dev16_ms:.3f} "
                  f"host_ms={host16_ms:.3f} us_per_row={p16 / rows:.2f} "
                  f"d2h_bytes={rows * HIDDEN * 2} fmt=bf16 cpp_ws={args.cpp_ws}{err16}",
                  flush=True)
            # fp16 wire, same call. Both conversions are round-to-nearest, so the bytes must be the
            # ones torch would write for the same fp32 answer - that is what makes the error bars
            # below the wire's error bar and not an extra rounding on top of them.
            moe.set_out_mode(2)
            times, wall_ns, dev_ns = run_cpp(moe, args.layer, rows, ptrs, args.iters)
            ph, p99_h, mnh, mxh, _ = stats(times)
            devh_ms = sum(dev_ns) / len(dev_ns) / 1e6
            hosth_ms = sum(wall_ns) / len(wall_ns) / 1e6 - devh_ms
            refh = view(out_ptr, rows * HIDDEN * 2, np.uint16).reshape(rows, HIDDEN).copy()
            errh = ""
            if before is not None:
                want = torch.from_numpy(before).half().numpy().view(np.uint16)
                same = float((refh == want).mean()) * 100.0
                gh = refh.view(np.uint16).astype(np.uint32) << 16
                r, ma, _, _, _, ngt = compare(before, gh.view(np.float32).reshape(rows, HIDDEN))
                n_inf = int((~np.isfinite(gh.view(np.float32))).sum())
                errh = (f" rel_fro_vs_py={r:.3e} max_abs={ma:.3e} n_gt_1e-3={ngt} "
                        f"bytes_eq_torch={same:.2f}% n_inf={n_inf}")
            print(f"SEAM_MOE_CPP_AFTERH rows={rows} total_ms_p50={ph / 1e3:.3f} "
                  f"total_ms_p99={p99_h / 1e3:.3f} total_ms_min={mnh / 1e3:.3f} "
                  f"total_ms_max={mxh / 1e3:.3f} dev_ms={devh_ms:.3f} "
                  f"host_ms={hosth_ms:.3f} us_per_row={ph / rows:.2f} "
                  f"d2h_bytes={rows * HIDDEN * 2} fmt=fp16 cpp_ws={args.cpp_ws}{errh}",
                  flush=True)
            moe.set_out_mode(mode0)
        finally:
            moe.set_out_mode(mode0)
        # Error bars of both half formats, taken from the Python arm's fp32 answer so they do not
        # depend on which format the C arm just wrote.
        if before is not None:
            tref = torch.from_numpy(before)
            den = max(float(np.linalg.norm(before)), 1e-30)
            for name, cast in (("bf16", tref.bfloat16()), ("fp16", tref.half())):
                d = cast.float().numpy() - before
                # fp16's ceiling is 65504; a real activation never reaches it, and a probe that
                # does would report an infinite error bar instead of a rounding one.
                n_inf = int((~torch.isfinite(cast.float())).sum()) if name == "fp16" else 0
                print(f"SEAM_MOE_CPP_OUT_ERR rows={rows} fmt={name} "
                      f"rel_fro={float(np.linalg.norm(d)) / den:.3e} "
                      f"max_abs={float(np.abs(d).max()):.3e} "
                      f"max_rel={float((np.abs(d) / np.maximum(np.abs(before), 1e-30)).max()):.3e} "
                      f"max_val={float(np.abs(before).max()):.3e} n_inf={n_inf} "
                      f"x_mode={args.x_mode} scale={args.scale} ids_mode={args.ids_mode} "
                      f"resp_bytes_row={HIDDEN * 2} fp32_bytes_row={HIDDEN * 4}", flush=True)
        del stage, before, after, ref16
        torch.cuda.empty_cache()

    calls, refused = moe.stats()
    print(f"SEAM_MOE_CPP_STATS calls={calls} refused={refused}", flush=True)
    moe.close()


if __name__ == "__main__":
    main()
