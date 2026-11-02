#!/usr/bin/env python3
"""Split one GX cold-expert call into host Python, kernels and gaps. Measurement only.

Run inside the GX container, against the resident cold pack:

    EXL3_MOE_TIER_TRACE=1 python3 -m qmesh.prof_layer_call --layers 5 --rows-list 512,2048

Four numbers per (layer, rows), all on the production call shape:

  call_ms   SeamServer.compute's own body on pinned slab buffers (numpy copy -> .to(device) ->
            .half() -> ffn.forward on the side stream -> .cpu().numpy() into the response half).
            This is the GX side of a seam round trip, wire excluded.
  ffn_ms    the same ffn.forward with everything already on the device. call_ms - ffn_ms is the
            copy/host glue the server adds around the module.
  dev_ms    sum of self device time of every kernel and memcpy inside ffn.forward, from
            torch.profiler's CUDA activity. ffn_ms - dev_ms is host Python (argsort, counts,
            expert_count.tolist() host sync, tier planning, buffer zeroing) plus launch gaps.
  top       the kernels that make dev_ms, largest first, so the cost has a name.
"""
import argparse
import json
import os
import statistics
import threading
import time

import numpy as np
import torch

from exllamav3 import Config, Model
import qmesh.worker_seam_server as wss
from qmesh.worker_seam_server import HIDDEN, TOPK, SeamServer, build_layer, view


def pinned(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


def npv(t: torch.Tensor, dtype, shape):
    return t.numpy().view(dtype).reshape(shape)


def fill_inputs(rows, nc, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, HIDDEN, generator=g) * 0.02).to(torch.bfloat16)
    ids = torch.stack([torch.randperm(nc, generator=g)[:TOPK] for _ in range(rows)]).to(torch.int32)
    w = torch.rand(rows, TOPK, generator=g)
    w = (w / w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
    return x, ids, w


def histogram(ids, nc):
    c = np.bincount(ids.reshape(-1).numpy(), minlength=nc)
    return int(c.max()), int((c > 0).sum())


def make_server(ffn, layer, device, moe_kernel="mixedk"):
    """A SeamServer with only the attributes _forward/_forward_piece touch.

    The incumbent is the default here: the cuda-exl3 phase restacks layer `layer` in place and
    drops the per-expert copies, after which `ffn.forward` cannot run any more.
    """
    wss.MOE_KERNEL = moe_kernel
    srv = object.__new__(SeamServer)
    srv.device = device
    srv.stream = torch.cuda.Stream(device=device)
    srv.layers = {layer: (ffn, ffn.num_experts)}
    srv.stub = False
    srv.ws_rows = 0
    srv.fail_after = 0
    srv.seen = 0
    srv.compute_lock = threading.Lock()
    return srv


def time_call(srv, layer, rows, xb, ib, wb, ob, iters, warmup):
    for _ in range(warmup):
        srv.compute(layer, rows, xb.data_ptr(), ib.data_ptr(), wb.data_ptr(), ob.data_ptr())
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        srv.compute(layer, rows, xb.data_ptr(), ib.data_ptr(), wb.data_ptr(), ob.data_ptr())
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def time_ffn(ffn, rows, x, ids, w, iters, warmup):
    xd = x.to("cuda").half().view(1, rows, HIDDEN)
    sel = ids.to("cuda").long()
    gw = w.to("cuda").half()
    params = {"external_selected": sel, "external_weights": gw}
    for _ in range(warmup):
        ffn.forward(xd, params)
    torch.cuda.synchronize()
    ts, dev = [], []
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        t0 = time.perf_counter()
        ffn.forward(xd, params)
        e1.record()
        e1.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
        dev.append(e0.elapsed_time(e1))
    return ts, dev


def profile_ffn(ffn, rows, x, ids, w, iters):
    from torch.profiler import ProfilerActivity, profile
    xd = x.to("cuda").half().view(1, rows, HIDDEN)
    params = {"external_selected": ids.to("cuda").long(), "external_weights": w.to("cuda").half()}
    for _ in range(3):
        ffn.forward(xd, params)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            ffn.forward(xd, params)
        torch.cuda.synchronize()
    dev, host = [], []
    for ev in prof.key_averages():
        if ev.self_device_time_total > 0:
            dev.append((ev.key, ev.self_device_time_total / iters))
        if ev.self_cpu_time_total > 0:
            host.append((ev.key, ev.self_cpu_time_total / iters))
    dev.sort(key=lambda kv: -kv[1])
    host.sort(key=lambda kv: -kv[1])
    return dev, host




def mem_avail():
    """MemAvailable and SwapFree in MiB: the box is one unified pool, so the cold pack and the
    scratch come out of the same number a decoder sees."""
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            vals[k] = int(v.split()[0]) / 1024.0
    return vals["MemAvailable"], vals["SwapFree"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/cold")
    ap.add_argument("--placement", default="/cold/placement.exl3.json")
    ap.add_argument("--layers", default="5")
    ap.add_argument("--rows-list", default="512,2048")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--prof-iters", type=int, default=5)
    ap.add_argument("--cx", action="store_true",
                    help="after the incumbent measurements, restack the layer for cuda-exl3 "
                         "(qmesh.exl3_moe_cuda) and measure that path against the stored answers")
    ap.add_argument("--cx-block-m", type=int, default=0,
                    help="force cuda-exl3's tile block_m (16/32/64/128) instead of the cost "
                         "model, to price a wider tile against the weights it re-reads")
    args = ap.parse_args()
    if args.cx_block_m:
        import qmesh.exl3_moe_cuda as cx_mod
        cx_mod.BLOCK_M_OVERRIDE = args.cx_block_m
    device = torch.device("cuda:0")
    placement = json.load(open(args.placement))
    layers = [int(v) for v in args.layers.split(",")]
    rows_list = [int(v) for v in args.rows_list.split(",")]

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)

    for layer in layers:
        cold = placement["layers"][str(layer)]["warm"]
        base = next(m for m in model.modules if m.key == f"layers.{layer}").mlp
        ffn, nc = build_layer(base, cold, device)
        del base
        torch.cuda.empty_cache()
        pack = os.path.join(args.model_dir, f"cold-L{layer:02d}.safetensors")
        pmib = os.path.getsize(pack) / 2**20 if os.path.exists(pack) else 0.0
        avail, swap = mem_avail()
        print(f"PROF layer={layer} n_cold={nc} pack_mib={pmib:.0f} alloc_gib="
              f"{torch.cuda.memory_allocated(device) / 2**30:.2f} "
              f"avail_mib={avail:.0f} swapfree_mib={swap:.0f}", flush=True)

        refs = {}
        for rows in rows_list:
            x, ids, w = fill_inputs(rows, nc, seed=layer * 1000 + rows)
            mx, touched = histogram(ids, nc)
            xb, ib, wb, ob = (pinned(rows * HIDDEN * 2), pinned(rows * TOPK * 4),
                              pinned(rows * TOPK * 2), pinned(rows * HIDDEN * 4))
            npv(xb, np.uint16, (rows, HIDDEN))[:] = x.view(torch.uint16).numpy()
            npv(ib, np.int32, (rows, TOPK))[:] = ids.numpy()
            npv(wb, np.uint16, (rows, TOPK))[:] = w.view(torch.uint16).numpy()

            srv = make_server(ffn, layer, device)
            call = time_call(srv, layer, rows, xb, ib, wb, ob, args.iters, args.warmup)
            ffn_t, ev_t = time_ffn(ffn, rows, x, ids, w, args.iters, args.warmup)

            # Sanity: the slab path and the device path must agree bit for bit.
            srv.compute(layer, rows, xb.data_ptr(), ib.data_ptr(), wb.data_ptr(), ob.data_ptr())
            ref = ffn.forward(x.to("cuda").half().view(1, rows, HIDDEN),
                              {"external_selected": ids.to("cuda").long(),
                               "external_weights": w.to("cuda").half()}).view(rows, HIDDEN)
            got = npv(ob, np.float32, (rows, HIDDEN))
            err = float(np.abs(got - ref.float().cpu().numpy()).max())
            if args.cx:
                # Kept for the cuda-exl3 phase below, which runs after the incumbent tensors are
                # gone: the reference has to be captured while the mixed-K module can still run.
                refs[rows] = (x, ids, w,
                              ref.float().cpu().numpy().astype(np.float32).copy())
                del ref

            if args.prof_iters > 0:
                dev, host = profile_ffn(ffn, rows, x, ids, w, args.prof_iters)
            else:
                dev, host = [], []
            dev_sum = sum(v for _, v in dev) / 1e3
            call_p50 = statistics.median(call)
            ffn_p50 = statistics.median(ffn_t)
            print(f"PROF_ROWS layer={layer} rows={rows} nc={nc} max_rows_per_expert={mx} "
                  f"touched={touched} call_ms={call_p50:.2f} call_p50_ms={call_p50:.2f} "
                  f"ffn_ms={ffn_p50:.2f} ffn_event_ms={statistics.median(ev_t):.3f} "
                  f"dev_ms={dev_sum:.2f} host_gap_ms={ffn_p50 - dev_sum:.2f} "
                  f"glue_ms={call_p50 - ffn_p50:.2f} maxabs_vs_ref={err:.3e}", flush=True)
            for name, us in dev[:8]:
                print(f"PROF_DEV rows={rows} us={us:.1f} pct={100 * us / (dev_sum * 1e3):.1f} {name[:110]}",
                      flush=True)
            for name, us in host[:8]:
                print(f"PROF_HOST rows={rows} us={us:.1f} {name[:110]}", flush=True)
            del srv, xb, ib, wb, ob

        if args.cx:
            # Same layer through cuda-exl3: restack per bitrate (this drops the per-expert copies),
            # then measure the production call shape and the device-only time per row bucket.
            from qmesh.exl3_moe_cuda import CudaExl3Moe
            wss.MOE_KERNEL = "cudaexl3"
            srv = make_server(ffn, layer, device, moe_kernel = "cudaexl3")
            avail0, swap0 = mem_avail()
            cx = CudaExl3Moe(ffn, device)
            torch.cuda.empty_cache()
            srv._cx_layers = {layer: cx}
            avail1, swap1 = mem_avail()
            print(f"PROF_CX layer={layer} groups={cx.describe()} "
                  f"stacked_mib={cx.nbytes() / 2**20:.0f} "
                  f"avail_mib={avail0:.0f}->{avail1:.0f} swapfree_mib={swap0:.0f}->{swap1:.0f} "
                  f"alloc_gib={torch.cuda.memory_allocated(device) / 2**30:.2f}", flush=True)
            for rows in rows_list:
                x, ids, w, ref = refs[rows]
                xb, ib, wb, ob = (pinned(rows * HIDDEN * 2), pinned(rows * TOPK * 4),
                                  pinned(rows * TOPK * 2), pinned(rows * HIDDEN * 4))
                npv(xb, np.uint16, (rows, HIDDEN))[:] = x.view(torch.uint16).numpy()
                npv(ib, np.int32, (rows, TOPK))[:] = ids.numpy()
                npv(wb, np.uint16, (rows, TOPK))[:] = w.view(torch.uint16).numpy()

                a0, s0 = mem_avail()
                call = time_call(srv, layer, rows, xb, ib, wb, ob, args.iters, args.warmup)
                a1, s1 = mem_avail()
                srv.compute(layer, rows, xb.data_ptr(), ib.data_ptr(), wb.data_ptr(), ob.data_ptr())
                got = npv(ob, np.float32, (rows, HIDDEN))
                d = got - ref
                err = float(np.abs(d).max())
                ref_l2 = float(np.linalg.norm(ref))
                d_l2 = float(np.linalg.norm(d))
                rel = d_l2 / ref_l2
                # Where the difference lives: a real mapping bug concentrates it in a few rows,
                # fp16 noise on a cancelling sum spreads it out.
                per_row = np.abs(d).max(1)
                rms = float(np.sqrt((ref.astype(np.float64) ** 2).mean()))
                rel_rows = np.linalg.norm(d, axis=1) / np.maximum(
                    np.linalg.norm(ref, axis=1), 1e-30)
                k = max(1, rows // 100)
                top = np.sort(np.linalg.norm(d, axis=1))[::-1][:k]
                conc = float((top ** 2).sum() / max(d_l2 ** 2, 1e-30))

                # Device-only time and the transient workspace of the bare module call.
                xd = x.to(device).half()
                sel = ids.to(device).long()
                gw = w.to(device).half()
                for _ in range(args.warmup):
                    cx.forward(xd, sel, gw)
                torch.cuda.synchronize()
                ev0 = torch.cuda.Event(enable_timing=True)
                ev1 = torch.cuda.Event(enable_timing=True)
                ev0.record()
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    cx.forward(xd, sel, gw)
                ev1.record()
                ev1.synchronize()
                cx_ms = ev0.elapsed_time(ev1) / args.iters
                host_ms = 1e3 * (time.perf_counter() - t0) / args.iters
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                base = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                out = cx.forward(xd, sel, gw)
                torch.cuda.synchronize()
                ws = torch.cuda.max_memory_allocated(device) - base
                del out, got
                print(f"PROF_CX_ROWS layer={layer} rows={rows} "
                      f"call_ms={statistics.median(call):.2f} cx_ms={cx_ms:.2f} "
                      f"cx_host_ms={host_ms:.2f} ws_mib={ws / 2**20:.1f} "
                      f"ws_kib_per_row={ws / 2**10 / rows:.1f} "
                      f"maxabs_vs_ref={err:.3e} rel_fro_vs_ref={rel:.3e} "
                      f"ref_rms={rms:.3e} err_rms={d_l2 / np.sqrt(d.size):.3e} "
                      f"row_rel_p99={float(np.percentile(rel_rows, 99)):.3e} "
                      f"err_share_top1pct_rows={conc:.2f} "
                      f"avail_mib={a0:.0f}->{a1:.0f} "
                      f"swapfree_mib={s0:.0f}->{s1:.0f}", flush=True)
                del xb, ib, wb, ob, xd, sel, gw
            del srv, cx
            wss.MOE_KERNEL = "mixedk"
            refs.clear()
        del ffn
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
