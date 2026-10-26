"""Correctness and latency of the one-shot 2-GPU P2P collectives (exllamav3/tp2/p2p.py) vs NCCL, two processes.

Run from the repo root (both GPUs visible as cuda:0 / cuda:1):

    python3 qmesh/tp2/bench_p2p.py

Sizes (fp32 payload): 20 KB = (1, 5120), 128 KB = (1, 32768).

Per size, per rank:
  - ITERS eager iterations with fresh inputs each step, each checked bit-exact against a CPU fp32 reference:
      all_reduce_ (rank0 + rank1), all_gather ([rank0 | rank1]), bcast_i64 (root 0 and root 1, same byte count).
    A sha256 digest of every all_reduce_ result is sent to the parent; the parent requires both ranks' digest
    sequences to be identical (bit-identity across ranks).
  - CUDA graph: GRAPH_N sequential all_reduce_ ops, each preceded by a small compute op (mul_ 0.5), captured once.
    GRAPH_CHECKS replays with reset inputs are checked bit-exact against a CPU reference of the same op chain.
  - Latency, us per collective:
      p2p_eager_1   : ITERS back-to-back all_reduce_ launches, no compute, sync at end
      p2p_eager_seq : GRAPH_N-op sequence (mul + all_reduce_) run eagerly, REPLAYS times
      p2p_graph     : same GRAPH_N-op sequence as one graph, REPLAYS replays
      nccl_*        : identical structure with torch.distributed NCCL all_reduce on the same tensors

Timeouts and diagnostics:
  - Every worker phase is reported by the parent as it starts. Each worker arms faulthandler.dump_traceback_later
    for PHASE_S seconds per phase, so a hang dumps all thread stacks and exits.
  - Setup and phase boundaries use mp.Barrier with timeout; a stuck peer breaks the barrier instead of hanging.
  - The parent enforces RUN_TIMEOUT_S overall, terminates stragglers, and prints the last phase each rank reached.
"""

import faulthandler
import hashlib
import os
import queue
import socket
import sys
import time
import shutil
import tempfile
import traceback
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from exllamav3.tp2.p2p import P2PComm  # noqa: E402

SIZES = [("20KB", (1, 5120)), ("128KB", (1, 32768))]
ITERS = 1000
GRAPH_N = 80
GRAPH_CHECKS = 10
REPLAYS = 200
WARM = 50
NBLK = int(os.environ.get("P2P_NBLK", "8"))
PHASE_S = 600  # per-phase hang watchdog in each worker
BARRIER_S = 600  # mp.Barrier timeout
RUN_TIMEOUT_S = 3600  # parent overall deadline


def _log(rank, msg):
    print(f"[rank{rank} {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _inputs(shape, it, n):
    # deterministic on both ranks: identical CPU tensors generated in each process
    g = torch.Generator().manual_seed(7919 * it + 1 + n)
    return [torch.randn(shape, generator=g) * (1 + it % 7) for _ in range(2)]


def _ref_chain(i0, i1):
    # Same op chain as the graph: each rank halves its state, then all_reduce leaves the sum on both ranks.
    a = i0.clone()
    b = i1.clone()
    for _ in range(GRAPH_N):
        a = a * 0.5
        b = b * 0.5
        s = a + b
        a = s
        b = s.clone()
    return a


def _seq(step, x):
    for _ in range(GRAPH_N):
        x.mul_(0.5)
        step(x)


def _check_eager(comm, rank, dev, shape, n):
    """Returns (ok flags dict, list of sha256 digests of all_reduce results, first failure description)."""
    flags = {"ar": True, "ag": True, "bc": True}
    digests = []
    first_bad = None
    m = n // 2  # int64 payload with the same byte count as the fp32 payload
    for it in range(ITERS):
        xs = _inputs(shape, it, n)
        mine = xs[rank].to(dev)
        comm.all_reduce_(mine)
        got = mine.cpu()
        ref = xs[0] + xs[1]
        if not torch.equal(got, ref):
            flags["ar"] = False
            first_bad = first_bad or f"all_reduce it={it}"
        digests.append(hashlib.sha256(got.contiguous().numpy().tobytes()).hexdigest())

        gath = torch.empty(2 * n, device=dev)
        comm.all_gather(xs[rank].to(dev).flatten(), gath)
        if not torch.equal(gath.cpu(), torch.cat([xs[0].flatten(), xs[1].flatten()])):
            flags["ag"] = False
            first_bad = first_bad or f"all_gather it={it}"

        for root in (0, 1):
            src = torch.arange(m, dtype=torch.int64) * (rank + 3) - 17 + it
            dst = comm.bcast_i64(src.to(dev), root)
            expect = torch.arange(m, dtype=torch.int64) * (root + 3) - 17 + it
            if not torch.equal(dst.cpu(), expect):
                flags["bc"] = False
                first_bad = first_bad or f"bcast_i64 root={root} it={it}"
        if it % 200 == 0:
            _log(rank, f"{shape} eager check it={it} flags={flags}")
    torch.cuda.synchronize(dev)
    return flags, digests, first_bad


def _capture(step, x, dev):
    s = torch.cuda.Stream(device=dev)
    s.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(s):
        for _ in range(3):
            _seq(step, x)
    torch.cuda.current_stream(dev).wait_stream(s)
    torch.cuda.synchronize(dev)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        _seq(step, x)
    torch.cuda.synchronize(dev)
    return graph


def _check_graph(graph, xg, rank, dev, shape, n):
    ok = True
    first_bad = None
    for k in range(GRAPH_CHECKS):
        init = [torch.randn(shape, generator=torch.Generator().manual_seed(55 + 1000 * k + r + n)) for r in range(2)]
        xg.copy_(init[rank].to(dev))
        torch.cuda.synchronize(dev)
        graph.replay()
        torch.cuda.synchronize(dev)
        if not torch.equal(xg.cpu(), _ref_chain(init[0], init[1])):
            ok = False
            first_bad = first_bad or f"graph check k={k}"
    return ok, first_bad


def _time_lat(step, x, dev, barrier, graph):
    out = {}
    # eager single collective, back-to-back
    for _ in range(WARM):
        step(x)
    torch.cuda.synchronize(dev)
    barrier.wait(timeout=BARRIER_S)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        step(x)
    torch.cuda.synchronize(dev)
    out["eager_1"] = (time.perf_counter() - t0) / ITERS * 1e6

    # eager sequence with compute, same ops as the graph
    _seq(step, x)
    torch.cuda.synchronize(dev)
    barrier.wait(timeout=BARRIER_S)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        _seq(step, x)
    torch.cuda.synchronize(dev)
    out["eager_seq"] = (time.perf_counter() - t0) / (REPLAYS * GRAPH_N) * 1e6

    # graph replay of the same sequence
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize(dev)
    barrier.wait(timeout=BARRIER_S)
    t0 = time.perf_counter()
    for _ in range(REPLAYS):
        graph.replay()
    torch.cuda.synchronize(dev)
    out["graph"] = (time.perf_counter() - t0) / (REPLAYS * GRAPH_N) * 1e6
    return out


def _arm(rank, phase, q):
    q.put(("phase", rank, phase, time.time()))
    _log(rank, f"PHASE {phase}")
    faulthandler.dump_traceback_later(PHASE_S, exit=True)


def worker(rank, rdv, nccl_port, q, barrier):
    try:
        _arm(rank, "p2p_construct", q)
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
        comm = P2PComm(rank, world=2, rendezvous_path=rdv, nblk=NBLK)
        _arm(rank, "p2p_construct_barrier", q)
        barrier.wait(timeout=BARRIER_S)

        _arm(rank, "nccl_init", q)
        dist.init_process_group(
            "nccl",
            init_method=f"tcp://127.0.0.1:{nccl_port}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=PHASE_S),
        )
        _arm(rank, "nccl_init_barrier", q)
        barrier.wait(timeout=BARRIER_S)

        results = {}
        for name, shape in SIZES:
            n = shape[0] * shape[1]
            res = {}

            _arm(rank, f"{name} eager correctness", q)
            flags, digests, bad = _check_eager(comm, rank, dev, shape, n)
            res.update(flags)
            res["first_bad"] = bad
            q.put(("digest", name, rank, digests))
            _log(rank, f"{name} eager correctness done {flags} first_bad={bad}")
            barrier.wait(timeout=BARRIER_S)

            _arm(rank, f"{name} p2p graph capture+check", q)
            xg = torch.randn(shape, device=dev)
            step_p2p = comm.all_reduce_
            graph_p2p = _capture(step_p2p, xg, dev)
            res["graph_exact"], g_bad = _check_graph(graph_p2p, xg, rank, dev, shape, n)
            res["graph_first_bad"] = g_bad
            barrier.wait(timeout=BARRIER_S)

            _arm(rank, f"{name} p2p latency", q)
            xl = torch.randn(shape, device=dev)
            lat = _time_lat(step_p2p, xl, dev, barrier, graph_p2p)
            res["p2p"] = lat
            del graph_p2p
            barrier.wait(timeout=BARRIER_S)

            _arm(rank, f"{name} nccl latency", q)
            def step_nccl(t):
                dist.all_reduce(t)
            xn = torch.randn(shape, device=dev)
            graph_nccl = _capture(step_nccl, xn, dev)
            barrier.wait(timeout=BARRIER_S)
            res["nccl"] = _time_lat(step_nccl, xn, dev, barrier, graph_nccl)
            del graph_nccl
            _log(rank, f"{name} done: {res}")
            results[name] = res
            barrier.wait(timeout=BARRIER_S)

        faulthandler.cancel_dump_traceback_later()
        q.put(("result", rank, results))
        dist.destroy_process_group()
    except BaseException:
        faulthandler.cancel_dump_traceback_later()
        q.put(("error", rank, traceback.format_exc()))
        raise


def _cleanup(rdv):
    shutil.rmtree(rdv, ignore_errors=True)


def main():
    mp.set_start_method("spawn", force=True)
    rdv = tempfile.mkdtemp(prefix="tp2_p2p_bench_")
    nccl_port = _free_port()
    barrier = mp.Barrier(2)
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(r, rdv, nccl_port, q, barrier), name=f"rank{r}") for r in range(2)]
    for p in procs:
        p.start()
    print(f"launched ranks rdv={rdv} nccl_port={nccl_port} NBLK={NBLK} ITERS={ITERS} GRAPH_N={GRAPH_N}", flush=True)

    phases = {0: "start", 1: "start"}
    results = {}
    digests = {}
    errors = []
    deadline = time.monotonic() + RUN_TIMEOUT_S
    timed_out = False
    while len(results) < 2 and not errors:
        if time.monotonic() > deadline:
            timed_out = True
            break
        try:
            msg = q.get(timeout=2)
        except queue.Empty:
            dead = [r for r, p in enumerate(procs) if p.exitcode not in (None, 0) and r not in results]
            if dead:
                errors.append((dead[0], f"rank{dead[0]} exited with code {procs[dead[0]].exitcode} without result"))
            continue
        kind = msg[0]
        if kind == "phase":
            phases[msg[1]] = msg[2]
            print(f"[main] rank{msg[1]} -> {msg[2]}", flush=True)
        elif kind == "digest":
            digests[(msg[1], msg[2])] = msg[3]
        elif kind == "result":
            results[msg[1]] = msg[2]
        elif kind == "error":
            errors.append((msg[1], msg[2]))

    for p in procs:
        if p.is_alive():
            p.terminate()
    for p in procs:
        p.join(timeout=10)
        if p.is_alive():
            p.kill()
            p.join()
    _cleanup(rdv)

    if timed_out:
        print(f"TIMEOUT after {RUN_TIMEOUT_S}s; last phases: {phases}")
    for r, tb in errors:
        print(f"ERROR rank{r}:\n{tb}")
    print(f"last phases: {phases}")

    if len(results) < 2:
        print("FAILURES (missing results)")
        return

    print(f"NBLK={NBLK} ITERS={ITERS} GRAPH_N={GRAPH_N} GRAPH_CHECKS={GRAPH_CHECKS} REPLAYS={REPLAYS}")
    hdr = (f"{'size':>6} {'AR':>5} {'AG':>5} {'BC':>5} {'digestEq':>9} {'graphOK':>8} "
           f"{'p2p_e1':>9} {'p2p_seq':>9} {'p2p_gr':>9} {'nccl_e1':>9} {'nccl_seq':>9} {'nccl_gr':>9}")
    print(hdr)
    ok_all = True
    for name, _ in SIZES:
        r0 = results[0][name]
        r1 = results[1][name]
        ar = r0["ar"] and r1["ar"]
        ag = r0["ag"] and r1["ag"]
        bc = r0["bc"] and r1["bc"]
        dig_eq = digests.get((name, 0)) is not None and digests.get((name, 0)) == digests.get((name, 1))
        gok = r0["graph_exact"] and r1["graph_exact"]
        ok_all &= ar and ag and bc and dig_eq and gok
        p = r0["p2p"]
        nc = r0["nccl"]
        print(f"{name:>6} {str(ar):>5} {str(ag):>5} {str(bc):>5} {str(dig_eq):>9} {str(gok):>8} "
              f"{p['eager_1']:>9.2f} {p['eager_seq']:>9.2f} {p['graph']:>9.2f} "
              f"{nc['eager_1']:>9.2f} {nc['eager_seq']:>9.2f} {nc['graph']:>9.2f}")
        for r, rr in ((0, r0), (1, r1)):
            if rr["first_bad"] or rr["graph_first_bad"]:
                print(f"  rank{r} first_bad={rr['first_bad']} graph_first_bad={rr['graph_first_bad']}")
    print("ALL_OK" if ok_all and not errors and not timed_out else "FAILURES")


if __name__ == "__main__":
    main()
