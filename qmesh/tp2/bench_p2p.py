"""Correctness + latency of the one-shot 2-GPU P2P collectives (exllamav3/tp2/p2p.py), two processes.

Run in exl3-tp2-p2p (both GPUs visible as cuda:0 / cuda:1):

    cd /src/exllamav3 && python3 qmesh/tp2/bench_p2p.py

Checks, per size in SIZES (fp32, (1, 5120) main case, (64, 512), (1, 32768) = 128 KB LSE payload):
  - all_reduce_: result equals rank0_input + rank1_input computed on CPU in fp32 with the same operand order,
    bit-exact on both ranks (compared bitwise, and the two ranks' outputs compared to each other);
  - all_gather: out == [rank0 | rank1] exactly;
  - bcast_i64 (root 0 and root 1): out == root's tensor exactly;
  - ITERS eager iterations with fresh random inputs each step (bit-exact);
  - graph: GRAPH_N sequential collectives captured in one CUDA graph, with a small compute op between each
    (a mul_ on the payload), replayed REPLAYS times; the replayed result is checked against the CPU reference
    computed by the same op sequence.
Latency reported as us per collective: eager (back-to-back launches, sync at end) and in-graph (replay time / N).
"""

import os
import sys
import time

import torch
import torch.multiprocessing as mp

sys.path.insert(0, "/src/exllamav3")

SIZES = [(1, 5120), (64, 512), (1, 32768)]
ITERS = 1000
GRAPH_N = 80
REPLAYS = 200
WARM = 50
NBLK = int(os.environ.get("P2P_NBLK", "1"))


def _ref_sum(a, b):
    # fixed operand order: rank0 + rank1, fp32 on CPU
    return a.float() + b.float()


def worker(rank, rdv, result_q, barrier):
    torch.cuda.set_device(rank)
    from exllamav3.tp2.p2p import P2PComm

    comm = P2PComm(rank, 2, rdv, nblk=NBLK)
    dev = torch.device("cuda", rank)
    out = {"rank": rank, "sizes": {}}
    barrier.wait()

    for shape in SIZES:
        n = shape[0] * shape[1]
        g = torch.Generator().manual_seed(1000 + n)
        base = [torch.randn(shape, generator=g) for _ in range(2)]
        res = {}

        # ---- correctness: eager all_reduce, ITERS random steps
        ok = True
        for it in range(ITERS):
            ga = torch.Generator().manual_seed(7919 * it + 1)
            xs = [torch.randn(shape, generator=ga) * (1 + it % 7) for _ in range(2)]
            mine = xs[rank].to(dev)
            comm.all_reduce_(mine)
            ref = _ref_sum(xs[0], xs[1])
            if not torch.equal(mine.cpu(), ref):
                ok = False
                res.setdefault("first_bad_iter", it)
                break
        torch.cuda.synchronize(dev)
        res["allreduce_exact"] = ok

        # ---- bit-identity across ranks: both ranks compute the same sum from the same two inputs
        fixed = base[rank].to(dev)
        comm.all_reduce_(fixed)
        torch.cuda.synchronize(dev)
        out_bits = fixed.cpu().view(torch.int32).clone()
        result_q.put(("bits", shape, rank, out_bits))
        barrier.wait()
        # ---- all_gather
        gath = torch.empty(2 * n, device=dev)
        comm.all_gather(base[rank].to(dev).flatten(), gath)
        torch.cuda.synchronize(dev)
        res["allgather_exact"] = torch.equal(gath.cpu(), torch.cat([base[0].flatten(), base[1].flatten()]))

        # ---- broadcast int64 (root 0 and root 1)
        bc_ok = True
        for root in (0, 1):
            src = torch.arange(2 * n, dtype=torch.int64) * (rank + 3) - 17
            src_t = src.to(dev)
            dst = comm.bcast_i64(src_t, root)
            torch.cuda.synchronize(dev)
            expect = (torch.arange(src.numel(), dtype=torch.int64) * (root + 3) - 17)
            bc_ok &= torch.equal(dst.cpu(), expect)
        res["bcast_i64_exact"] = bc_ok

        # ---- eager latency (kernel only, back-to-back, one stream)
        x = torch.randn(shape, device=dev)
        for _ in range(WARM):
            comm.all_reduce_(x)
        torch.cuda.synchronize(dev)
        barrier.wait()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            comm.all_reduce_(x)
        torch.cuda.synchronize(dev)
        res["eager_us"] = (time.perf_counter() - t0) / ITERS * 1e6

        # ---- graph: GRAPH_N collectives with a mul between each, one captured graph
        xg = torch.randn(shape, device=dev)
        s = torch.cuda.Stream(device=dev)
        s.wait_stream(torch.cuda.current_stream(dev))
        with torch.cuda.stream(s):
            for _ in range(3):
                for _ in range(GRAPH_N):
                    xg.mul_(0.5)
                    comm.all_reduce_(xg)
        torch.cuda.current_stream(dev).wait_stream(s)
        torch.cuda.synchronize(dev)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            for _ in range(GRAPH_N):
                xg.mul_(0.5)
                comm.all_reduce_(xg)
        torch.cuda.synchronize(dev)

        # correctness of replay: reset to known inputs, replay once, compare to CPU reference of same op chain
        ginit = [torch.randn(shape, generator=torch.Generator().manual_seed(55 + r)) for r in range(2)]
        xg.copy_(ginit[rank].to(dev))
        torch.cuda.synchronize(dev)
        barrier.wait()
        graph.replay()
        torch.cuda.synchronize(dev)
        a = ginit[0].clone()
        b = ginit[1].clone()
        for _ in range(GRAPH_N):
            a = a * 0.5
            b = b * 0.5
            s_ = a + b  # rank0 + rank1, same order on both ranks
            a = s_
            b = s_.clone()
        res["graph_exact"] = torch.equal(xg.cpu(), a)

        # graph latency (after warm replays)
        for _ in range(10):
            graph.replay()
        torch.cuda.synchronize(dev)
        barrier.wait()
        t0 = time.perf_counter()
        for _ in range(REPLAYS):
            graph.replay()
        torch.cuda.synchronize(dev)
        res["graph_us_per_collective"] = (time.perf_counter() - t0) / REPLAYS / GRAPH_N * 1e6
        out["sizes"][shape] = res
        barrier.wait()

    result_q.put(("result", rank, out))


def main():
    mp.set_start_method("spawn", force=True)
    rdv = f"/tp2_p2p_bench_{os.getpid()}"
    barrier = mp.Barrier(2)
    q = mp.Queue()
    procs = [mp.Process(target=worker, args=(r, rdv, q, barrier)) for r in range(2)]
    for p in procs:
        p.start()

    bits = {}
    results = {}
    deadline = time.monotonic() + 900
    while len(results) < 2 and time.monotonic() < deadline:
        try:
            msg = q.get(timeout=5)
        except Exception:
            continue
        if msg[0] == "bits":
            bits[(msg[1], msg[2])] = msg[3]
        elif msg[0] == "result":
            results[msg[1]] = msg[2]
    for p in procs:
        p.join(timeout=60)

    print(f"NBLK={NBLK} ITERS={ITERS} GRAPH_N={GRAPH_N} REPLAYS={REPLAYS}")
    print(f"{'shape':>12} {'AR exact':>9} {'AG exact':>9} {'BC exact':>9} {'graph exact':>12} "
          f"{'bitsEq r0r1':>12} {'eager us':>9} {'graph us':>9}")
    ok_all = True
    for shape in SIZES:
        if 0 not in results or 1 not in results:
            print("missing results; a rank failed")
            ok_all = False
            break
        r0 = results[0]["sizes"][shape]
        r1 = results[1]["sizes"][shape]
        bits_eq = torch.equal(bits[(shape, 0)], bits[(shape, 1)]) if (shape, 0) in bits and (shape, 1) in bits else False
        row = (r0["allreduce_exact"] and r1["allreduce_exact"], r0["allgather_exact"] and r1["allgather_exact"],
               r0["bcast_i64_exact"] and r1["bcast_i64_exact"], r0["graph_exact"] and r1["graph_exact"], bits_eq)
        ok_all &= all(row)
        print(f"{str(shape):>12} {str(row[0]):>9} {str(row[1]):>9} {str(row[2]):>9} {str(row[3]):>12} "
              f"{str(row[4]):>12} {r0['eager_us']:>9.2f} {r0['graph_us_per_collective']:>9.2f}")
        print(f"{'':>12} rank1 eager us {r1['eager_us']:.2f}, graph us {r1['graph_us_per_collective']:.2f}")
    for r in range(2):
        try:
            os.unlink(f"/dev/shm{rdv}.{r}")
        except FileNotFoundError:
            pass
    print("ALL_OK" if ok_all else "FAILURES")


if __name__ == "__main__":
    main()
