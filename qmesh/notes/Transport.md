# Transport: native P7 <-> GX1 expert seam (measured)

Source: `qmesh/native/` (`qmesh_expert_seam.h`, `qmesh_expert_seam.cu`, `qes_bench.cu`, `build.sh`).
Binaries: `qmesh/native/bin/qes_bench.x86_64` (P7), `bin/qes_bench.aarch64` (GX1).

## Design (as built)
- Host-staged RC RDMA verbs over `rocep142s0f0` (P7, GID 5 = `::ffff:10.0.100.1` RoCEv2) and
  `rocep1s0f0` (GX1, GID 3 = `::ffff:10.0.100.2` RoCEv2). GPUDirect RDMA is not used: nvidia-peermem is absent on P7
  and `ibv_reg_mr` on CUDA memory fails on GX1, as in the prior art.
- Memory: one `cudaHostAllocMapped|Portable` slab per process, 40 layers. Each layer has request, response, and two
  64 B flag lines (request seq, response seq). All regions are 64 B aligned.
- Client (P7): GPU D2H copies the request (bf16 x, int32 ids, bf16 w) into the slab, then a 1-thread publish kernel
  bumps the request seq with `st.release.sys`. A CPU proxy sees the seq, posts an RDMA WRITE of the payload, then an
  inline 8 B RDMA WRITE of the seq to the server's request flag.
- Server (GX1): proxy sees the request seq, runs the compute stub (`out = float(x) + (L+1)`, fp32), RDMA-writes the
  response, then the inline response-seq flag.
- Collect: 1-thread kernel spins on `ld.acquire.sys` of the response seq (30 s timeout, then `__trap`), then H2D copies
  the fp32 response to the device buffer.
- Both kernels and `cudaMemcpyAsync` are stream work, so issue/collect are capturable in CUDA graphs.

## Commands
```
# build (both hosts, native, not in containers)
cd /home/treadall/qmesh-exl3-runtime/qmesh/native && ./build.sh
# GX1 server (pidfile so it can be stopped safely)
ssh 10.0.100.2 'cd /home/treadall/qmesh-exl3-runtime/qmesh/native && setsid nohup sh -c "echo \$\$ > /tmp/qes_server.pid; exec ./bin/qes_bench.aarch64 server rocep1s0f0 3 47311 <ROWS>" </dev/null >/tmp/qes_server.log 2>&1 &'
# P7 client (GPU0, tiny footprint)
CUDA_VISIBLE_DEVICES=0 ./bin/qes_bench.x86_64 client rocep142s0f0 5 10.0.100.2 47311 <ROWS> 200 0
```
`<ROWS>`, the bucket list and the slab are fixed per server, so a different shape needs a restart;
a different client does not -- the server now takes one session after another (refusing or queuing a
peer that arrives while one is live) and keeps its handle for the next one.

Two switches the C-hook work added:
```
# what the server's compute hook costs per request, whichever hook is installed: one call site, so
# the Python cold path and the C hook are measured identically (stderr, one line per served request)
SEAM_HOOK_TIME=1 <server>           # -> SEAM_HOOK_US port=47311 layer=38 rows=1024 us=1450.0
# -arch=native asks the driver for the GPU in this container; the aarch64 build for a GX box has no
# device to ask (and a busy box must not be asked), so the arch is nameable from outside
NVCC_ARCH=sm_121 ./build.sh         # seam bench + libqmesh_expert_seam
CUDA_ARCH=sm_121 ./build_moe.sh     # cuda-exl3 kernels for qmc_compute
```

## Results (median, us; n = iterations; all four modes report correctness OK)

| rows | eager per layer p50/p99 | graph 40-layer per layer p50/p99 | pipelined (issue all, collect all) p50/p99 | busy_pipe per layer p50/p99 |
|---|---|---|---|---|
| 1  | 30.56 / 38.30 | 22.95 / 24.65 | 15.74 / 17.30 | 62.00 / 71.97 |
| 6  | 42.98 / 52.22 | 39.86 / 47.82 | 19.08 / 20.81 | 61.00 / 62.69 |
| 48 | 216.10 / 229.20 | 214.11 / 215.74 | 88.65 / 89.16 | 214.43 / 216.68 |

Per-iteration totals (40 layers): rows 1 pipelined 630 us, graph 918 us; rows 48 pipelined 3546 us, graph 8565 us.
Source: `qes_bench client` output, 200 iterations, run after the final build.

## Interpretation
- Proven (measured): 40-layer sequential round trip at rows=1 costs ~23 us/layer in a graph, and ~16 us/layer when
  layers are pipelined (issue all, then collect). That is ~0.6 ms per token of transport, well under 6.67 ms.
- Proven: at rows=48 the transport is size-bound. Request is 480 KiB bf16, response is 960 KiB fp32, and per-layer time
  scales roughly linearly with bytes (~215 us graph, ~89 us pipelined). At 150 tok/s with bsz 1 the rows=1 path is the
  relevant one, not rows=48.
- Inferred: the rows=1 and rows=6 floors (~15-23 us) are dominated by the per-WR CPU proxy hop and the two MMIO
  flag writes, not by wire time. Not profiled.
- Caveat on busy_pipe: the 100 us busy kernel uses `clock64()` cycles = 100000, which is not calibrated to 100 us. Its
  actual duration was not measured, so busy_pipe shows overlap qualitatively (per-layer time ~= busy kernel duration,
  not busy + RTT), but the "100 us" figure is [INFERENCE]. Calibrate before quoting overlap numbers.
- Caveat on eager: each layer includes a `cudaStreamSynchronize`, so it is a host-timed latency, not steady-state throughput.

## Correctness
Each run prints one `correctness: OK|FAIL` line. The check compares collected fp32 outputs against `float(x) + (L+1)` with absolute tolerance 1e-3 (`qes_bench.cu:57`). The single printed status covers the modes that ran. All runs printed `correctness: OK`.

## Not done
- Real expert compute on GX1 (the server is the stub).
- Python wrapper (`issue`/`collect` as torch ops). The C ABI is in `qmesh_expert_seam.h` for the wrapper.
- Integration with `remote_host.py`. It is unchanged.
- Multi-WR payload chunking. Chained WRs failed on this bnxt_re firmware in prior art. Each payload here is one WR of
  up to 960 KiB (rows=48 response), which works on MTU 1024 per the measurements above.

## GPU and memory footprint
P7 GPU0 use is the bench context plus tiny device buffers. Pinned host slab is ~ stride x 40 (rows=48: ~60 MiB). Both
are far below the 2 GiB cap. [INFERENCE from allocation sizes; GPU memory was not sampled during the run.]

## Real-compute seam (Transport2)

Server: `qmesh/worker_seam_server.py` in GX1 `exl3-fork`, one fused mixed-K `BlockSparseMLP` per layer over the warm
list (259-299 cold experts/layer), 40 layers. Load: 223.5-223.7 s (cold pack, host memory flat ~7.5 GiB used).
Client: P7 `qes_bench client rocep142s0f0 5 10.0.100.2 47311 <ROWS> 200 0` with CUDA_VISIBLE_DEVICES=0.

### GX1 per-layer compute (no seam, CUDA events, `qmesh/bench_seam_layer.py`, layer 0, 259 cold, 200 iters)
- rows=1: p50 358.2 us, p99 362.5 us, min 351.5 us
- rows=6: p50 2008.9 us, p99 2052.2 us, min 1998.2 us

### P7 round trip with real compute (40 layers, per-layer us)
- rows=1: eager p50 575.2 / p99 1346.6; graph p50 546.2 / p99 1064.7; pipelined p50 525.2 / p99 531.1
- rows=6: eager p50 2068.2 / p99 2201.7; graph p50 2075.5 / p99 2542.9; pipelined p50 2034.5 / p99 2055.7
- Per-layer round trip is ~170 us above compute at rows=1 and ~25-60 us above at rows=6 (eager). Seam overhead
  is the gap between these numbers. [INFERENCE from the two measurements above, not a separate timing.]

### Correctness: NOT MEASURED
- `qes_bench` reports `correctness: FAIL` on real-compute runs. This is expected: its check compares against the stub
  output `x+(L+1)`, not the expert output. It is not a numerical result.
- The P7-side reference for one layer could not be run independently. P7 host has no torch. The `vllm-moet` image
  runs torch on GPU but cannot import the fork's exllamav3 extension (JIT build fails there). The fork's
  `exl3_linear_ref` is the only reference path, and it is the same kernel code the server runs, so it does not test
  the module independently. No relative error is reported.

### Bugs fixed in this slice
- `build_layer` (`worker_seam_server.py`) had dropped `ffn.load(device)`. The module was never loaded, so the server
  returned `None` for every layer and the seam returned zeros. Fixed; the 47 s load was that broken run, and the real
  load is 223 s.
- `bench_seam_layer.py` unpacked `build_layer`'s `(ffn, n)` tuple incorrectly on the first attempt; fixed.

### Files
- `qmesh/bench_seam_layer.py` (new): per-layer CUDA-event timing.
- `qmesh/worker_seam_server.py`: `build_layer` restores `ffn.load(device)`.
