# P2PAllReduce: 2-GPU one-shot collectives (TP=2)

Branch `tp2/p2p-allreduce`, commit `ee9aaaba` (transport c). Status: transport works, latency NOT measured.

## Transport finding
- Peer SM load/store between GPU0 and GPU1 is non-functional in this container, though `nvidia-smi topo -p2p` reports OK.
  - Kernel on cuda:0 reading cuda:1 memory returns `0xbad0fb00`; a kernel write into cuda:1 is not visible.
  - cudaIpc (two processes) and cuda VMM fd-passing both return `0xbad0fb00`.
  - cudaMemcpy D2D (copy engine) reports rc=0.
- Host cause (Main): IOMMU domain type DMA-FQ (translation on) for both GPU groups, and BAR1 = 256 MiB (ReBAR off).
  Fix needs BIOS ReBAR + kernel `iommu=pt` + reboot. Owner decision, pending.
- Pinned host memory (`cudaHostAllocMapped`) is valid from both GPUs: cuda:0 and cuda:1 read `0xbeef`; a cuda:1 write is visible on host (`0x1234`).

## Implemented
- Transport (c): each rank creates `/tp2_p2p_<tag>.<rank>` (POSIX shm, `cudaHostRegister` mapped + portable), and maps the peer's inbox. The kernel writes into the peer's inbox and spins on its own. `P2PComm(rank, 2, shm_name, nblk, device)`; methods `all_reduce_`, `all_gather`, `bcast_i64`, `barrier` are unchanged.
- Two-process smoke test (`/tmp/smoke_shm.py`, container): rank0 = 1.0, rank1 = 2.0, all-reduce gives 3.0 on both ranks, exit 0 on both.
- Transport (d) copy-engine: NOT implemented.

## Not measured
- Microbench `qmesh/tp2/bench_p2p.py` (sizes `(1,5120)`, `(64,512)`, `(1,32768)`) has not produced output. Two runs timed out at 900 s with `error: interrupted`; the log is empty. Cause not yet diagnosed (possibly a hang in the spin protocol or an extension JIT lock; `TORCH_EXTENSIONS_DIR` was set to `/root/.cache/torch_extensions` on the second run).
- No latency table. No (c) vs (d) comparison. The ≤5 µs in-graph target for 20 KB is unverified.

## Caveats
- Two ranks only; payload cap 256 KiB per parity slot; nblk 1–8.
- The shm transport goes through host memory over PCIe; expect latency well above the ≤5 µs target. Confirm with a measurement.
- Interface is unchanged so the runner can swap transports after the BIOS/iommu fix (if ReBAR and `iommu=pt` restore SM P2P, the cudaIpc path would have to be restored; the current code has no cudaIpc).
