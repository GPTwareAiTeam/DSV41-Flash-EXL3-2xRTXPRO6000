# P2PAllReduce: 2-GPU one-shot collectives (TP=2)

Branch `tp2/p2p-allreduce`. Status: fast path (cudaIpc peer writes) implemented and measured; pinned-host fallback
implemented and tested.

## Transport
- Peer SM access works now (ReBAR + `iommu=pt` + reboot). Cross-process cudaIpc kernel read returns `0xbeef`.
- `dev` transport (default when `torch.cuda.can_device_access_peer` is true): each rank cudaMalloc's its arena
  (`P2P_ARENA_BYTES`), zeroes it, exports a cudaIpc handle through `<rendezvous_path>/rank<r>.json`. The peer opens
  it with `cudaIpcOpenMemHandle` after `cudaDeviceEnablePeerAccess`. The kernel writes into the peer arena over P2P.
- `host` transport (fallback when peer access is unavailable, or `transport="host"`): `/dev/shm` arena, pinned and
  device-mapped. Same kernel, same protocol.
- Both ends must agree on the transport; `P2PComm` raises on mismatch.

## Protocol (unchanged from the host version)
- One kernel per collective. Each block pushes its stripe into the peer's parity slot, `__threadfence_system`,
  `st.release.sys` of seq into the peer's flag. It then `ld.acquire.sys` spins on its own flag.
- Sequence counters are per block, live in the arena (device memory for `dev`), and are advanced by the kernel.
  Graph replays therefore stay correct without re-recording.
- fp32 all-reduce: `rank0 + rank1` in fixed order on both ranks. Bit-identical results.
- 10 s spin timeout with `__trap`.

## API
- `P2PComm(rank, world=2, rendezvous_path=<fresh dir>, nblk=8, device=None, transport="auto")`
- `all_reduce_(t)` (fp32, in place), `all_gather(t, out)` (fp32, `out=[r0|r1]`), `bcast_i64(t, root, out=None)`,
  `barrier()`.
- Limits: 2 ranks, payload <= 256 KiB per collective, `nblk` in 1..8.

## Measurements
`qmesh/tp2/bench_p2p.py` in `exl3-tp2-p2p`, NBLK=8 (the default), ITERS=1000, GRAPH_N=80. Latencies are us per
collective. `p2p_*` is the P2P path, `nccl_*` is `dist.all_reduce` in the same harness.

| size | AR/AG/BC exact | graph exact | p2p eager | p2p seq | p2p graph | nccl eager | nccl seq | nccl graph |
|---|---|---|---|---|---|---|---|---|
| 20 KB (1x5120 fp32) | yes | yes | 4.86 | 6.73 | 4.93 | 18.01 | 23.70 | 11.84 |
| 128 KB (1x32768 fp32) | yes | yes | 8.27 | 9.41 | 8.73 | 16.83 | 24.88 | 23.28 |

Notes on the table:
- "seq" = 80 back-to-back collectives with a `mul_` between each. "graph" = replay of the same sequence, divided by 80.
- NBLK sweep for 20 KB graph: NBLK=1 6.31 us, NBLK=4 5.11 us, NBLK=8 4.93 us.
- 128 KB at NBLK=1 was 29.2 us (graph). NBLK=8 is the best of those tried.
- Target <= 5 us in-graph at 20 KB: met at NBLK=8 (4.93 us).
- Host fallback: 200 exact iterations with `transport="host"` (not benchmarked for latency).

## Caveats
- Bench NCCL numbers are single-run measurements, not averaged across runs.
- The fallback has not been timed. Expect it to be slower.
