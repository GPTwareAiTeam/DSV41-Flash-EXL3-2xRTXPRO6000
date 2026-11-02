# E2E hybrid DeepSeek-V4.1-Flash over the native RDMA seam

Hot model on P7 GPU1 (`/models/dsv41/exl3-serve-final`, `qmesh/placement.final.json`); cold experts on GX1
(`qmesh/worker_seam_server.py`, fused mixed-K, buckets 1/6/64 on ports 47311/47312/47313).

## Result (bsz 1, greedy, 64 new tokens)

| run | prompt | decode ms/tok (steady) | accepted | rejected | accept len |
|---|---|---|---|---|---|
| no DSpark | France | 31.4 | - | - | - |
| no DSpark | Fibonacci | 28.0 | - | - | - |
| no DSpark | colors | 29.9 | - | - | - |
| DSpark K=5 | France | 141.1 (includes warm-up) | 20 | 7 | 0.74 |
| DSpark K=5 | Fibonacci | 17.6 | 47 | 10 | 0.82 |
| DSpark K=5 | colors | 32.8 | 32 | 8 | 0.80 |

Mean acceptance length = accepted / (accepted + rejected) over all verify rounds (driver stats).

Coherence: the greedy outputs are coherent English. France gives "Paris"; Fibonacci gives a correct Python-style
description. The colors prompt drifts (not garbage) and ends with "...I will rate 5 stars".

Target (150 tok/s = 6.67 ms/tok) is NOT met. Best steady-state decode was 17.6 ms/tok (DSpark, Fibonacci), and
the no-DSpark runs ran at 28-31 ms/tok. Mean acceptance ~0.8 means DSpark does not pay for itself here.

## Commands

One GX1 server serves every P7 run until it is killed: the control port negotiates a session, the client gets
the data ports of the buckets it asked for, and the next client queues (`SEAM_QUEUED`) or starts as soon as the
previous session's control connection closes. Buckets are data ports `ctl+1+i` in `--buckets` order; `SEAM_BUCKETS`
on the client is only the request (the granted set is the intersection, printed as `buckets=… ports=…`).
```
ssh 10.0.100.2 'docker exec -d seam-gx1 bash -lc "cd /src/exllamav3 && python3 -u qmesh/worker_seam_server.py \
  --model-dir /cold --placement /cold/placement.exl3.json --dev rocep1s0f0 --gid 3 --port 47311 --buckets 1,6,256 \
  > /tmp/seam_server.log 2>&1"'
```
Wait for `SEAM_SERVER_LOADED` (~225-237 s), the pinned-slab bill `SEAM_SERVER_SLAB buckets=… predicted_mib=…
rows_max=… avail_mib=… fits=…` (allocated in `qes.Server` right after the load, so a list that does not fit
swaps the pack out - `fits=0` plus `SEAM_SERVER_SLAB_WARN` says so), and `SEAM_SERVER_READY proto=2 ctl=47311
buckets=…`. Do not restart it between clients; `python3 qmesh/seam_status.py --port 47311` (or
`qes.status(host, port)`) reports `active=`, `waiting=` and the last session without disturbing anyone.

The largest bucket is the client's per-call prefill row limit (`SEAM_SPLIT` in the client log), so the list is a
memory decision: one pinned slab per bucket, 1.18 MiB per row, sized *after* the pack is resident. Measured on GX2
(same box model and pack as GX1): after the 113.65 GiB cold pack loads, MemAvailable is 496 MiB, so 1,6,256
(309 MiB) fits, 1,6,64,256 (384 MiB) is the ceiling, and 512 alone (600 MiB) does not.

P7 client (pinned to host GPU1):
```
docker exec -e CUDA_VISIBLE_DEVICES=1 [-e DSPARK=1] -e SEAM_DEV=rocep142s0f0 -e SEAM_GID=5 \
  -e SEAM_PEER=10.0.100.2 -e SEAM_PORT=47311 -e REMOTE_TRANSPORT=seam seam-p7 \
  bash -lc 'cd /src/exllamav3 && PYTHONPATH=/src/exllamav3 python3 -u qmesh/e2e_seam.py'
```

## Fixes made along the way

1. **Server fp16 input** (`worker_seam_server.py` `compute`): the expert kernel requires `hidden_state` in kHalf. The
   server passed bf16, the kernel threw, and the callback returned stale response bytes. That produced the
   incoherent output (text was multilingual noise). Fixed by `.to(device).half()` on x. After the fix, output is coherent.
2. **Device pinning**: `seam-p7` had no `CUDA_VISIBLE_DEVICES`, so `cuda:0` resolved to host GPU0 and collided with
   another agent's GPU0 work. All runs now set `CUDA_VISIBLE_DEVICES=1`.
3. **Session lifecycle**: a session is bounded by its control connection, not by a bucket. The client opens a
   bucket's data handle on its first request of that size, so a decode-only run uses one of the three granted
   buckets; the unopened ones are reported as `SEAM_SESSION_PARTIAL … up=1 of 3` and are not an error. A
   connection on a data port that fails its hello (a stale client, a probe) is closed and logged as
   `SEAM_REJECT … reason=stale-data-conn` instead of aborting the server; that used to kill the process with
   `qes: peer layout mismatch` and cost a full cold-pack reload.
4. **Bucket padding and prefill splits** (`remote_seam_host.py`): a request pads to the smallest granted bucket
   that holds it with `id=-1, w=0`, which contribute exact zeros on GX1. Prefill is no longer cut at 64 rows:
   `connect()` takes the largest granted bucket as `chunk_rows`, `_split()` cuts a chunk into
   `floor(R/chunk_rows)` calls at that width plus one call for the remainder in the smallest granted bucket that
   holds it, and each call is issued and collected before the next (one request slot per layer). The width is
   therefore whatever the GX server granted - 64 on an old server, larger where the box has the pinned memory.
5. **Placement check**: the router-order tail maps exactly to the `warm` list (checked for L0: 259/259), so the
   `e - cpu_split_first` tail offset is correct. A suspected id-translation bug was an error in my check, not in the code.
6. **Native libraries**: GX1 uses the `aarch64` seam library and P7 uses `x86_64`. Earlier notes had this swapped.

## Not done / open

- Target (6.67 ms/token) is not met; this is the open acceptance item.
- [INFERENCE] The seam RTT (Transport.md: ~525 us/layer at rows=1, x40 layers ~ 21 ms) is above the 6.67 ms budget
  by itself, so per-layer seam round trips cannot reach the target. Not measured in this run.
- DSpark mean acceptance ~0.8 gives no speedup at this draft quality.

## Correction (Main)
"accept len" above is accepted/(accepted+rejected) from generator.py:1958-1966 = per-drafted-token
ACCEPTANCE RATE (0.74–0.82), not mean accepted length. That rate is good; DSpark does help: Fibonacci
28.0 → 17.6 ms/tok (−37%). Hybrid is bound by GX1 cold compute (~358 µs/layer rows=1, ~525 µs/layer round
trip ≈ 21 ms/token), not by DSpark. Mean tokens per verify round must be computed as
1 + accepted/verify_rounds (verify round count not logged here).

## Direct pack load (2026-11-02)

The cold shard can be filled straight from the EXL3 pack (`SEAM_MOE_DIRECT=1`,
`qmesh/worker_seam_server.py`), skipping the host-side module-build restack that used to cost
100-150 s and a CUDA allocation per layer. Two `--fresh` loads of the 4460 pack on the dev pair
(P7 GPU1 + GX2), each followed by an ids arm and a 32k/8192 prefill
(`seamrobust/scripts/seam_direct_plain.sh`, evidence `seamrobust/evidence/directpack/direct_1102-092228.log`):

    SEAM_PACK_KERNEL cudaexl3_layers=40 mixedk_layers=0 of=40
    SEAM_MOE_DIRECT layer=0..39 n=... read_ms=2100..2600   40/40, fail 0, nofit 0
    32k prefill at 8192-row chunks                         957.3 tok/s (canary ae4fe322 957.7,
                                                           control 9fc3ca8a 881.8)
    ARM_IDS arm=dspark1 n=8                               67158aeb... == the release's SMOKE_ARM_IDS
    cx_direct_check.py layers 5 and 39                     from_pack == module build, 2/2

`SEAM_SLACK avail_mib=1200 swapped_mib=1303` is this box's steady state, not a fault of the direct
path: the shipping server on the same box reports 1440/3816, and the prefill rate is unchanged.
`f96fc065` is a different digest - `SMOKE_ARM_IDS_TOTAL`, the same 8 tokens summed over all rounds -
so a ticket that compares the `ARM_IDS` line against it will call every good run a WARN.

**Two tables, one contract.** The seam's cold-local ids are positions in a placement file, so the two
halves must read the same bytes or every id means a different expert. The contract is the file:

- `/cold/placement.exl3.json`, sha `844251e79b9d` - the pack's own table. Its `warm` is ascending on
  all 40 layers, so the warm-order numbering below is a byte-identical no-op for it.
- `/cold/placement.exl3.ranked.json`, sha `5982a8ba7363` - same hot/cold sets and bit widths, ordered
  by global rank: hot by rank, then all 2800 4-bit cold experts, then all 8100 2-bit ones, each by
  rank (`qmesh/tests/make_ranked_placement.py`, reproducible byte for byte). Checked against
  `ranking/expert_ranking_2.25.csv`: 0 violations over 40 layers, bands disjoint, 10864 of 10900 cold
  positions differ from the plain file, `warm_bpw` permuted with `warm`.

The server loads `layers.N.warm` as the cold order and prints
`SEAM_PLACEMENT path=... ranked=0|1 warm_4bpw=... source_sha=...`; both halves of the client
(`block_sparse_mlp_cpu.load_cpu_split` and the module registration) assert that their permutation is
the file's. Two asserts make the boundary a property of the file instead of of the numbers:

- client side, at registration and at split time: `moe_remote_warm == placement.warm(...)`, and the
  client derives its cold count from `len(warm)`;
- server side, per direct layer: `SEAM_PLACEMENT_GROUPS layer=N n=... warm_4bpw_n=... groups4=...
  groups2=... ok=1`, i.e. the shard's own 4-bit group size equals the file's `warm_4bpw_n`, the 4-bit
  experts come first, and the totals agree (`sum(warm_4bpw_n) == sum(groups4) == 2800`). A MISMATCH
  is a `SystemExit` before any client is served; it can only fire when the file and the pack disagree.

**A client on the other table is refused, not fatal.** The Hello carries `placement_sha =
sha256(file)[:12]`; `RemoteSeamHost._check_placement` compares it with the file the client loaded and
raises `RuntimeError: placement mismatch on host:port` before the first request. The transport refuses
one connection at a time (close, one line, `SEAM_REJECT`) and keeps the pack: an unexpected client
costs a line, not a 150 s reload. `qmesh/tests/refusal_stub.sh` proves it end to end with a `--stub`
server and `qmesh/tests/placement_sha_refusal.py` (which must call `placement.load()` first -
`current_sha()` is only set by `load()`).

**The GX tree is a copy.** `qmesh/tests/stage_gx.sh` rsyncs the worktree and proves by sha that the
container runs this commit before anything is measured; without it a ticket measures whatever was
synced last, including a transport library the container built days ago
(`qmesh/native/bin/BUILD_INFO.aarch64` records the commit and `.cu` hash that produced the shard's
library).

Staging has to run **inside the gate and after the pair's server is stopped**: the GX tree is a live
bind mount, so an rsync under a serving process can hand a process that starts halfway through a tree
that is half one commit and half another. The order the ranked ticket uses is `gate take -> stop the
pair's server -> stage (exit 13 if a sha disagrees) -> box-clear check -> refusal -> oracle -> two
fresh loads`; staging first and gating later costs the reload it was meant to avoid.

`stage_gx.sh` also prints `STAGE lib_src seam_cu=... built_from=... state=match|stale`. The container's
library is compiled from the `.cu` on disk, so a stage can leave the source one revision ahead of the
library - the two commits after `dabfa4c9` add a `SEAM_HOOK_US` stderr line and change one busy
message, nothing on the wire or in the layout. `state=stale` next to passing numbers is that, and
rebuilding is a `build_gx_transport.sh` away once the box is free, at the cost of a new build id for
every container sharing the tree.
