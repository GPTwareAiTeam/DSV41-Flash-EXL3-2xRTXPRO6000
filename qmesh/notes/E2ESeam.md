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

GX1 server (restart before each P7 run; each handle accepts one client):
```
ssh 10.0.100.2 'docker exec -d seam-gx1 bash -lc "cd /src/exllamav3 && python3 -u qmesh/worker_seam_server.py \
  --model-dir /cold --placement /cold/placement.exl3.json --dev rocep1s0f0 --gid 3 --port 47311 --buckets 1,6,64 \
  > /tmp/seam_server.log 2>&1"'
```
Wait for `SEAM_SERVER_LOADED` (~225-237 s) and `SEAM_LISTEN` x3.

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
3. **Handle lifecycle**: a seam handle accepts one client. A rerun against an un-restarted server fails with
   `qes: connect timeout (errno=111)`. Restart the server before each P7 run.
4. **Bucket padding** (`remote_seam_host.py`): batches pad to 1/6/64 rows with `id=-1, w=0`, which contribute exact
   zeros on GX1. Prefill chunks by 64 rows.
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
