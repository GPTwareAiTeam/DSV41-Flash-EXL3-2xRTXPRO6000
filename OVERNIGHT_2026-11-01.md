# Overnight plan, 2026-11-01 (owner asleep ~01:00 → morning)

Supervisor (main session) runs this without owner input. Team: max 3 GPU agents + 1 integrator
(skill gpu-multiagent-orchestration). GPU work only through the pair queues
(`/home/treadall/recipe-artifacts/gpu0-gx1/queue/run_{prod,dev}.sh`). P7 GPU 2 is never touched.

## Standing rules for every agent
- Own git worktree + branch + container per agent. `/home/treadall/recipe-gpu0-gx1-v2`
  (`recipe/gpu0-gx1-v2`) is integration-only: nothing uncommitted there while runs are queued.
- GPU0/GPU1 always load from `/models/dsv41/exl3-2.8535-p7hot-hc1` (SSD). Never from /mnt/data12.
- GX boxes: one server load at a time, nothing else heavy on a GX while its server loads
  (113.65 GiB pack on 121 GiB box; GX2 hung twice on this).
- Speed claims: back-to-back ABBA, warmed GEMM tune cache, 40 held prompts + paired bootstrap CI.
  Unmeasured = [INFERENCE].
- Supervisor checks every agent ~every 15 min, verifies claims against logs, re-tasks, and writes
  progress into this file's log section.

## Slot A — GPU0+GX1 then GPU1+GX2: owner's memory/overhead answer (GxPort2)  [first, ~01:30]
1. final_prod (GPU0+GX1) + final_dev (GPU1+GX2): HC1, adaptive K (gate=auto), FP4 KV 1M,
   EXL3_LOAD_ARENA=0, WHOLE_GRAPH=0, PREFILL_ADAPTIVE=1 MAX_CHUNK=8192, needle 131k (+512k on dev).
   Deliverable: MEMB after load/warm/long, prefill + decode peaks, limiter chunk rows + measured rise,
   prefill tok/s, decode ms/token at depth, needle y/n.
2. Refit limiter constants for one GPU (PREFILL_ROW_KIB, DEPTH_BASE_MIB, DEPTH_BYTES; RESULTS §9c)
   from PREFILL_ADAPTIVE_LOG; commit.
3. Write `artifacts/MEMORY_BUDGET.md`: per-piece GiB, free, peaks, suggested overhead line, expert
   room at 2/2.5/3/4 bpw. This is the owner's morning answer.

## Slot B — expert quant from the official FP8 (QuantHC)  [starts when download completes ~01:00]
1. Hot: ranks 1–8000 → 4 bpw. Cold: ranks 7361–15360 → 2 bpw (640 overlap on purpose).
   Ranking `/home/treadall/qmesh-exl3-mixed/ranking/expert_ranking_2.25.csv`.
2. GPU0 + GPU1 through the queues, after Slot A's two runs. Own worktree/container.
3. Verify keys/rates, proxy_err summary vs job_2499, sha256 manifests, README per output.
4. Supervisor: grant long queue slots (write minutes to the queue extend file); keep Slot A/C
   short runs from starving it.

## Slot C — seam transport that never crashes (SeamRobust)  [after Slot A frees the tree]
1. Apply staged work (`/tmp/stage`, stash@{0}) in its own worktree; persistent multi-session
   server, negotiated buckets/build, busy/queue, status tool, ensure-server scripts.
2. Battery on GPU1+GX2 (negotiate, buckets, busy, wait, stray, legacy, kill9, status, decode, rows)
   until BATTERY_PASS; token identity + ms/token vs final_dev.
3. GX1 same; integrator merges to recipe/gpu0-gx1-v2; both GX boxes on one recorded build.

## Integrator (after Slot C merges)
- GX1 moved to the v2 tree (`seam-gx1v2`, ext build finished), seam-gx1b retired.
- Same commit on P7, GX1, GX2 (VERSION file on each). Re-run short HC1 arms on both pairs:
  expect prod ≈ dev.

## Backlog (start one at a time as a slot frees; ranked)
1. Speed: GX cold-expert kernel at 1–6 rows (now ~90% of the GX wait: 0.56 ms/layer at 1 row,
   1.98 at 6; fixed ~330 µs + ~290 µs/row). Profile the GB10 MoE kernel, target the per-layer
   fixed cost.
2. Adaptive K retune for this pair: feed measured per-row round cost (seam_rows table) into
   EXL3_DSPARK_EV_COST; fixed K=5 vs auto vs ev on 40 held prompts with CI.
3. WHOLE_GRAPH + DSPARK compatibility (export the last trunk block's hidden state from the graph path).
4. Prefill with 8192-row chunks: seam call count per chunk (bucket 64 → 128 calls/layer/chunk);
   test a 256/512 bucket; first 64-row call measured 2.5 s (JIT?), measure warm.
5. sens/RESULTS.md write-up (per-matrix study); decide tp2/permatrix-k2 fix merge.
6. Archive: GX2 `~/afd` (214 GB) move, rerun when GX2 idle; ARCHIVE_INDEX + journal.
7. Cleanup: stopped containers `exl3-sens-1031`, `exl3-sens2-1031`; old seam-gx1b after v2 verified.

## Morning report (supervisor, ready by owner wake-up)
Memory table (Slot A), quant outputs (Slot B), seam robustness result (Slot C), speed numbers
on both pairs on the same build, anything that failed and what was done about it.

## Log
- 00:50 final_prod4/final_dev4 died at load: an uncommitted QuantHC edit put class PrefixIndex inside
  SafetensorsCollection. Fixed (module level) and committed a9adf627; rule broadcast: no uncommitted
  edits in the v2 tree. Final runs relaunching.
