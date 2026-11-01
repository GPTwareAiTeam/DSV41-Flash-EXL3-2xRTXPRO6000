# Pair releases: keeping PROD (GPU0+GX1) and DEV (GPU1+GX2) on one version

Both hybrid pairs run DeepSeek-V4.1-Flash-1017 EXL3 as two halves that must agree byte for byte:
a P7 client container and a GX server container. Every outage we had (hung prefills, `peer layout
mismatch`, `remote invalid request`, SIGBUS, 10-30 min silent hangs on a stale JIT lock) came from
the halves drifting: different trees mounted, different seam protocol versions, libs or torch
extensions built from sources that no longer matched the tree the other half ran.

This note describes the release system in `qmesh/release/` (branch `tools/pair-release`, installed
into `/home/treadall/recipe-artifacts/gpu0-gx1/scripts/`). It answers one question per pair:
**are both halves on the same release, and is that release the one we think it is?**

## One release, one tree, both halves

```
/home/treadall/pairs/<pair>/            # P7 local dir; GX has the same layout at the same path
  releases/<sha>/
    RELEASE.json                        # sha, build_id, seam fingerprint, source list
    src/                                # the worktree, cut from the repo, identical both sides
    ext/                                # torch JIT extension cache for this release
  current -> releases/<sha>             # the only thing that moves on a flip
  extcache -> releases/<sha>/ext        # containers' TORCH_EXTENSIONS_DIR
  pairs/<pair>.json                     # manifest: what the guard compares against (in artifacts dir)
```

Containers mount the pair root at `/pair` and use `/src/exllamav3 -> /pair/current/src`. A tree
edit therefore reaches a container only through a release, and never reaches a half at all until
`current` moves. Staging containers (`seam-gx1stg`, `seam-gx2stg`, `exl3-staging-p7`) and the
robust line (`seam-*-robust`) are deliberately outside the manifests: they are some other
experiment's containers, and touching them would repeat the drift we are fixing.

## One container per pair -- never shared

Each pair owns its own client container on P7 and its own server container on the GX box, and the
names in `pairs.conf` belong to exactly one pair. Sharing a client container between the two pairs
is what let a dev update SIGKILL two prod arms on 2026-11-01: `dev` was configured with
`p7ct = seam-p7v2`, which was the container GPU0 arms actually ran in, so re-creating it for dev
killed them mid-prefill with `CLIENT_RC=137` and no traceback. Two guards now refuse that class of
mistake before any docker call:

* `pair_containers.py` refuses to touch a container that is another pair's `p7ct`/`gxct` in
  `pairs.conf` (`die: belongs to pair <p>`), and refuses to recreate one whose name appears on the
  command line of a live ticket runner in any queue (`die: named by a live ticket`). It applies to
  `apply` and `rollback` alike.
* `pair_update_run.sh` refuses to start when the pair's GPU has a compute app or when a client or
  server process is alive in either of the pair's containers.

## Release identity

* `sha` = the commit the source tree was cut from. The first release of the system is a commit on
  `tools/pair-release`, so the tooling that installed a release is part of it.
* `build_id = <pair>-<sha12>-<UTC stamp>`, written into `RELEASE.json` and into the stamp beside
  every artifact. Nothing is guessed later: a release tree and its artifacts are stamped together.
* `seam.fingerprint` = sha256 over `qmesh/native/qmesh_expert_seam.cu`, `...h` and `qes.py`. Those
  three define the wire. A pair whose halves disagree on it is a protocol mismatch by definition,
  whatever a constant in the code says.

## Life of an update

`pair_update.sh <pair> [sha] [--max-min N]` enqueues `pair_update_run.sh` through the pair's own
queue (rank 1, agent `PairUpdate`), so an update can never run concurrently with a job:

1. **refuse if busy** — GPU has a compute app, or a client/server process is alive in either of the
   pair's containers (an arm exec'd into a container dies with `CLIENT_RC=137` and no traceback
   when the container is re-created; that happened on 2026-11-01 and is why this check exists).
   Queued tickets are only logged: the queue serialises them.
2. **cut** — both sides get `releases/<sha>/src` from the repo, verified identical file counts and a
   matching seam fingerprint.
3. **containers** — `pair_containers.py apply` re-creates each side from `containers/<pair>-<side>.json`
   only when the live container differs (image, mounts, env, ulimits, devices), keeping the old one
   as `<name>.bak-<ts>`, wiring `/src/exllamav3`, then `plan` must be clean.
4. **build** — native lib (`--side both`, `nice -n 15`, `MAX_JOBS<=8`) and the torch extension, by
   absolute path into `releases/<sha>`, so `current` still points at the old release while the new
   one is being built. Native libs are not byte-reproducible across containers: compare the source
   sha and build stamp written beside each lib, never a `.so` hash across boxes.
5. **flip** — `current` and `extcache` move; then `inner` verifies each container really resolves
   `/src/exllamav3` to `/pair/releases/<sha>/src`.
6. **smoke** — start the pair's seam server (`seam_server_ensure.sh --fresh`, bounded wait) and run
   one decode row through `pair_arm.sh`, which pins `CUDA_VISIBLE_DEVICES=<pair GPU>`. The row
   decides pass/fail; memory numbers are logged, the slack gate stays off.
7. **refs** — the smoke's text hash is recorded, so a later release can be compared against it.
8. **manifest + verify** — `pairs/<pair>.json` is written (and refused if any stamp is missing), then
   `pair_check.sh <pair>` must return 0 before `PAIR_UPDATED` is printed.

Any failure after the flip calls `fail()`, which prints `PAIR_UPDATE_FAILED phase=… why=…` and then
restores: `current` back to the old sha, containers back from `.bak-<ts>`, old server restarted. The
process group is `setsid`ed and the TERM/INT handler flips `current` back and exits 143; cancel with
`kill -- -<pgid>`, never a bare `kill`.

## The guard the queue uses

`pair_check.sh <pair>` → 0 `IN_SYNC`, 4 `PAIR_DRIFT`, 5 error. `gpu_run_pair.sh` (guarded copy in
`qmesh/release/`) calls it before a job starts and retries 3×10 s:

* no manifest = `PAIR_DRIFT no_manifest` (fail closed: a pair that was never released can not be
  trusted to run a job);
* `AGENT=PairUpdate|PairUpdateRun` and `PAIR_CHECK_SKIP=1` skip the check — otherwise the update
  could never run on a drifted pair;
* the check compares, per side: `current` == release, `extcache` == release ext, `RELEASE.json`
  build_id, lib stamp (arch, sm, source sha, build_id), ext stamp, container wiring, and the tree
  hash of the seam files against the other side and against the manifest.

## Commands

```
pair_cut.sh <pair> [sha]              # cut a release tree on both sides, print sha/build_id
pair_containers.py plan|apply|rollback <pair> [sha|ts]
pair_build_lib.sh <pair> <sha> [--side both|p7|gx]
pair_build_ext.sh <pair> <sha> [--side both|p7|gx]
pair_env.sh <pair>                    # the env a job should run with (CUDA_VISIBLE_DEVICES etc.)
pair_arm.sh <pair> [--rows N]         # one client arm through the release
pair_smoke.sh <pair>                  # server + 1-row arm, SMOKE_OK / SMOKE_FAIL
pair_ref.sh <pair> [--compare]        # record / compare the decode reference
pair_check.sh <pair> [--quiet]        # guard: 0 IN_SYNC, 4 PAIR_DRIFT, 5 error
pair_status.sh <pair>                 # human summary of both sides
pair_manifest.py write|show <pair> <sha>
pair_update.sh <pair> [sha]           # queue an update (rank 1)
pair_promote.sh <sha> [--force]       # dev-passed sha -> prod, queued behind prod's job
```

Roll-out order is always dev first: `pair_update.sh dev`, then `pair_check.sh dev` must be
`IN_SYNC`, then `pair_promote.sh <sha>` (it refuses unless dev's manifest carries that sha and the
seam fingerprint matches; `--force` is for a deliberate protocol change).

## Dry runs

`PAIR_DRY=1` plus `PAIR_DRY_SSH=<stub>` runs any of the above against a fake transport: builders
print `DRY LIB_BUILD`/`DRY EXT_BUILD` and write nothing, `pair_containers.py` refuses to call docker
without `PAIR_DRY_SSH`, and `pair_update_run.sh` ends with `PAIR_UPDATED_DRY`. `PAIR_DRY_FAIL_AT=<phase>`
forces a failure at a named phase to exercise the restore path. A dry run can therefore never
produce production evidence, and it can never touch a real container.

## Memory is a separate question from drift

The GX2 pack is ~113.5 GiB against a ~120 GiB box: loads sit at 860-960 MiB of headroom with
3.5-4.0 GiB swapped, and a 6-row prefill can still die with `CUDA error: unspecified launch failure`
on that build. A smoke that fails there is a capacity failure, not a code mismatch: the release
still reports `PAIR_UPDATED` with `smoke` marked failed and the memory lines attached. Shrinking the
pack or gating on `dev_free_mib` is the owner's call, not the release's.
