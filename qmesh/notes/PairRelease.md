# Pair releases: keeping PROD (GPU0+GX1) and DEV (GPU1+GX2) on one version

Both hybrid pairs run DeepSeek-V4.1-Flash-1017 EXL3 as two halves that must agree byte for byte:
a P7 client container and a GX server container. Every outage we had (hung prefills, `peer layout
mismatch`, `remote invalid request`, SIGBUS, 10-30 min silent hangs on a stale JIT lock) came from
the halves drifting: different trees mounted, different seam protocol versions, libs or torch
extensions built from sources that no longer matched the tree the other half ran.

This note describes the release system in `qmesh/release/` (branch `tools/pair-release`, installed
into `/home/treadall/recipe-artifacts/gpu0-gx1/scripts/`). It answers one question per pair:
**are both halves on the same release, and is that release the one we think it is?**

## A smoke stops the server it started (2026-11-02)

`pair_smoke.sh` used to keep its worker on a passing verdict and delete the pidfile that named it, on
the theory that the loaded server was "the update's result". Prod's 01:17 smoke did exactly that: at
02:30 `seam-gx1-prod` was still serving a release from an hour earlier, with no record of which ticket
owned it -- and a later ticket that wanted the box could not tell a live pair server from a leak. A
server whose owner is a finished ticket is a leak even when it measures well.

The EXIT trap now runs `pair_server_stop` and then `pair_gx_gate_down`, in that order: a stop is a
lifecycle event on the box and belongs inside the claim. The success path no longer touches the
pidfile, so the trap always has the record it needs; `pair_update_run.sh` no longer asks the smoke to
keep the server, and no longer deletes the record after verify. A pair that needs a server gets one
from the ensure when a client asks, under that client's claim.

## A container that lost its GPU is restarted, not recreated (2026-11-02)

The 02:18 systemd daemon reload (an `nmcli connection modify` for the new MTU) re-derived the device
cgroup of every *running* container on P7, GX1 and GX2 and dropped its GPU access. The containers stayed
`Up`, the hosts were healthy, and their processes died on the next CUDA/NVML call: "Failed to initialize
NVML: Unknown Error", then "No CUDA GPUs are available" from `torch.cuda.Stream`. That is the whole of
the 02:22 and 02:26 failures -- not a module reload, not the UVM reset a create causes.

`pair_nvml_ok <p7|gx> <ct>` asks the container itself; `pair_ensure_nvml` restarts it once and waits up to
30 s for the driver to answer again. The smoke calls it for both containers *after* `pair_gx_gate_take`
and before the memory wait: a restart kills whatever the container was doing, so it may only happen
under the box's claim. `pair_prepare.sh` only reports (`PREPARE_NVML_WARN`): prepare runs out of band
with nobody's claim, and a restart there is the same accident with a different trigger.

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

`pair_cut.sh` stamps a release **once and for both sides**. If one box already carries
`releases/<sha>/RELEASE.json`, the other side takes that file verbatim (`CUT_ADOPT`) instead of
rendering its own; `--force` is the only way to re-stamp. Without that rule a prepare that found the
P7 half already cut would give the GX half a fresh `build_id`/`cut_at` inside the same release, and
the smoke -- which compares the build id the client reads with the one the server reads -- would call
a pair with byte-identical code mismatched. The adopted stamp is also checked against the commit: a
digest or file count that does not describe `sha` is refused, not adopted.

The stamp is sent **inside the ssh command** (base64), never on the ssh stdin: a redirect written
outside `$(pair_gxsh …)` binds to the calling shell, and the remote `cat` then read the host's own
stdin and wrote a 0-byte `RELEASE.json`. That is how gx2 came to hold an empty stamp for
`9fc3ca8a` on 2026-11-02: dev smoked green (857.9 tok/s, ids `f96fc065…`) and then failed verify
with `DRIFT release_json=missing`, which rolls `current` back. The cut now refuses to ship a stamp
it cannot read back (`STAMP_EMPTY`/`STAMP_WRITE_FAILED`), refuses a gx stamp that is not
byte-identical to the p7 one (`STAMP_MISMATCH`), and a later cut **repairs** an existing release
whose stamp is 0 bytes (`STAMP_REPAIR`) without touching the tree.

A release may also be a **merge** of a runtime branch into the release line, and that is how the
server fixes arrive: `bd15ae60` merged `seam/session-multiclient` before release 1, and release 2 is
the same merge for the compute lock (`39f07dc4`). The ancestry guard is what makes this deliberate
rather than accidental -- `pair_cut.sh` refuses a sha that is not on `PAIR_BRANCH`
(`tools/pair-release`), so a fix reaches a pair only after it is merged there:

```
git -C ~/recipe-gpu0-gx1-v2 merge --no-ff seam/session-multiclient    # in the release worktree
pair_cut.sh dev <merge-sha>; pair_prepare.sh dev <merge-sha>          # then the flip ticket
```

Before cutting a merge, run the release suites on the merged tree (`git commit-tree` + a detached
`git worktree add` runs them without moving any branch): the merge brings the runtime's own tests
and the tooling's, and only the merged tree says whether they still agree.



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
4. **build** — native lib (`--side both`, `nice -n 15`, `MAX_JOBS<=8`) and the torch extension into
   `releases/<sha>`, so `current` still points at the old release while the new one is being built.
   The extension is built *through* `/src/exllamav3` (the link is repointed to the new release for
   the length of the build and put back afterwards) because that path is part of the ninja command
   line; see the JIT incident below. Native libs are not byte-reproducible across containers: compare the source
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
restores: `current` back to the old sha, containers back from `.bak-<ts>`, and the old server
restarted -- but only if this pair was serving before the update (`UPDATE_SERVER_BEFORE serving=0`
leads to `RESTORE_SERVER_SKIP`): a pair the ticket found idle must be left idle, because reloading a
113 GiB pack on GX2 costs minutes and is exactly the memory-tight state the smoke just failed on. The
process group is `setsid`ed and the TERM/INT handler flips `current` back and exits 143; cancel with
`kill -- -<pgid>`, never a bare `kill`.

## Prepare, then flip: an update is two steps now

Building a release costs 5-13 minutes per side inside the pair's containers, and doing that inside
the ticket kept the pair's GPU idle the whole time. So the work is split:

1. `pair_prepare.sh <pair> [sha] [--side p7|gx|both]` -- **outside** the queue, while the pair keeps
   running whatever it is running. It runs the same cut / containers / lib / ext phases, niced, into
   `releases/<sha>` only, and ends with `PREPARED <pair> sha=… build=…`. It never moves `current`
   (the one exception stays the first-release bootstrap that container wiring needs) and never
   starts, stops or re-creates a serving container: a serving container, or a live ticket naming one
   of the pair's containers, refuses the prepare outright.
2. `pair_update.sh <pair> <sha> --flip-only` -- enqueues a ticket that builds nothing. It verifies
   what the prepare left (`pair_build_lib.sh verify`, `pair_build_ext.sh verify`, and `plan` must
   show no `RECREATE`), then flips, smokes, writes refs, the manifest and runs `pair_check.sh`. A
   missing or foreign stamp stops it at `phase=prepared`, with "run pair_prepare.sh <pair> <sha>",
   before anything is flipped. The one delta a flip ticket can be allowed to apply is a container
   spec change: with `--recreate` it re-creates from the spec inside the queue (see "A pack lives on
   the drive that serves it"); without it, a `RECREATE` stops the ticket.

Verify is read-only and runs before the build guards in both builders: a flip-only ticket verifies
while GX is still loading a server and while other builds own the box, and neither has anything to
do with reading a stamp. `pair_build_ext.sh verify` needs `pair_ext_stamp_check.py` (one definition
of "this stamp names this build").

For prod, the prepare has to happen in a container that is **not serving**: create `seam-p7-prod` /
`seam-gx1-prod` fresh and prepare there, while the arms that are running keep their own containers.

## Dev is the canary, and the gate is code

`pair_update.sh prod` only exists behind `pair_promote.sh`. Without `PAIR_PROMOTE=1` it exits 2 with
the command to use, before it reads anything or touches the queue; `pair_promote.sh` is the only
thing that sets that flag. The flag alone is not enough: for that exact sha there must be a
`pairs/dev/evidence/<sha>.json` written by the dev ticket itself, and it must say
`ok=true`, carry a smoke row, and carry **generated token ids** (`smoke.ids.n > 0`).
The promote queues `pair_update.sh prod <sha> --flip-only`, the same two-step shape dev uses: what
prod runs was cut, wired and built by `pair_prepare.sh` outside the queue, and the ticket only
verifies the stamps, flips and smokes. Until 2026-11-02 it dropped the flag, so every promote left
the prod queue compiling (that day's prod ticket rebuilt both transport libs and both torch
extensions between 00:42:03 and 00:42:34). `pair_promote.sh --build` is the escape hatch for a pair
that was never prepared, and `tests/sweep_rules.py` pins the default.

Text is not proof. A wrong build can still emit fluent-looking output; the ids it actually generated
are what a mismatched tree, a stale JIT cache or a half-flipped release cannot reproduce. The smoke
arm therefore runs through `pair_arm_ids.py` (`pair_arm.sh --ids`), which wraps the release's own
`run_arm` and prints `ARM_IDS arm=… n=… sha256=… first=…`; the rows are unchanged, so a reference row
recorded with ids still compares against one recorded without. An arm that produces no ids fails the
smoke with `arm_no_ids` (opt out with `SMOKE_IDS_OPTIONAL=1` for a release whose driver cannot report
them -- such a release can not be promoted).

One line per arm, plus `SMOKE_ARM_IDS_TOTAL n=<sum> sha256=<commitment over every ARM_IDS line>`;
that total is what the ticket records in the evidence, so two arms of one smoke can never be
confused with one arm that ran twice.

Evidence is per sha, not per pair: dev's manifest moves on with the next release, a promotion may
follow days later. A failed attempt writes the same file with `ok=false`, the phase it died in and
the reason, so a later promotion says *why* rather than "no evidence".

`pair_canary_gate` in `pair_lib.sh` is the single definition of "dev passed": `pair_update.sh` refuses
early with it, `pair_promote.sh` is the only door, and `pair_update_run.sh` -- the script that
actually flips -- checks it again before any phase, so queueing the ticket's script by hand is not a
way around either door. `tests/promote_gate.sh` sweeps all of them, including that a refused ticket
runs no docker command at all.

The order is therefore: `pair_prepare.sh dev <sha>` (out of band) → `pair_update.sh dev <sha>
--flip-only` → wait for `PAIR_UPDATED … ids=n>` → `pair_prepare.sh prod <sha>` in the fresh,
non-serving prod containers → `pair_promote.sh <sha>` (which re-checks dev, the seam fingerprint and
the evidence, then queues the prod ticket).

## A pack lives on the drive that serves it

Owner rule, 2026-11-02: the expert packs are on the local drive of the box that serves them --
P7 hot packs under `/models/dsv41` on the P7's 4 TB NVMe, GX cold packs on the GX box's own disk,
and `cuda-exl3` weights under GX1's `/mnt/data10` and P7's `/mnt/data12`. Nothing a pair serves with
is reached over the network: GX1 mounts `/mnt/data10` and `/mnt/data12` from P7 over NFS, so a pack
under `/mnt` there is P7's disk read across the wire, and every decode step waits on it.

`pair_pack_disk.py <pair>` is the check, and it reads the spec the pair actually runs
(`containers/<pair>-<side>.json`), so it cannot drift from what the containers mount. It refuses a
P7 spec that mounts `/mnt`, a `/models` that is not the drive itself, a pack, placement profile or
symlink that resolves outside `/models/dsv41`, and any of those on a network filesystem
(`nfs*`, `cifs`, `9p`, `sshfs`, …); on the GX it refuses a `/cold` source under `/mnt` or `/net`, or
on a network filesystem, and holds the pack's symlinks to the same rule. It runs in
`pair_prepare.sh` (before the cut) and again in `pair_smoke.sh` (before the GX gate), and
`tests/pack_disk.py` pins the refusals.

The P7 containers used to be handed `/mnt/data12` so the 4460 hot pack's links could be resolved;
that mount is gone from both P7 specs, so the containers no longer see the 12 TB drive. A spec
change is the one thing a flip-only ticket will not do on its own: `plan` reports a `RECREATE`, and
the ticket stops unless it was queued with `--recreate` (`pair_promote.sh <sha> --recreate` for
prod), which is the only path that sets `PAIR_ALLOW_RECREATE=1`. The ticket then re-creates from the
spec inside the queue, renaming the old container to `<name>.bak-<ts>` -- never a hand-written
`docker rm`, and dev still goes first.

## Incidents that shaped these guards

* **Stale compiler storm** -- `exl3-tp2-moe` left 349 `nvcc`/`cc1plus` processes alive for days and
  the "wait for the box to be free" loops waited on them forever. `compilers()` counts only
  processes younger than `PAIR_BUILD_STALE_SECS` (1800 s) and prints `stale_ignored=<n>`: an older
  compile is a corpse, not a build.
* **Root-owned `bin/`** -- the container built the `.so` as root and the stamp write then failed with
  `Permission denied` on the host. Builds `chown` the artifact back to the invoking uid/gid before
  the stamp is written.
* **Absolute `current`** -- `current -> /home/treadall/pairs/<pair>/releases/<sha>` resolves, inside
  the container, to a path that does not exist there, so `/pair/current/src` dangled. The links are
  created relative to the pair root (`ln -sfn releases/<sha> …`), so one tree works on the host and
  at `/pair` alike.
* **`seam_server_ensure.sh` not executable** -- `pair_smoke.sh` requires `-x` before it starts a
  server; a non-executable copy made every smoke die with `SMOKE_FAIL no_seam_server_ensure`.
* **Killed JIT build, then silence forever** -- the extension cache was built with
  `PYTHONPATH=<release>/src` while every client imports `exllamav3` through `/src/exllamav3`; the
  include path is part of the ninja command line, so the client's first import recompiled all 157
  edges. torch runs ninja with stdout captured and `verbose=False`, so the smoke's arm printed
  *nothing* for 300 s and died `rc=124` -- indistinguishable from a seam mismatch. Worse, ninja
  writes `.ninja_log` only when it exits: a build killed by a timeout leaves no log entries, and the
  next run recompiles everything again, so a killed rebuild repeats forever. The builder now builds
  through the same `/src/exllamav3` path a client uses (temporarily repointing the link for the
  length of the build, restoring it afterwards) and ends with a plain client import (`EXT_CLIENT_HIT`,
  capped by `EXT_CLIENT_TIMEOUT`) that must be a cache hit; a miss fails the build with
  `EXT_FAIL … EXT_CLIENT_MISS` instead of shipping a cache every run rebuilds. `pair_smoke.sh` keeps
  the arm's bytes in `<state>/logs/smoke-arm-<pair>-<stamp>.log` and prints `SMOKE_ARM_LOG`, the GX
  server log tail (`SMOKE_SERVER_TAIL`) and both containers' `extcache` listings on failure.
* **A ticket with nothing to run** -- `pair_update.sh` enqueued `"${cmd[@]}"` without ever
  building `cmd`, and bash expands an empty array to nothing without a word: the queue got a
  ticket with no command, `gpu_run_pair.sh` started it, `end rc=1 elapsed=0m` and the pair was
  reported as having had its update. `pair_enqueue` now refuses an empty command, and
  `tests/update_enqueue.sh` reads the argv the runner actually receives.
* **A prefix is not the release** -- the stamps name the full commit because that is what
  prepare builds, while the operator types `4f299c3a`. `<build>.sh verify <pair> 4f299c3a`
  compared the 8-character prefix to the stamp and printed `LIB_STAMP_MISMATCH
  source_sha=<the same sha>` -- a refusal that hid itself. Every entry point resolves the
  argument to a commit (`rev-parse --verify <sha>^{commit}`) before it is compared.
* **The grant is not free** -- the server reserves workspace for its *widest* granted bucket
  whether or not anything asks for it: 294 MiB at 2048 rows against ~9 MiB at 64. On a box
  whose pack leaves ~1 GiB free that reservation is the difference between a smoke and a
  paging wall. `SMOKE_BUCKETS` (default `1,6,64`) narrows the grant to what the smoke
  exercises; the pair's real server still takes `--buckets` from `pairs.conf`.
* **Smoke arm killed silently under swap** -- on a GX box that is paging (avail < 1 GiB, 3-4 GiB
  swapped) the first token of the smoke row can outlast the arm timeout, which then returns 124 with
  no output at all. That is a capacity failure: report the memory lines, do not read it as a
  protocol mismatch, and raise `SMOKE_ARM_TIMEOUT` rather than shrinking the smoke.

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

The guard is installed **last**, and only when it can pass: a pair with no manifest fails closed, so
installing it before `pairs/<pair>.json` exists turns every job on that pair into `PAIR_DRIFT
no_manifest`. And a `gpu_run_pair.sh` that a ticket is already executing must not be replaced under
it — bash reads the file as it goes, and every live `bash …/gpu_run_pair.sh <Agent> …` is a byte
offset into the old text. Order: dev manifest → prod manifest → both queues idle → install, `cmp`
against the branch copy.

## Prod's first release: new containers, new names

Prod's GX row in `pairs.conf` still names `seam-gx1v2`, a container someone wired by hand before
this tooling existed (wrong `/pair` mount, `TORCH_EXTENSIONS_DIR=/root/.cache/…`, `MAX_JOBS=12`).
`pair_containers.py plan prod` therefore reports `RECREATE` for it — and a RECREATE is exactly what
must not happen to a container that arms are running in. Prod moves to containers of its own
instead:

1. When prod's queue is idle and its arms are done, point the prod row of `pairs.conf` and the
   `name` in `containers/prod-gx.json` at `seam-p7-prod` / `seam-gx1-prod`, and install the two files
   byte-identically. `seam-p7v2`, `seam-gx1v2` and `seam-p7v2-robust` are then simply not ours: they
   are never re-created, stopped or exec'd by this tooling again.
2. `pair_prepare.sh prod <sha>` — plan now reads `RECREATE missing:seam-p7-prod`,
   `RECREATE missing:seam-gx1-prod`, which is a create, and creates both while the arms keep their
   own containers.
3. `pair_promote.sh <sha>` — the ticket flips `current`/`extcache` inside the new pair containers,
   smokes, writes the prod manifest and `pair_check.sh prod` must end `IN_SYNC`.

## Commands

```
pair_cut.sh <pair> [sha]              # cut a release tree on both sides, print sha/build_id
pair_containers.py plan|apply|rollback <pair> [sha|ts]
pair_build_lib.sh <pair> <sha> [--side both|p7|gx]
pair_build_lib.sh verify <pair> <sha> [--side both|p7|gx]   # stamps only, builds nothing
pair_build_ext.sh <pair> <sha> [--side both|p7|gx]
pair_build_ext.sh verify <pair> <sha> [--side both|p7|gx]   # stamps only, builds nothing
pair_prepare.sh <pair> [sha] [--side both|p7|gx]            # cut+wire+build, outside the queue
pair_env.sh <pair>                    # the env a job should run with (CUDA_VISIBLE_DEVICES etc.)
pair_arm.sh <pair> [--rows N]         # one client arm through the release
pair_smoke.sh <pair>                  # server + 1-row arm, SMOKE_OK / SMOKE_FAIL
pair_ref.sh <pair> [--compare]        # record / compare the decode reference
pair_check.sh <pair> [--quiet]        # guard: 0 IN_SYNC, 4 PAIR_DRIFT, 5 error
pair_status.sh <pair>                 # human summary of both sides
pair_manifest.py write|show <pair> <sha>
pair_update.sh <pair> [sha] [--flip-only]   # queue an update (rank 1); --flip-only builds nothing
pair_promote.sh <sha> [--force]       # dev-passed sha -> prod, queued behind prod's job
```

Roll-out order is always dev first: `pair_prepare.sh dev <sha>` (or `pair_update.sh dev <sha>`
for the one-shot form), then the flip-only ticket, then `pair_check.sh dev` must be `IN_SYNC`, then
`pair_promote.sh <sha>` (it refuses unless dev's manifest carries that sha and the seam fingerprint
matches; `--force` is for a deliberate protocol change).

## Dry runs

`PAIR_DRY=1` plus `PAIR_DRY_SSH=<stub>` runs any of the above against a fake transport: builders
print `DRY LIB_BUILD`/`DRY EXT_BUILD` and write nothing, `pair_containers.py` refuses to call docker
without `PAIR_DRY_SSH`, and `pair_update_run.sh` ends with `PAIR_UPDATED_DRY`. `PAIR_DRY_FAIL_AT=<phase>`
forces a failure at a named phase to exercise the restore path. `PAIR_STATE_OVERRIDE=<dir>` keeps
the dry run's logs, refs and restore notes out of the real `pairs/` tree. The sweeps are
`tests/dry_run.sh` (whole update) and `tests/flip_only.sh` (the prepare/flip split: no sha, unwired
containers, a foreign ext stamp, then a prepared release that flips without building). A dry run can
therefore never produce production evidence, and it can never touch a real container.

## Memory is a separate question from drift

The GX2 pack is ~113.5 GiB against a ~120 GiB box: loads sit at 860-960 MiB of headroom with
3.5-4.0 GiB swapped, and a 6-row prefill can still die with `CUDA error: unspecified launch failure`
on that build. A smoke that fails there is a capacity failure, not a code mismatch: the release
still reports `PAIR_UPDATED` with `smoke` marked failed and the memory lines attached. Shrinking the
pack or gating on `dev_free_mib` is the owner's call, not the release's.

So the ticket tells the two apart, and only capacity failures skip the rollback. `pair_smoke.sh`
marks a refusal from the box (`PREFLIGHT_FAIL gx-mem`, `SEAM_SLACK` with the gate on) and a client
the kernel or CUDA killed (`out of memory`, `Cannot allocate memory`, `unspecified launch failure`)
**in the client's own output** as `SMOKE_FAIL memory …`; `pair_update_run.sh` then takes the
`memory_fail` path instead of `fail`: the release stays installed and current, evidence says
`ok=false, why=memory:…`, **no manifest is
written**, and it prints `PAIR_UPDATED … smoke=FAILED_MEMORY rollback=none manifest=none` plus the
`SMOKE_MEM_*` lines. That last part is the point: the pair fails closed -- the queue refuses jobs on
an unverified release -- while the box, not the code, is named as the reason. A wire mismatch, a
stamp mismatch or a crash still rolls back.

What is *not* in that list is the server's own `SEAM_COMPUTE_FAIL layer=N rows=1 err=AcceleratorError:
CUDA error: unspecified launch failure ... exiting: a server that cannot compute must not answer`.
It arrives in the server's tail, not in the client's output, so the pair treats it as a hard failure
and rolls back (that is what the dev smoke did on 2026-11-01 at avail 560 MiB). Widening the memory
verdict to cover a server-side line is a decision for the owner, not something a smoke script should
quietly assume.


## Measured, 2026-11-01: the margin, and what names a build

The numbers above are one night's, and they are what the pair now encodes:

| | GX2 (dev) | GX1 (prod) |
| --- | --- | --- |
| MemTotal | 124546 MiB | 124546 MiB |
| clean `MemAvailable` | 120586 | 120311 |
| pack load needs | 116362 | 116362 |
| after the load | avail 1120-1183, swapped 3692-3890 | avail 1549 refused / 1633-1657 ok |
| first compute dies at | avail 560 (`SEAM_COMPUTE_FAIL layer=19 rows=1`) | avail <= 1549 |

The load costs the pack plus ~1100-1300 MiB (`exl3` module scratch, pinned slabs, the CUDA context)
and one forward needs ~1600 MiB more. `SEAM_LOAD_MARGIN_MIB` (`qmesh/worker_seam_server.py:99`) is
the only place that decides whether the box has that; its default is now 2048 (it was 1024) and that
is a *floor*, not a promise. `pairs.conf` states the pair's own number -- prod 2560, dev 3072 --
`pair_lib.sh` exports it as `LOAD_MARGIN`, and both places that start a server (`pair_smoke.sh`,
the restore in `pair_update_run.sh`) pass `SEAM_LOAD_MARGIN_MIB=$LOAD_MARGIN` through `SEAM_ENV`.
Nobody starts a server on the default any more, and the sweep rule keeps it that way.

Be exact about what the margin can do: GX2 passes 3072 by ~1150 MiB and *still* lands on the wall,
because the box is 120586 MiB clean and the pack is 116362. The margin refuses a box that something
else has already made dirtier; it cannot make GX2 fit. The operator signal that the box is over its
head before the first request is `SEAM_SERVER_SWAP_WARN` (512 MiB): GX2 prints it at ~3.7 GiB swapped
on every start, margin or no margin.

The second half of tonight's trap: a released tree carries no `.git`, so `qes.build_id()` (env ->
`git rev-parse` -> `<tree>/qmesh/BUILD_ID`) answered `"unknown"` and
`SEAM_SESSION_START ... build=unknown proto=3` was accepted -- the pair was mixed and nothing said
so. `pair_cut.sh` now writes `<tree>/qmesh/BUILD_ID` into both halves when it builds a release and
`pair_arm.sh` exports the same id as `SEAM_BUILD_ID` into the client container, so `build=` names
one release on both sides and a mixed pair is visible in the first log line.

Drift that is deliberate and therefore recorded rather than chased: `qmesh/native/*.so` is not
tracked in git, so a release *rebuilds* the seam lib from the commit. Three builds of the same
sources exist on the boxes (79048 B measured by GxPort2 in `exllamav3-fork`, 78712 B in
`recipe-gpu0-gx1-v2`, 154936 B from this builder), so a release's lib is not byte-comparable with a
hand-built one. Release stamps name the build; numbers measured against a hand-built lib are tied
to that lib's sha and must be re-baselined when a release takes the tree over. In the same spirit,
`seam-gx1b` was patched in place (`worker_seam_server.py` `242854c1` -> `da7e82c0`, the port of the
compute lock) while it serves no release tree: `pair_check.sh` reporting `DRIFT` there is expected
until a release owns that container.

## Release 2 does not measure like a hand-patched tree

Release 2 (`3c361e8f`, the `--no-ff` merge of `39f07dc4`) is release 1 plus the compute lock, the
2048 MiB margin default and the `1,6,64,256,2048` bucket default. Its seam fingerprint is
**unchanged** (`fp:6a6bde315471d3f5`, proto 3): `qes.py`, `qmesh_expert_seam.cu` and `.h` did not
move, so the wire, the staging cost and the per-row device cost are release 1's, and a release-1
client could still talk to a release-2 server. What changed is how the server schedules work: one
`_forward` at a time, whatever the handle count.

That is why a release-2 number is not interchangeable with a number measured on a hand-patched
tree. GxPort2's arms run `242854c1` (no lock) in `seam-gx1b`; a multi-handle client there can
interleave two forbids inside one module scratch on one stream, and a run that survives is paying a
different scheduling cost than the release's serialized one. The fingerprint says a *pair* is
consistent and is the right check for mixing; it does not say two *trees* measure the same. A
throughput baseline taken on a hand-patched tree is re-taken on the release before it is compared
(`pair_ref.sh <pair> --compare` is where that comparison lives).

**The lock is not the fix, and release 2 must not be sold as one.** The identity gate on the prod
pair (release 2's own server file, one client, legacy buckets `1,6,64`, ns=1) reproduced the morning
death exactly: `12:35:38 SPLIT_GATE_SERVER … avail_min_mib=560 swap_max_mib=3762` then
`SEAM_COMPUTE_FAIL layer=19 rows=6 err=AcceleratorError: CUDA error: unspecified launch failure`,
and the split arm was never reached. The lock removes a real hazard -- two `_forward`s interleaving
inside one module scratch and one stream -- but a single-threaded client still dies, so the faulting
frame is elsewhere. Release 2 is still the right thing to *hold* a pair on (release 1's wire, the
margin floor, a build id on both halves); it is not a cure, and a dev flip onto it tonight would
only move the same crash to the dev pair.

The provenance caveat, concretely: the 512-row and 131k arms were measured on `242854c1` -- no lock,
`SEAM_LOAD_MARGIN_MIB` 1024 -- while release 2 carries the lock and 2048. So the sequence for
quoting any of those numbers after a flip is: re-measure on the release
(`pair_ref.sh <pair> --compare`), then compare. Skipping that step compares two schedulers and two
margins, not two builds.

### Status of release 2: withdrawn, not flipped

`3c361e8f` was cut for both pairs at 12:33 and the dev build was started; the negative identity
gate arrived while it ran. The build was stopped, the leftover compiler processes were killed, and
`releases/3c361e8f…` was removed from all four sides (`pairs/{dev,prod}/releases` on P7 and on both
GX boxes) together with its half-built ext cache and the torch `lock` baton a killed build leaves
behind. No `current` link was ever moved: dev still points at `32923b9a`, prod has no link, no
manifest exists, and nothing was queued, so neither pair changed state tonight.

The merge itself is **reverted** (`2e4eec4b`, `git revert -m 1 3c361e8f`), so the release line is
release 1's server again: `qmesh/worker_seam_server.py` is `8eb4f792…`, the file release 1 shipped.
That is the honest state -- a lock that does not fix the crash must not ride into a release on the
strength of being harmless. The fix that is coming (the `SEAM_COMPUTE_TRACE` / `CUDA_LAUNCH_BLOCKING`
work, `c011674d`) is merged as its own change; if it turns out to need the lock, the way back is
`git revert 2e4eec4b` (a re-merge of the same branch would otherwise be a no-op, because git still
counts `3c361e8f` as an ancestor). Cutting an unreleased tree is free; flipping a pair onto one is
not.

### The piece size is a pair setting, not an image default

`pairs.conf` gained a `ws_rows` column (both pairs 1024). `pair_load` turns it into `WS_ENV`
(`SEAM_WS_ROWS=1024 SEAM_WS_ROWS_AUTO=0`), and both places that start a server -- the smoke and
the failed-update restore -- pass `$WS_ENV` inside `SEAM_ENV`, so no start can drift to whatever
the image happens to default to. `ws_rows=auto` is the escape hatch: it sends nothing and leaves
the choice to the released server binary.

Why pin at all: 1024 rows is the shape every number on this line was measured on. 512-row pieces
cost -7%/-22% (32k/65k tok/s) against `1d5d49f7`; a single 4096-row piece runs 224-326 ms p50 with
3-5 s tails. 2048 would be 1.4x faster on the GX cold-expert path (115 ms vs 164 ms for 8x1024)
but wants ~1 GiB of GX headroom and neither box has it while paging. A picker may adapt to that on
its own, which is exactly why `AUTO=0` is the other half of the pin: the pair has to run the shape
its numbers were taken on, not the one the server judges affordable this minute.

### The pack lives where the box serves it, proven by device, and the gate is binding

The P7 containers were given `/models` (a directory on the 1.9 TB root disk) when the 4 TB pool is
`/models/dsv41`; the pack was reachable only because the 12 TB drive happened to be mounted under
that tree. Both P7 specs now mount the pool itself (`/models/dsv41:/models/dsv41`), and
`pair_pack_disk.py` no longer trusts a path prefix: it probes the pool and compares `st_dev` with the
pack, the placement file and every link target, refusing a pack that is merely *under* the pool path
on another device (`p7:pack:… is on device 66305, but /models/dsv41 is device 66306`). On the GX side
`/cold` must share the home directory's device, so a `/cold` that resolves into `/mnt` (P7's disk over
NFS) is refused even when the string looks local. `/mnt` in any spec is refused by the loader
(`SPEC_REFUSED`, exit 2) and reported, not thrown, by the checker (`PACK_DISK_FAIL`, rc 3); the plan
renders a removal as `mount:-<dst>=<src> (not in spec)`, a wrong source as `~`, an absent wanted
mount as `+…absent`, so dropping the 12 TB mount can never read like adding it.

The GX gate is binding for every launch, not only the smoke that takes it: `seam_server_ensure.sh`
calls `seam_gate_check` before it touches the box and refuses with rc 3, naming the state
(`gate_free`, `gate_bad`, `gate_expired`, `gate_stale`, `gate_foreign`) and the owner/pid/until. It
only ever *checks* a self or ancestor claim — a permanent claim by a restarted server would deadlock
the next ticket — and `pair_gx_gate_state` now treats a claim whose pid is gone as free. The dispatch
passes the real status through (`exit $?`), so a box-busy refusal no longer arrives as
`server_timeout_or_died` after 900 s. The preflight names the holder as `server_in=<ct>:<pid>` and
`mem_in=<ct> <MiB>` (`held_by=… top_mem=…`), and checks `gx-busy` before `gx-mem`: on 2026-11-02 the
memory line fired first and blamed a full box when the pack and port belonged to `seam-gx2-4460`.

The gate only binds launches that go through `seam_server_ensure.sh`, and three scripts in SeamRobust's
tree start `worker_seam_server.py` directly (`dev_trace_decode.sh`, `seam_interleave_ab.sh`,
`prod_rows_probe.sh`). So the launch carries a marker only this script sets: the ensure passes
`-e SEAM_GATE_OK=1` on every docker exec, and the worker (in the seam tree, landing after
`9fc3ca8a`) refuses to start without it, before it touches the pack or CUDA. A direct launch then
fails immediately on the box instead of loading 113 GiB onto somebody else's claim; a release test
asserts the marker is set here and nowhere else in the tree.

The ticket's own restore path had the same hole from the other side: after a failed update it starts the
previous release's server to put the box back the way it found it, and with the gate binding it would
now be refused `gate_free`. `restore()` therefore claims the box itself (`restore-<pair>`, 15 min, no
wait) before starting that server and drops the claim after; if somebody else holds the box it records
`RESTORE_SERVER_SKIP` and leaves their load alone instead of queueing 113 GiB behind it.

A container create invalidates live CUDA/UVM handles outside the new container, so it is not a local
act even when it names one card. On 2026-11-02 02:22 the dev recreate's create killed a client arm in an
ad-hoc container on the same P7 GPU ("Can't initialize NVML: Unknown Error"; host healthy, and the
container recovered by itself, so no restart was needed). Measured cause: not a module reload and not
a GPU reset -- kern.log has no NVRM line, no module load and no Xid at that second, and the `nvidia_uvm`
refcount is unchanged -- it is the runtime's device hook (runc + libnvidia-container `configure` on the
host's single UVM node) tearing down the UVM state the other container held. Hence the damage reaches
every P7 GPU, including GPU2, which is not ours. The recreate branch now asks all three
(`pair_p7_gpu_users [index...]`, `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name` mapped to
index by UUID) and refuses `gpu_busy p7_gpus=0,1 users=<pid name;...>` before it creates or renames
anything, i.e. before the flip; processes inside the pair's own container are exempt
(`pair_pid_in_container`), `PAIR_GPU_WAIT_MIN` (default 0) covers an arm that is already ending, and
GPU2's users are reported as `UPDATE_GPU2_IN_USE` without ever being waited on or touched.
