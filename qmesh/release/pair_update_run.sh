#!/usr/bin/env bash
# pair_update_run.sh -- move one pair onto a release, under the pair's queue, and verify it there.
#
#   pair_update_run.sh <pair> [sha]
#
# Run by the queue as agent PairUpdate (see pair_update.sh); running it by hand is allowed only with
# the GPU idle. Phases, in this order, each stopping the job on real failure:
#
#   1. cut      the release exists on P7 and on the pair's GX box (git archive, same path both sides)
#   2. containers  recreate any container whose wiring differs from qmesh/release/containers/*.json
#               (mounts /pair + /src/exllamav3 -> /pair/current/src); the old one is kept as .bak-<ts>
#   3. libs     native seam lib for both arches, built from the release tree, stamped beside the .so
#   4. ext      torch JIT cache for the release, built from the release tree (not from `current`)
#   5. flip     current -> releases/<sha> on both sides, and the in-container symlink
#   6. smoke    start the server from that release, check GX memory, run one decode row
#   7. manifest write pairs/<pair>.json, then pair_check must return 0 -- the same check the queue
#               guard runs before it starts anybody's job
#
# Nothing here is allowed to stop a container or the server on interruption: a wrapper that did that
# once outlived its job and took down the next arm's server. The TERM/INT trap flips the symlink back
# (so the next job sees the release it saw before) and exits; explicit failures are rolled back by
# hand in restore().
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_update_run.sh <pair> [sha]}
SHA_ARG=${2:-}
pair_load "$PAIR_ARG"

# The script that actually flips checks the canary too, so queueing it directly is not a way around
# pair_update.sh/pair_promote.sh. It needs the sha up front: dev evidence is recorded per sha, and a
# prod flip of "whatever current happens to point at" is exactly the drift this exists to stop.
if [ "$PAIR" = prod ] && [ "${PAIR_DRY:-0}" != 1 ]; then
  [ -n "$SHA_ARG" ] || pair_die "prod needs an explicit sha: dev canary evidence is recorded per sha" \
    "(use scripts/pair_promote.sh <sha>)"
  canary=$(pair_canary_gate "$SHA_ARG" "$PAIR_STATE/dev/evidence/$SHA_ARG.json")
  [ "$canary" = ok ] || pair_die "prod refused: dev canary for ${SHA_ARG:0:12} does not qualify: " \
    "$canary (use scripts/pair_promote.sh <sha>)"
  pair_step "UPDATE_CANARY pair=prod sha=${SHA_ARG:0:12} dev_evidence=$PAIR_STATE/dev/evidence/$SHA_ARG.json ok"
fi

mkdir -p "$PAIR_STATE_DIR/logs"
PAIR_LOG=${PAIR_LOG:-$PAIR_STATE_DIR/logs/update-$PAIR-$(date -u +%Y%m%dT%H%M%SZ).log}
export PAIR_LOG
ln -sfn "$(basename "$PAIR_LOG")" "$PAIR_STATE_DIR/logs/latest-update-$PAIR.log" 2>/dev/null || true
PHASE=start; TS=""; OLD_SHA=""; FLIPPED=""; VERIFIED=""
# PAIR_DRY only: the stub harness sets PAIR_DRY_FAIL_AT=<phase> to force a failure there and exercise
# the restore path. A real run never sets PAIR_DRY, so this can not fire on the boxes.
dry_gate() {
  if [ "${PAIR_DRY:-0}" = 1 ] && [ "${PAIR_DRY_FAIL_AT:-}" = "$1" ]; then
    fail "PAIR_DRY_FAIL_AT=$1 (harness-injected)"
  fi
  return 0
}
phase() { PHASE=$1; dry_gate "$1"; }
status_write() {  # <eta_min> <doing>
  [ -n "${GPU_RUN_STATUS:-}" ] && echo "eta_min=$1 doing=$2" > "$GPU_RUN_STATUS" 2>/dev/null
  return 0
}
# The release's evidence: what a promotion (pair_promote.sh, and the prod gate in pair_update.sh)
# reads to decide whether this pair may be copied to the other one. It is written per sha, not per
# pair, because the pair's manifest moves on with the next release while a promotion may come later.
#   evidence <ok|failed> <why>
evidence() {
  local ok=$1 why=${2:-}
  [ "${PAIR_DRY:-0}" = 1 ] && return 0
  python3 - "$PAIR_STATE_DIR/evidence/${SHA:-none}.json" "$PAIR" "${SHA:-}" "${BUILD_ID:-}" \
           "$ok" "$why" "$PHASE" "${PAIR_LOG:-}" "${SMOKE_ROW:-}" "${SMOKE_IDS:-}" \
           "$P7CT" "$GXCT" "${REL:-}" <<'PY' 2>&1 | sed 's/^/  /' | tee -a "${PAIR_LOG:-/dev/null}"
import datetime, json, os, sys
path, pair, sha, build_id, ok, why, phase, log, row, ids, p7ct, gxct, rel = sys.argv[1:14]
if not sha:
    print("EVIDENCE_SKIP no release sha yet")
    sys.exit(0)
seam = ""
try:
    seam = json.load(open(os.path.join(rel, "RELEASE.json"))).get("seam", {}).get("fingerprint", "")
except Exception:                                                       # noqa: BLE001
    pass
ids_doc = {}
for tok in ids.split():
    k, _, v = tok.partition("=")
    if k == "n":
        ids_doc["n"] = int(v)
    elif k == "sha256":
        ids_doc["sha256"] = v
doc = {
    "format": 1,
    "pair": pair,
    "release_sha": sha,
    "build_id": build_id,
    "seam": seam,
    "ok": ok == "ok",
    "failed_at_phase": "" if ok == "ok" else phase,
    "why": why,
    "smoke": {"row": row, "ids": ids_doc},
    "containers": {"p7": p7ct, "gx": gxct},
    "log": log,
    "written_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}
os.makedirs(os.path.dirname(path), exist_ok=True)
tmp = f"{path}.tmp.{os.getpid()}"
json.dump(doc, open(tmp, "w"), indent=2, sort_keys=True)
open(tmp, "a").write("\n")
os.replace(tmp, path)
print(f"EVIDENCE {path} pair={pair} sha={sha[:12]} ok={doc['ok']} ids={ids_doc.get('n', 0)}")
PY
}
# Who else is on this pair: tickets in its queue (not our own) or a live client/server process in
# either of its containers. Recreating a container SIGKILLs an arm someone else is running (2026-11-01
# 09:39: an 12-minute prefill arm died with CLIENT_RC=137 and no traceback), so this refuses instead.
# Only processes matter: the queue already serialises tickets, so a queued ticket behind ours can not
# race us, but an arm exec'd into the container (or a server still up) will be killed by the recreate.
pair_others_busy() {
  local n
  n=$(docker exec "$P7CT" pgrep -fc 'worker_seam_server|pair_arm|pair_p7_smoke|qes' 2>/dev/null)
  if [ "${n:-0}" != 0 ]; then printf 'p7_procs=%s ' "$n"; fi
  n=$(pair_gxsh "docker exec $GXCT pgrep -fc 'worker_seam_server|pair_arm|qes' 2>/dev/null")
  if [ "${n:-0}" != 0 ]; then printf 'gx_procs=%s ' "$n"; fi
}
gpu_free() {
  local used
  used=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU" 2>/dev/null)
  [ -z "$used" ]
}
on_term() {
  pair_step "PAIR_UPDATE_INTERRUPTED pair=$PAIR phase=$PHASE"
  status_write 1 "interrupted"
  # Only the symlink: cheap, and it keeps the next job on the release it was checked against.
  if [ -n "$FLIPPED" ] && [ -n "$OLD_SHA" ] && [ -z "$VERIFIED" ]; then
    pair_step "RESTORE_LINK pair=$PAIR -> ${OLD_SHA:0:12} (interrupted before verification)"
    python3 "$HERE/pair_containers.py" link "$PAIR" "$OLD_SHA" >/dev/null 2>&1 \
      || pair_step "RESTORE_LINK_FAILED pair=$PAIR old=${OLD_SHA:0:12}"
  fi
  [ -n "$TS" ] && echo "recreated at ts=$TS; undo with: $HERE/pair_containers.py rollback $PAIR $TS" \
      > "$PAIR_STATE_DIR/restore-$PAIR-$TS.txt"
  exit 143
}
trap on_term TERM INT

restore() {  # only on explicit failure, never from a trap
  pair_step "RESTORE_BEGIN pair=$PAIR phase=$PHASE"
  if [ -n "$FLIPPED" ] && [ -n "$OLD_SHA" ]; then
    python3 "$HERE/pair_containers.py" link "$PAIR" "$OLD_SHA" >/dev/null 2>&1 \
      && pair_step "RESTORE_LINK pair=$PAIR -> ${OLD_SHA:0:12}" \
      || pair_step "RESTORE_LINK_FAILED pair=$PAIR old=${OLD_SHA:0:12}"
  fi
  if [ -n "$TS" ]; then
    python3 "$HERE/pair_containers.py" rollback "$PAIR" "$TS" 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
    pair_step "RESTORE_CONTAINERS pair=$PAIR ts=$TS"
  fi
  if [ "${SERVER_BEFORE:-0}" = 0 ]; then
    pair_step "RESTORE_SERVER_SKIP pair=$PAIR (no server was serving before this update)"
  elif [ -n "$OLD_SHA" ] && [ -d "$(pair_release_dir "$OLD_SHA")" ]; then
    local old_build
    old_build=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get("build_id",""))' \
                "$(pair_release_dir "$OLD_SHA")/RELEASE.json" 2>/dev/null)
    pair_step "RESTORE_SERVER pair=$PAIR build=${old_build:-unknown}"
    GX=$GX CT=$GXCT GX_DEV=${GX_DEV:-rocep1s0f0} GX_GID=$GX_GID PORT=$PORT SERVER_BUCKETS=$BUCKETS \
      SEAM_SERVER_TIMEOUT=${RESTORE_TIMEOUT:-600} SEAM_ENV="SEAM_BUILD_ID=$old_build" \
      "$PAIR_SCRIPTS/seam_server_ensure.sh" --fresh --buckets "$BUCKETS" 2>&1 \
      | grep -E '^(SEAM_ENSURE|SEAM_SLACK|SEAM_SERVER_READY)' | sed 's/^/  /' | tee -a "$PAIR_LOG"
    pair_step "RESTORE_SERVER_DONE pair=$PAIR rc=${PIPESTATUS[0]}"
  fi
  pair_step "RESTORE_END pair=$PAIR"
}
fail() {
  pair_step "PAIR_UPDATE_FAILED pair=$PAIR phase=$PHASE why=$*"
  status_write 1 "rollback ($PHASE)"
  # The failed attempt is evidence too: a later promotion of this sha must see it and say why.
  [ -n "${SHA:-}" ] && evidence failed "$*"
  restore
  exit 1
}

pair_step "UPDATE_START pair=$PAIR gpu=$GPU ct=$P7CT/$GXCT root=$PAIR_ROOT log=$PAIR_LOG"
if ! gpu_free && [ "${PAIR_UPDATE_FORCE:-0}" != 1 ]; then
  pair_step "PAIR_UPDATE_REFUSED pair=$PAIR gpu_busy (GPU $GPU has a compute app; queue it instead)"
  exit 3
fi
for t in "$PAIR_QUEUE"/gpu_queue/[0-9]*-*; do
  [ -e "$t" ] || continue
  base=$(basename "$t")
  case "$base" in *-PairUpdate|*-PairUpdateRun) continue ;; esac
  pair_step "UPDATE_QUEUE_NOTE pair=$PAIR other_ticket=$base (queued behind us; the queue serialises)"
done
others=$(pair_others_busy)
if [ -n "$others" ] && [ "${PAIR_UPDATE_FORCE:-0}" != 1 ]; then
  pair_step "PAIR_UPDATE_REFUSED pair=$PAIR pair_busy $others (a client or ticket owns this pair; wait for it)"
  exit 3
fi
[ -n "$others" ] && pair_step "UPDATE_FORCED pair=$PAIR pair_busy $others (PAIR_UPDATE_FORCE=1)"
# Whether this pair was actually serving before the update decides what the restore has to put back.
# A pair that was idle (the normal case: the busy check above refuses a serving one) must be left
# idle on failure -- restarting a server for a release that never served costs a full 113 GiB pack
# load on GX2, i.e. minutes of the very box the smoke just proved is memory-tight.
SERVER_BEFORE=$(pair_gxsh "docker exec $GXCT pgrep -fc worker_seam_server.py 2>/dev/null" | tail -1)
SERVER_BEFORE=${SERVER_BEFORE:-0}
pair_step "UPDATE_SERVER_BEFORE pair=$PAIR serving=$SERVER_BEFORE"
status_write 120 "cutting release"

# ---- 1..4: release, containers, libs, ext --------------------------------------------------------
# A flip-only ticket is what pair_update.sh enqueues after `pair_prepare.sh <pair> <sha>` has already
# cut, wired and built the release outside the queue. Nothing is built here: the stamps must already
# name this exact sha and build, and the containers must already be wired -- otherwise the ticket
# stops before the flip and says which prepare to run, instead of holding the GPU for an hour of JIT.
if [ "${PAIR_FLIP_ONLY:-0}" = 1 ]; then
  phase prepared
  [ -n "$SHA_ARG" ] || fail "PAIR_FLIP_ONLY needs an explicit sha (nothing would be prepared)"
  SHA=$SHA_ARG
  REL=$(pair_release_dir "$SHA")
  [ -d "$REL/src" ] || fail "release ${SHA:0:12} is not cut at $REL (run scripts/pair_prepare.sh $PAIR $SHA)"
  BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json") \
    || fail "release ${SHA:0:12} has no build_id (run scripts/pair_prepare.sh $PAIR $SHA)"
  pair_step "UPDATE_RELEASE pair=$PAIR sha=${SHA:0:12} build=$BUILD_ID (prepared out of band)"
  plan_out=$(python3 "$HERE/pair_containers.py" plan "$PAIR" 2>&1)
  printf '%s\n' "$plan_out" | sed 's/^/  /' | tee -a "$PAIR_LOG" >/dev/null
  case "$plan_out" in
    *RECREATE*) fail "containers are not prepared: run scripts/pair_prepare.sh $PAIR $SHA first" ;;
  esac
  pair_step "UPDATE_CONTAINERS pair=$PAIR unchanged (prepared)"
  lib_out=$("$HERE/pair_build_lib.sh" verify "$PAIR" "$SHA" --side both 2>&1); lib_rc=$?
  printf '%s\n' "$lib_out" | sed 's/^/  /' | tee -a "$PAIR_LOG"
  [ "$lib_rc" = 0 ] || fail "lib verify rc=$lib_rc (run scripts/pair_prepare.sh $PAIR $SHA)"
  ext_out=$("$HERE/pair_build_ext.sh" verify "$PAIR" "$SHA" --side both 2>&1); ext_rc=$?
  printf '%s\n' "$ext_out" | sed 's/^/  /' | tee -a "$PAIR_LOG"
  [ "$ext_rc" = 0 ] || fail "ext verify rc=$ext_rc (run scripts/pair_prepare.sh $PAIR $SHA)"
else
# ---- 1. cut -------------------------------------------------------------------------------------
phase cut
cut_out=$("$HERE/pair_cut.sh" "$PAIR" $SHA_ARG 2>&1)
cut_rc=$?
printf '%s\n' "$cut_out" | sed 's/^/  /' | tee -a "$PAIR_LOG"
[ "$cut_rc" = 0 ] || fail "cut rc=$cut_rc"
SHA=$(printf '%s\n' "$cut_out" | tail -1)
[ -n "$SHA" ] && [ -d "$(pair_release_dir "$SHA")/src" ] || fail "cut produced no release dir"
REL=$(pair_release_dir "$SHA")
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json") \
  || fail "release ${SHA:0:12} has no build_id"
pair_step "UPDATE_RELEASE pair=$PAIR sha=${SHA:0:12} build=$BUILD_ID"

# ---- 2. containers ------------------------------------------------------------------------------
phase containers
status_write 110 "checking container wiring"
TS=$(date -u +%Y%m%dT%H%M%SZ)
plan_out=$(python3 "$HERE/pair_containers.py" plan "$PAIR" 2>&1)
printf '%s\n' "$plan_out" | sed 's/^/  /' | tee -a "$PAIR_LOG" >/dev/null
case "$plan_out" in
*RECREATE*)
  status_write 100 "recreating containers (ts=$TS)"
  python3 "$HERE/pair_containers.py" apply "$PAIR" "$SHA" --ts "$TS" 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
  rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || fail "containers apply rc=$rc"
  pair_step "UPDATE_CONTAINERS pair=$PAIR recreated ts=$TS"
  ;;
*)
  TS=""
  pair_step "UPDATE_CONTAINERS pair=$PAIR unchanged"
  ;;
esac

# ---- 3. native libs -----------------------------------------------------------------------------
phase lib
status_write 85 "building seam lib (x86_64 + aarch64)"
"$HERE/pair_build_lib.sh" "$PAIR" "$SHA" --side both 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || fail "lib build rc=$rc"

# ---- 4. torch ext -------------------------------------------------------------------------------
phase ext
status_write 60 "building torch ext (P7, then GX)"
"$HERE/pair_build_ext.sh" "$PAIR" "$SHA" --side both 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || fail "ext build rc=$rc"

fi

# ---- 5. flip ------------------------------------------------------------------------------------
phase flip
OLD_SHA=$(pair_current_sha "$PAIR_ROOT")
GX_OLD=$(pair_current_sha "$PAIR_ROOT" 1)
[ "$OLD_SHA" = "$GX_OLD" ] || pair_step "UPDATE_NOTE old releases differ p7=${OLD_SHA:0:12} gx=${GX_OLD:0:12}; restoring ${OLD_SHA:0:12} on both"
[ -n "$OLD_SHA" ] && pair_step "UPDATE_FLIP from=${OLD_SHA:0:12} to=${SHA:0:12}" || pair_step "UPDATE_FLIP from=- to=${SHA:0:12} (first release)"
status_write 30 "flipping current -> ${SHA:0:12}"
python3 "$HERE/pair_containers.py" link "$PAIR" "$SHA" 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || fail "link flip rc=$rc"
FLIPPED=1
python3 "$HERE/pair_containers.py" inner "$PAIR" 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || fail "inner link rc=$rc"

# ---- 6. smoke -----------------------------------------------------------------------------------
phase smoke
status_write 20 "server start + one decode row"
smoke_out=$("$HERE/pair_smoke.sh" "$PAIR" "$SHA" 2>&1)
smoke_rc=$?
printf '%s\n' "$smoke_out" | sed 's/^/  /' | tee -a "$PAIR_LOG"
[ "$smoke_rc" = 0 ] || fail "smoke rc=$smoke_rc $(printf '%s' "$smoke_out" | grep -m1 SMOKE_FAIL)"
row=$(printf '%s\n' "$smoke_out" | sed -n 's/^SMOKE_ROW //p' | head -1)
SMOKE_ROW=$row
# n= and sha256= of the ids the arms generated (total across arms, and a commitment over every
# ARM_IDS line); the promotion gate reads this, not the text. Fall back to the first arm's line for
# a smoke that predates the total.
SMOKE_IDS=$(printf '%s\n' "$smoke_out" | sed -n 's/^SMOKE_ARM_IDS_TOTAL n=\([0-9]*\) sha256=\([0-9a-f]*\).*/n=\1 sha256=\2/p' | head -1)
[ -n "$SMOKE_IDS" ] || SMOKE_IDS=$(printf '%s\n' "$smoke_out" | sed -n 's/^SMOKE_ARM_IDS .* n=\([0-9]*\) sha256=\([0-9a-f]*\).*/n=\1 sha256=\2/p' | head -1)
pair_step "UPDATE_SMOKE pair=$PAIR $row ${SMOKE_IDS:-ids=none}"

# the smoke row is the release's reference row: same release must reproduce it (pair_ref.sh)
python3 - "$PAIR_STATE_DIR/refs-$SHA.json" "$SHA" "$BUILD_ID" "$REL" "$row" <<'PY' | tee -a "$PAIR_LOG"
import datetime, json, sys
path, sha, build_id, rel, row = sys.argv[1:6]
kv = dict(tok.split("=", 1) for tok in row.split() if "=" in tok)
seam = json.load(open(f"{rel}/RELEASE.json")).get("seam", {}).get("fingerprint")
doc = {"format": 1, "release_sha": sha, "build_id": build_id, "seam": seam,
       "rows": [{"i": int(kv.get("i", 0)), "text_sha256": kv.get("text_sha256", "")}],
       "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
json.dump(doc, open(path, "w"), indent=2, sort_keys=True)
open(path, "a").write("\n")
print(f"REF {path} rows=1")
PY

# ---- 7. manifest + the guard's own check --------------------------------------------------------
if [ "$PAIR_DRY" = 1 ]; then
  # the dry builders write no stamps on purpose, so a manifest here would be fiction
  pair_step "DRY MANIFEST pair=$PAIR sha=$SHA (skipped: no stamps outside a real build)"
  pair_step "DRY VERIFY pair=$PAIR (skipped: nothing to verify against)"
  VERIFIED=1
  pair_step "PAIR_UPDATED_DRY pair=$PAIR sha=$SHA build=$BUILD_ID (dry: no manifest, no verify)"
  echo "$SHA"
  exit 0
fi
phase manifest
status_write 8 "writing manifest and verifying"
python3 "$HERE/pair_manifest.py" write "$PAIR" "$SHA" --ref "$PAIR_STATE_DIR/refs-$SHA.json" \
  --log "$PAIR_LOG" --by "PairUpdate" 2>&1 | sed 's/^/  /' | tee -a "$PAIR_LOG"
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || fail "manifest write rc=$rc"

phase verify
check_out=$("$HERE/pair_check.sh" "$PAIR" 2>&1)
check_rc=$?
printf '%s\n' "$check_out" | sed 's/^/  /' | tee -a "$PAIR_LOG"
[ "$check_rc" = 0 ] || fail "post-update check rc=$check_rc"
VERIFIED=1
status_write 1 "done"

# Evidence last, so the file only exists once the pair is really on the release and verified.
evidence ok
pair_step "PAIR_UPDATED pair=$PAIR sha=$SHA build=$BUILD_ID check=IN_SYNC ${SMOKE_IDS:-ids=none}"
echo "$SHA"
