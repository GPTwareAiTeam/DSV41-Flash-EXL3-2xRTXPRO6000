#!/usr/bin/env bash
# pair_prepare.sh -- cut, wire and build a release for one pair, OUTSIDE its queue.
#
#   pair_prepare.sh <pair> [sha] [--side p7|gx|both]
#
# Why this exists: the torch JIT build of a release takes 5-13 minutes per side, and doing it inside
# the PairUpdate ticket kept the pair's GPU idle for that whole time. Everything expensive happens
# here instead -- niced, MAX_JOBS<=8, writing only into <root>/releases/<sha> -- while the pair keeps
# running whatever it is running. The ticket that follows (`pair_update.sh <pair> [sha] --flip-only`)
# only flips `current`, restarts the server, smokes and verifies: minutes, not an hour.
#
# What it does NOT do: it never moves `current`/`extcache` toward the new release (the one exception
# is the first-release bootstrap that container wiring needs to exist at all), never starts, stops or
# recreates a serving container, and never touches another pair's containers.
#
# Guards (all of them refuse, none of them "work around"):
#   * a container of this pair that is serving (worker_seam_server.py inside it) -- the build runs in
#     that container and a rebuild under a live server has already SIGBUSed a box,
#   * a container name that belongs to another pair in pairs.conf, or is named by a live ticket,
#   * an existing container whose spec differs: creating containers is allowed, recreating is not,
#   * GX MemAvailable below GX_BUILD_AVAIL_GIB (checked by the builders themselves).
#
# Prints, in order: CUT / PLAN / CREATE|EXISTS / LINK_UNPROMOTED? / LIB_* / EXT_* and finally
#   PREPARED <pair> sha=<sha> build=<build_id> sides=<sides>
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_prepare.sh <pair> [sha] [--side p7|gx|both]}
SHA_ARG=${2:-}
[ "${SHA_ARG:-}" = "--side" ] && SHA_ARG=""
SIDES=both
while [ $# -gt 0 ]; do
  case "$1" in
    --side) SIDES=${2:?--side needs p7|gx|both}; shift ;;
    -*) pair_die "unknown flag $1" ;;
  esac
  shift
done
pair_load "$PAIR_ARG"

# ---- guards --------------------------------------------------------------------------------------
python3 "$HERE/pair_containers.py" guard "$PAIR" || exit 2
for side in p7 gx; do
  ct=$([ "$side" = p7 ] && echo "$P7CT" || echo "$GXCT")
  running=$([ "$side" = p7 ] && docker inspect -f '{{.State.Running}}' "$ct" 2>/dev/null \
                                || pair_gxsh "docker inspect -f '{{.State.Running}}' $ct 2>/dev/null")
  [ "$running" = true ] || continue
  n=$([ "$side" = p7 ] && docker exec "$ct" pgrep -fc worker_seam_server.py 2>/dev/null \
                       || pair_gxsh "docker exec $ct pgrep -fc worker_seam_server.py 2>/dev/null")
  [ "${n:-0}" = 0 ] || pair_die "PREPARE_REFUSED $side $ct is serving ($n worker_seam_server.py): " \
      "prepare needs a container of this pair that is not serving"
done

# ---- pack locality (refuse before cutting anything onto a wrong disk) ----------------------------
# Owner rule 2026-11-02: an expert pack lives on the local drive of the box that serves it. A pack
# reached over NFS (or through a symlink that lands on someone else's drive) makes serving speed
# depend on the network, so it is refused here -- before a release is cut or a library built -- and
# again by the smoke, which is the last gate before `current` moves.
pack_out=$(python3 "$HERE/pair_pack_disk.py" "$PAIR" 2>&1); pack_rc=$?
printf '%s\n' "$pack_out" | sed 's/^/  /'
[ "$pack_rc" = 0 ] || pair_die "PREPARE_FAILED $PAIR phase=packs rc=$pack_rc (a pack is not on the serving box's own disk)"

# ---- 1. cut (or reuse an existing release of this sha) -------------------------------------------
cut_out=$("$HERE/pair_cut.sh" "$PAIR" $SHA_ARG 2>&1)
cut_rc=$?
printf '%s\n' "$cut_out" | sed 's/^/  /'
[ "$cut_rc" = 0 ] || pair_die "PREPARE_FAILED $PAIR phase=cut rc=$cut_rc"
SHA=$(printf '%s\n' "$cut_out" | tail -1)
REL=$(pair_release_dir "$SHA")
[ -n "$SHA" ] && [ -d "$REL/src" ] || pair_die "PREPARE_FAILED $PAIR phase=cut no_release_dir"
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json") \
  || pair_die "PREPARE_FAILED $PAIR phase=cut no_build_id"
pair_step "PREPARE_RELEASE pair=$PAIR sha=${SHA:0:12} build=$BUILD_ID"

# ---- 2. containers (create-only) -----------------------------------------------------------------
plan_out=$(python3 "$HERE/pair_containers.py" plan "$PAIR" 2>&1)
printf '%s\n' "$plan_out" | sed 's/^/  /'
case "$plan_out" in
*CREATE*|*RECREATE*)
  # A pair that has never run (prod on 2026-11-02) has no containers at all: the plan says CREATE and
  # apply makes them. Saying "unchanged" there left the release cut and the lib unbuilt, because the
  # build runs inside those containers. RECREATE still needs the ticket's PAIR_ALLOW_RECREATE, and the
  # apply refuses that here on its own.
  why=$(printf '%s\n' "$plan_out" | grep -o 'CREATE\|RECREATE' | head -1)
  ts=$(date -u +%Y%m%dT%H%M%SZ)
  python3 "$HERE/pair_containers.py" apply "$PAIR" "$SHA" --ts "$ts" 2>&1 | sed 's/^/  /'
  rc=${PIPESTATUS[0]}
  [ "$rc" = 0 ] || pair_die "PREPARE_FAILED $PAIR phase=containers rc=$rc ($why: creating needs the " \
      "containers to be absent, recreating needs PAIR_ALLOW_RECREATE=1 inside a queued PairUpdate)"
  pair_step "PREPARE_CONTAINERS pair=$PAIR $why wired ts=$ts"
  ;;
*)
  pair_step "PREPARE_CONTAINERS pair=$PAIR unchanged"
  ;;
esac

# ---- 3. native lib -------------------------------------------------------------------------------
pair_step "PREPARE_LIB pair=$PAIR sides=$SIDES"
"$HERE/pair_build_lib.sh" "$PAIR" "$SHA" --side "$SIDES" 2>&1 | sed 's/^/  /'
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || pair_die "PREPARE_FAILED $PAIR phase=lib rc=$rc"

# ---- 4. torch ext --------------------------------------------------------------------------------
pair_step "PREPARE_EXT pair=$PAIR sides=$SIDES"
"$HERE/pair_build_ext.sh" "$PAIR" "$SHA" --side "$SIDES" 2>&1 | sed 's/^/  /'
rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] || pair_die "PREPARE_FAILED $PAIR phase=ext rc=$rc"

pair_step "PREPARED pair=$PAIR sha=$SHA build=$BUILD_ID sides=$SIDES"
echo "$SHA"
