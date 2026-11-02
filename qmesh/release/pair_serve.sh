#!/usr/bin/env bash
# pair_serve.sh -- load the pair's current release and leave its server serving.
#
#   queue/run_prod.sh ServerUp 20 "scripts/pair_serve.sh prod"
#   queue/run_dev.sh  ServerUp 20 "scripts/pair_serve.sh dev"
#
# The smoke and the update both start a server and both stop it when they end, so a pair that is not
# in the middle of a ticket has nothing serving -- and "the pair is down" is a state nobody chose. This
# is the one explicit way to reach the steady state instead of the middle of an operation: it loads
# from the release `current` names (never a working tree), records the worker in the pair's pidfile so
# a later ticket can stop exactly that process, and drops its box claim as soon as the load is done --
# the claim protects the load, the server outliving it is the pair's job, not a leak.
#
# It refuses when the release is not IN_SYNC: serving a tree whose artifact came from somewhere else
# is the ad-hoc delivery this whole system exists to end.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

pair_load "${1:?usage: pair_serve.sh <prod|dev>}"
SHA=${2:-$(pair_current_sha "$PAIR_ROOT")}
[ -n "$SHA" ] || pair_die "PAIR_SERVE_NO_RELEASE pair=$PAIR (no $PAIR_ROOT/current)"
REL=$(pair_release_dir "$SHA")
[ -d "$REL" ] || pair_die "PAIR_SERVE_NO_RELEASE pair=$PAIR sha=${SHA:0:12} dir=$REL"
[ -f "$REL/RELEASE.json" ] || pair_die "PAIR_SERVE_NO_RELEASE pair=$PAIR sha=${SHA:0:12} (no RELEASE.json)"
BUILD_ID=$(python3 -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("build_id",""))
except Exception: print("")' "$REL/RELEASE.json")
ENSURE=$PAIR_SCRIPTS/seam_server_ensure.sh
[ -x "$ENSURE" ] || pair_die "PAIR_SERVE_NO_ENSURE $ENSURE"
pair_step "PAIR_SERVE_START pair=$PAIR sha=${SHA:0:12} ct=$GXCT build=${BUILD_ID:-unknown}"

# IN_SYNC first: the sync check is the only thing that knows whether the on-disk artifact is the one
# the stamp names, and a server loading a mismatched extension is exactly the 08:03Z smoke.
chk=$("$PAIR_SCRIPTS/pair_check.sh" "$PAIR" 2>&1)
printf '%s\n' "$chk" | sed 's/^/  /'
case "$chk" in
  *DRIFT*|*MISSING*) pair_die "PAIR_SERVE_DRIFT pair=$PAIR (build it first: scripts/pair_build_ext.sh $PAIR --side p7|gx)" ;;
esac
case "$chk" in
  *"RESULT $PAIR P7 IN_SYNC"*"RESULT $PAIR GX IN_SYNC"*) ;;
  *) pair_die "PAIR_SERVE_NOT_IN_SYNC pair=$PAIR (pair_check did not report both halves IN_SYNC)" ;;
esac

if [ "$(pair_gx_foreign_server)" != "" ]; then
  pair_die "PAIR_SERVE_FOREIGN $(pair_gx_foreign_server) (one pack load per box)"
fi
if [ "$(pair_gxsh "docker exec $GXCT pgrep -fc '^python3 -u qmesh/worker_seam_server.py'" 2>/dev/null)" != 0 ]; then
  # Something already serves here. Say what, and let the caller decide to stop it first; silently
  # reusing it would mean the pair serves whatever the last ticket left loaded.
  pair_die "PAIR_SERVE_ALREADY $(pair_server_pidfile) $(cat "$(pair_server_pidfile)" 2>/dev/null)"
fi

pair_gx_gate_take "serve-$PAIR" $(( ${SERVE_TIMEOUT_MIN:-25} )) 0 \
  || pair_die "PAIR_SERVE_GX_BUSY $(pair_gx_gate_state)"

out=$(GX=$GX CT=$GXCT GX_DEV=${GX_DEV:-rocep1s0f0} GX_GID=$GX_GID PORT=$PORT \
      SEAM_GATE_DIR="$(dirname "$(pair_gx_gate_file)")" \
      SERVER_BUCKETS=$BUCKETS SEAM_SERVER_TIMEOUT=${SERVE_SERVER_TIMEOUT:-600} \
      SEAM_SLAB_SLOTS=${SERVE_SLAB_SLOTS:-1} \
      SEAM_ENV="$(pair_server_env "$BUILD_ID")" \
      "$ENSURE" --fresh --buckets "$BUCKETS" 2>&1)
rc=$?
[ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$out" >> "$PAIR_LOG"
printf '%s\n' "$out" | grep -E '^(SEAM_ENSURE|SEAM_SLACK|SEAM_NO_MEMORY|SEAM_NO_PACK|SEAM_SERVER)' | sed 's/^/  /'
if [ "$rc" != 0 ]; then
  pair_gx_gate_down
  pair_die "PAIR_SERVE_FAILED pair=$PAIR build=${BUILD_ID:-unknown} rc=$rc"
fi
pair_server_record "$BUILD_ID" || { pair_gx_gate_down; pair_die "PAIR_SERVE_NO_PID pair=$PAIR (the load returned 0 but no worker argv was found)"; }
pair_gx_gate_down
pair_step "PAIR_SERVE_UP pair=$PAIR sha=${SHA:0:12} build=${BUILD_ID:-unknown} $(cat "$(pair_server_pidfile)")"
