#!/usr/bin/env bash
# ab_arms.sh -- the release A/B as one ticket, so the two arms cannot be separated in time.
#
#   ab_arms.sh <ref_sha> <cand_sha>
#
# Two tickets (ref, then cand) are only adjacent if nothing else is queued between them, and tonight
# something always is. One ticket that runs both arms back to back, and the canary after them, keeps
# the pair inside a single host state: same session, same clocks, same box load, same caches.
#
# The ticket also lands the container-spec change of 2026-11-04 (TORCH_EXTENSIONS_DIR defaults to
# /pair/scratch-ext, so a stray `import exllamav3` cannot rewrite a release's compiled cache). The flip
# refuses to run while the live containers carry the old spec (PAIR_UPDATE_FAILED ... the container
# spec changed), so the recreate has to happen before arm 1 -- and it belongs inside the ticket, not
# beside it, or another agent's ticket grabs the slot in between.
set -uo pipefail
S=/home/treadall/recipe-artifacts/gpu0-gx1/scripts
. "$S/pair_lib.sh"
REF=${1:?usage: ab_arms.sh <ref_sha> <cand_sha>}
CAND=${2:?usage: ab_arms.sh <ref_sha> <cand_sha>}
pair_load dev
say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
st()  { [ -n "${GPU_RUN_STATUS:-}" ] && printf 'eta_min=%s doing=%s\n' "$1" "$2" > "$GPU_RUN_STATUS"; }

st 6 "A/B ref 35bfaecb vs cand 3593531b with the worker cache pinned to the release ext"
# No recreate here: the containers were recreated at 20261104T020235Z and both halves now carry the
# scratch-ext default (RECREATE_EFFECTIVE yes), so this ticket only measures. What changed since the
# 02:03Z attempt is seam_server_ensure.sh, which now pins the worker's cache to the release's ext for
# the release tree -- without it the worker JIT'ed into scratch, ate 3.4 GiB of MemAvailable, and the
# ref arm died SEAM_NO_MEMORY (and the cand arm imported a .so built inside the window).
p7env=$(docker exec "$P7CT" sh -c 'echo $TORCH_EXTENSIONS_DIR' 2>/dev/null)
say "ENV_PRE p7=${p7env:-?} (expect /pair/scratch-ext; the arms state the release cache explicitly)"

st 25 "A/B arm 1/2: ref ${REF:0:12} (flip + decode 256 + wiki2 quick8 ppl)"
bash /tmp/promote_arm.sh "$REF" ref; ref_rc=$?; say "AB_REF_RC $ref_rc"

st 25 "A/B arm 2/2: cand ${CAND:0:12} (same harness, same box)"
bash /tmp/promote_arm.sh "$CAND" cand; cand_rc=$?; say "AB_CAND_RC $cand_rc"

if [ "$ref_rc" = 0 ] && [ "$cand_rc" = 0 ]; then
  st 20 "canary on the candidate (warm-up lines, bucket grant, prefill ratio, HTTP)"
  PAIR_SMOKE_SERVER=1 bash "$S/pair_smoke.sh" dev; can_rc=$?
  say "CANARY_RC $can_rc"
else
  can_rc=skip; say "CANARY_SKIP arms_failed ref=$ref_rc cand=$cand_rc"
fi

now=$(pair_current_sha "$PAIR_ROOT" 2>/dev/null | cut -c1-12)
wl=$(pair_gxsh "docker exec $GXCT sh -c 'pgrep -x python3 -a | grep worker_seam_server[.]py | wc -l'" 2>/dev/null | tail -1)
say "AB_ALL_DONE ref=$REF ref_rc=$ref_rc cand=$CAND cand_rc=$cand_rc canary=$can_rc current=$now workers=$wl"
[ "$cand_rc" = 0 ] || exit 6
