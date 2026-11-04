#!/bin/bash
# ppl_ab2.sh -- the wiki2 quick8 half of the 3cd80749 acceptance, alone.
#
# Why this exists: the 20261104T043901Z A/B measured both decode rows (ref 61.87, cand 61.60 tok/s at
# rows=256) and the cand ppl (2.0424 +/- 0.0414), but its ref ppl came back rc=127 because a comment I
# had inserted inside promote_arm's `docker exec` argument list ate the continuation. The release gate is
# already satisfied by the absolute check (2.0391 +/- 0.005); this buys the *paired* delta on the same
# session and harness, which is what stops a future reader from having to argue about a missing arm.
#
# Enqueued as agent `PairRelease-ppl` so it sorts last (rank 99): the dev pair is the only box and
# FusedGate-2's r7 has been waiting an hour for it.
set -uo pipefail
REF=35bfaecb59eb0f2a710c80b28b9ffeb670d2ff61
CAND=3cd80749f4dbcbdde1974812cbcda93c3d72d9d3
OUT=/home/treadall/recipe-artifacts/gpu0-gx1/pairs/dev/promote_ab
: "${GPU_RUN_STATUS:=/tmp/gpu_run_status_unset}"

say() { echo "$(date -u +%H:%M:%S) PPLAB $*"; }

say "START ref=${REF:0:12} cand=${CAND:0:12} mode=quality-only"
echo "eta_min=35 doing=ppl-only A/B ref+cand wiki2 quick8" >"$GPU_RUN_STATUS" 2>/dev/null

PROMOTE_SKIP_RATE=1 /tmp/promote_arm.sh "$REF" ref2
rc_ref=$?
say "REF_DONE rc=$rc_ref"
echo "eta_min=20 doing=ppl cand repeat after ref arm" >"$GPU_RUN_STATUS" 2>/dev/null

PROMOTE_SKIP_RATE=1 /tmp/promote_arm.sh "$CAND" cand2
rc_cand=$?
say "CAND_DONE rc=$rc_cand"

# The paired verdict, from the two json files the arms wrote (ppl is the only number either produced).
ppl_of() {
  python3 -c 'import json,sys
try:
    print("%.6f" % json.load(open(sys.argv[1]))["wiki2"]["ppl"])
except Exception:
    print("nan")' "$1"
}
ppl_ref=$(ppl_of "$OUT/ppl_ref2.json")
ppl_cand=$(ppl_of "$OUT/ppl_cand2.json")
delta=$(python3 -c 'import sys
try: print("%.4f" % (float(sys.argv[2]) - float(sys.argv[1])))
except Exception: print("nan")' "$ppl_ref" "$ppl_cand")
say "PPL_PAIRED ref=$ppl_ref cand=$ppl_cand delta=$delta tol=0.005"
case "$delta" in
  nan) say "PPL_PAIRED_UNJUDGED missing arm output" ;;
  *) python3 -c "import sys;exit(0 if abs(float(sys.argv[1]))<=0.005 else 1)" "$delta" \
       && say "PPL_PAIRED VERDICT=pass" || say "PPL_PAIRED VERDICT=FAIL" ;;
esac

current=$(basename "$(readlink -f /home/treadall/pairs/dev/current 2>/dev/null)" 2>/dev/null | cut -c1-12)
workers=$(ssh -o BatchMode=yes gx10-2 "docker exec seam-gx2-dev pgrep -x python3 -a 2>/dev/null | grep -c worker_seam_server" 2>/dev/null | tail -1)
say "PPLAB_DONE ref_rc=$rc_ref cand_rc=$rc_cand current=${current:-?} gx_workers=${workers:-?}"
[ "$rc_ref" = 0 ] && [ "$rc_cand" = 0 ] || exit 6
exit 0
