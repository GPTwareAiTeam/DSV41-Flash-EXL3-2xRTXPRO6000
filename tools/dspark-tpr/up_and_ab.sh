#!/bin/bash
# up_and_ab.sh -- one ticket: bring the dev pair up, gate it, run the K A/B on the server it started.
#
# One ticket on purpose. The A/B's whole claim is that the two passes ran against the same server, same
# release, same bucket grant; a sibling ticket landing between "start the server" and "measure" would
# restart the worker under the measurement and the numbers would be of a different pair. Serving and
# measuring inside one ticket is what makes that impossible.
#
# Usage: run_dev.sh DSparkTune 45 bash up_and_ab.sh [tag]
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
S=/home/treadall/recipe-artifacts/gpu0-gx1/scripts
TAG=${1:-k35}
LOG=$HERE/logs
mkdir -p "$LOG"
UP=$LOG/up_$(date -u +%Y%m%dT%H%M%SZ).log
echo "eta_min=8 doing=dev server load" > "${GPU_RUN_STATUS:-/dev/null}"
bash "$S/pair_serve.sh" dev > "$UP" 2>&1
rc=$?
grep -E '^(PAIR_SERVE_UP|PAIR_SERVE_MEM|PAIR_SERVE_FAILED|SEAM_ENSURE|SEAM_NO_)' "$UP"
[ "$rc" = 0 ] || { echo "AB_ABORT reason=serve_failed rc=$rc log=$UP" >&2; exit 3; }
echo "eta_min=5 doing=swap gate + worker age" > "${GPU_RUN_STATUS:-/dev/null}"
g=$HERE/gate_gx.sh
bash "$g" > "$LOG/gate_up.txt" 2>&1
cat "$LOG/gate_up.txt"
grc=$(sed -n 's/^gate_rc=\([0-9]*\)$/\1/p' "$LOG/gate_up.txt" | tail -1)
[ "${grc:-1}" = 0 ] || [ "${grc:-1}" = 2 ] || { echo "AB_ABORT reason=gate gate_rc=${grc:-?}" >&2; exit 4; }
bash "$HERE/ab_k35.sh" "$TAG"
