#!/usr/bin/env bash
# pair_smoke.sh -- prove a pair can serve one decode row on the release it is wired to.
#
#   pair_smoke.sh <pair> [sha]
#
# What it does, in order, with a bound on every wait:
#   1. records the GX box's memory before the load (MemAvailable, swap),
#   2. starts the seam server from the release (seam_server_ensure --fresh), with SEAM_BUILD_ID set so
#      the SEAM_SERVER_LOADED line names the release that is actually running,
#   3. records the memory *after* the pack is loaded and the slack the ensure script reports -- on
#      GX2 a 113.5 GiB pack leaves ~1 GiB, and a server that dies of that has to be caught here, not
#      in the middle of someone's arm,
#   4. runs one decode row through the P7 client half and checks it produced tokens and that the wire
#      did not answer `peer layout mismatch` / `remote invalid request`,
#   5. prints SMOKE_OK or SMOKE_FAIL <reason> and exits 0/1.
#
# A server that never prints READY is reported as `server_timeout` (never waited on indefinitely):
# the bound comes from SMOKE_SERVER_TIMEOUT and is enforced by seam_server_ensure itself.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_smoke.sh <pair> [sha]}
pair_load "$PAIR_ARG"
SHA=${2:-$(pair_current_sha "$PAIR_ROOT")}
[ -n "$SHA" ] || { echo "SMOKE_FAIL no_release"; exit 1; }
REL=$(pair_release_dir "$SHA")
[ -d "$REL/src" ] || { echo "SMOKE_FAIL release_absent $REL"; exit 1; }
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json" 2>/dev/null)
[ -n "$BUILD_ID" ] || { echo "SMOKE_FAIL no_build_id"; exit 1; }

SMOKE_SERVER_TIMEOUT=${SMOKE_SERVER_TIMEOUT:-600}
SMOKE_ARM_TIMEOUT=${SMOKE_ARM_TIMEOUT:-300}
SMOKE_NEW=${SMOKE_NEW:-8}
SMOKE_CACHE_TOKENS=${SMOKE_CACHE_TOKENS:-4096}
SMOKE_SLACK_GATE=${SMOKE_SLACK_GATE:-0}          # 1 = a thin slack refuses the release (see notes)
ENSURE=${SEAM_ENSURE:-}
if [ -z "$ENSURE" ]; then
  for c in "$PAIR_SCRIPTS/seam_server_ensure.sh" "$HERE/seam_server_ensure.sh"; do
    [ -x "$c" ] && { ENSURE=$c; break; }
  done
fi
[ "$PAIR_DRY" = 1 ] || [ -x "$ENSURE" ] || { echo "SMOKE_FAIL no_seam_server_ensure ${ENSURE:-<not found>}"; exit 1; }

fail() { echo "SMOKE_FAIL $*"; exit 1; }

gx_mem() {  # prints "avail_mib=<n> swap_free_mib=<n> swap_total_mib=<n>"
  pair_gxsh "awk '/^MemAvailable:/{a=int(\$2/1024)} /^SwapFree:/{f=int(\$2/1024)} /^SwapTotal:/{t=int(\$2/1024)} END{printf \"avail_mib=%d swap_free_mib=%d swap_total_mib=%d\n\", a, f, t}' /proc/meminfo" \
    | tail -1
}

if [ "$PAIR_DRY" = 1 ]; then
  echo "DRY ensure: ${ENSURE:-<not found>} GX=$GX CT=$GXCT GX_GID=$GX_GID PORT=$PORT SEAM_BUILD_ID=$BUILD_ID SEAM_SERVER_TIMEOUT=$SMOKE_SERVER_TIMEOUT $ENSURE --fresh --buckets $BUCKETS"
  echo "DRY arm: $HERE/pair_arm.sh $PAIR --rows 1 --max-new $SMOKE_NEW --cache-tokens $SMOKE_CACHE_TOKENS"
  echo "SMOKE_ROW i=0 new_tokens=$SMOKE_NEW text_sha256=dry"
  echo "SMOKE_OK"
  exit 0
fi

echo "SMOKE_MEM_BEFORE $(gx_mem)"
pair_step "SMOKE_SERVER_START pair=$PAIR ct=$GXCT build=$BUILD_ID buckets=$BUCKETS timeout=${SMOKE_SERVER_TIMEOUT}s"
srv_out=$(GX=$GX CT=$GXCT GX_DEV=${GX_DEV:-rocep1s0f0} GX_GID=$GX_GID PORT=$PORT \
          SERVER_BUCKETS=$BUCKETS SEAM_SERVER_TIMEOUT=$SMOKE_SERVER_TIMEOUT \
          SEAM_SLACK_GATE=$SMOKE_SLACK_GATE SEAM_ENV="SEAM_BUILD_ID=$BUILD_ID" \
          "$ENSURE" --fresh --buckets "$BUCKETS" 2>&1)
srv_rc=$?
[ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$srv_out" >> "$PAIR_LOG"
printf '%s\n' "$srv_out" | grep -E '^(SEAM_ENSURE|SEAM_SLACK)' | sed 's/^/SMOKE_SERVER /'
case "$srv_rc" in
  0) ;;
  3) fail "server_preflight_refused $(printf '%s' "$srv_out" | grep -m1 PREFLIGHT_FAIL)"
     ;;
  4) fail "server_slack_refused $(printf '%s' "$srv_out" | grep -m1 SEAM_SLACK)"
     ;;
  2) fail "server_timeout_or_died after<${SMOKE_SERVER_TIMEOUT}s $(printf '%s' "$srv_out" | grep -m1 SEAM_ENSURE)"
     ;;
  *) fail "server_start rc=$srv_rc $(printf '%s' "$srv_out" | tail -1)"
     ;;
esac
case "$srv_out" in
  *state=started*) ;;
  *) fail "server_not_ready $(printf '%s' "$srv_out" | grep -m1 SEAM_ENSURE)" ;;
esac

loaded=$(pair_gxsh "docker exec $GXCT sh -c \"grep -m1 '^SEAM_SERVER_LOADED' /tmp/seam_server.log\"" 2>/dev/null | tail -1)
echo "SMOKE_SERVER_LOADED $loaded"
case "$loaded" in
  *"build=$BUILD_ID"*) ;;
  *) fail "server_build_mismatch loaded=${loaded:-none} want=$BUILD_ID" ;;
esac
echo "SMOKE_MEM_SERVER $(gx_mem)"

pair_step "SMOKE_ARM pair=$PAIR rows=1 new=$SMOKE_NEW"
arm_out=$("$HERE/pair_arm.sh" "$PAIR" --rows 1 --max-new "$SMOKE_NEW" --cache-tokens "$SMOKE_CACHE_TOKENS" \
          --timeout "$SMOKE_ARM_TIMEOUT" 2>&1)
arm_rc=$?
[ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$arm_out" >> "$PAIR_LOG"
if [ "$arm_rc" != 0 ]; then
  fail "arm rc=$arm_rc $(printf '%s' "$arm_out" | grep -m1 -E 'peer layout mismatch|remote invalid request|Traceback|Error|error' | cut -c1-300)"
fi
case "$arm_out" in
  *"peer layout mismatch"*|*"remote invalid request"*)
    fail "wire_mismatch $(printf '%s' "$arm_out" | grep -m1 -E 'peer layout mismatch|remote invalid request' | cut -c1-300)" ;;
esac

row=$(printf '%s\n' "$arm_out" | python3 - <<'PY'
import hashlib, json, sys
rows = []
for line in sys.stdin:
    line = line.strip()
    if not (line.startswith("{") and line.endswith("}")):
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if "text" not in d:
        continue
    rows.append(d)
if not rows:
    print("NONE")
    sys.exit(0)
rows.sort(key=lambda d: d.get("i", 0))
blob = "\n".join(f"{d.get('i', 0)}|{d['text']}" for d in rows)
new = sum(int(d.get("new_tokens") or 0) for d in rows)
print(f"i={rows[0].get('i', 0)} rows={len(rows)} new_tokens={new} "
      f"ttft_ms={rows[0].get('ttft_ms', '?')} decode_ms_per_tok={rows[0].get('decode_ms_per_tok', '?')} "
      f"text_sha256={hashlib.sha256(blob.encode()).hexdigest()}")
print(f"NEW {new}")
PY
)
printf '%s\n' "$row" | sed -n 's/^i=/SMOKE_ROW /p'
new_tokens=$(printf '%s\n' "$row" | sed -n 's/^NEW //p')
[ "${new_tokens:-0}" -gt 0 ] || fail "arm_no_tokens $(printf '%s\n' "$row" | head -1)"
echo "SMOKE_OK $PAIR $SHA $BUILD_ID"
