#!/usr/bin/env bash
# pair_smoke.sh -- prove a pair can serve one decode row AND one 32k prefill on the release it is
# wired to, on the kernel and piece size it ships.
#
#   pair_smoke.sh <pair> [sha]
#
# What it does, in order, with a bound on every wait:
#   1. records the GX box's memory before the load (MemAvailable, swap),
#   2. starts the seam server from the release (seam_server_ensure --fresh) with the pair's shipping
#      server env (pair_lib.sh: cuda-exl3 preload, load margin, pinned piece size) and SEAM_BUILD_ID
#      set, so the SEAM_SERVER_LOADED line names the release that is actually running,
#   3. refuses a server that came up without the cuda-exl3 kernel -- SEAM_MOE_NOEXT means the vendored
#      extension was not built into this release and every layer runs the mixed-K module at a
#      fraction of the rate -- and one where too few layers could be restacked,
#   4. records the memory *after* the pack is loaded and the slack the ensure script reports -- on GX2
#      a 116 GiB pack leaves ~1.4 GiB, and a server that dies of that has to be caught here, not in
#      the middle of someone's arm,
#   5. runs one decode row through the P7 client half and checks it produced tokens and ids and that
#      the wire did not answer `peer layout mismatch` / `remote invalid request`,
#   6. runs one 32k prefill arm and gates its rate and aborts (dev: the owner's canary bar; prod: what
#      the same release measured on prod's own hardware),
#   7. prints SMOKE_OK or SMOKE_FAIL <reason> and exits 0/1.
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

SMOKE_SERVER_TIMEOUT=${SMOKE_SERVER_TIMEOUT:-900}
SMOKE_ARM_TIMEOUT=${SMOKE_ARM_TIMEOUT:-600}
SMOKE_PREFILL_TIMEOUT=${SMOKE_PREFILL_TIMEOUT:-1800}
SMOKE_NEW=${SMOKE_NEW:-8}
# The shipping cache: 1M tokens of FP4 KV. A smoke that runs with a smaller cache tests a different
# memory shape than the pair serves.
SMOKE_CACHE_TOKENS=${SMOKE_CACHE_TOKENS:-1048576}
SMOKE_SLACK_GATE=${SMOKE_SLACK_GATE:-0}          # 1 = a thin slack refuses the release (see notes)
# Payload slots per handle. One is what the validated run used: a second slot is only needed to
# pipeline a split prefill, and on GX2 (~1.4 GiB free after the load) the extra slab is what a paging
# wall costs. seam_server_ensure.sh only passes --slab-slots to a worker that knows the flag.
SMOKE_SLAB_SLOTS=${SMOKE_SLAB_SLOTS:-1}
# The grant the smoke server is given: the pair's own list (pairs.conf). Workspace for a bucket nobody
# asks for still comes out of the box (242 MiB slab at 8192 rows), so the smoke grants exactly what
# the pair grants -- otherwise it would pass a release on a config nobody serves.
#
# The prefill gate. dev is the canary and carries the owner's bar: 32k tokens in one arm at >= 800
# tok/s with no abort (the validated release did 837.6). prod's own hardware measured 796.2 tok/s on
# the same release, so its bar is that number minus a smoke's worth of noise; a prod smoke that lands
# near dev's is the promotion confirming the copy, not a different instrument.
SMOKE_BUCKETS=${SMOKE_BUCKETS:-$BUCKETS}
SMOKE_PREFILL_DEPTH=${SMOKE_PREFILL_DEPTH:-32768}
if [ "$PAIR" = dev ]; then
  SMOKE_PREFILL_MIN_TPS=${SMOKE_PREFILL_MIN_TPS:-800}
else
  SMOKE_PREFILL_MIN_TPS=${SMOKE_PREFILL_MIN_TPS:-760}
fi
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
  echo "DRY ensure: ${ENSURE:-<not found>} GX=$GX CT=$GXCT GX_GID=$GX_GID PORT=$PORT SEAM_SERVER_TIMEOUT=$SMOKE_SERVER_TIMEOUT SEAM_ENV='$(pair_server_env "$BUILD_ID")' $ENSURE --fresh --buckets $SMOKE_BUCKETS"
  echo "DRY arm: $HERE/pair_arm.sh $PAIR --rows 1 --max-new $SMOKE_NEW --cache-tokens $SMOKE_CACHE_TOKENS"
  echo "DRY prefill: $HERE/pair_arm.sh $PAIR --prefill $SMOKE_PREFILL_DEPTH --timeout $SMOKE_PREFILL_TIMEOUT"
  echo "SMOKE_ROW i=0 new_tokens=$SMOKE_NEW text_sha256=dry"
  echo "SMOKE_PREFILL depth=$SMOKE_PREFILL_DEPTH tok_per_s=dry aborted=0"
  echo "SMOKE_OK"
  exit 0
fi

# The pack check runs before the GX gate (and only for a real smoke: DRY above never reads a
# filesystem). It costs a second and it reads the spec and the pack tree, so a pack that is not on the
# serving box's own disk is refused without a server start or a GPU second. PAIR_PACK_DISK=0 skips it
# (tests whose stub ssh has no filesystem to resolve).
if [ "${PAIR_PACK_DISK:-1}" = 1 ]; then
  pack_out=$(python3 "$HERE/pair_pack_disk.py" "$PAIR" 2>&1); pack_rc=$?
  printf '%s\n' "$pack_out" | sed 's/^/  /'
  [ "$pack_rc" = 0 ] || fail "pack_disk rc=$pack_rc (a pack is not on the serving box's own disk)"
fi

# The smoke owns the GX box until this gate expires: it loads the pack and measures avail_mib, and an
# experiment that loads or unloads a pack next to it turns every number below into a measurement of two
# jobs at once. A foreign claim is waited out (SMOKE_GATE_WAIT_MIN, then the smoke gives up rather than
# measure on a shared box); ours is self-expiring, so a killed smoke frees the box.
if ! pair_gx_gate_take "smoke-$PAIR" $(( (SMOKE_SERVER_TIMEOUT + SMOKE_ARM_TIMEOUT + SMOKE_PREFILL_TIMEOUT) / 60 + 15 )) "${SMOKE_GATE_WAIT_MIN:-20}"; then
  echo "SMOKE_GATE_BUSY $(pair_gx_gate_state)"
  fail "gx_busy $(pair_gx_gate_state)"
fi
# The claim and the server this smoke loaded are both its own. pair_gx_gate_down used to be the whole
# trap, so a smoke that was terminated left the pack resident and every later ticket queued behind it.
# A smoke that reached its verdict keeps the server: that server is the update's result, not a leak.
SMOKE_VERDICT=0
smoke_exit() {
  pair_gx_gate_down
  [ "$SMOKE_VERDICT" = 1 ] || pair_server_stop
}
trap smoke_exit EXIT
echo "SMOKE_GATE owner=smoke-$PAIR until=$(sed -n 's/.*until=\([0-9][0-9]*\).*/\1/p' "$(pair_gx_gate_file)")"

echo "SMOKE_MEM_BEFORE $(gx_mem)"

# A previous server's memory is not returned the instant its process exits: on 2026-11-02 the smoke
# started 40 s after an experiment had stopped, read avail 116726 MiB, and the worker refused a load
# of 116220+3072; minutes later the same box read 120254. Nothing about the release was wrong, and the
# ticket failed on a transient. So before asking the worker, wait for the box to come back: poll until
# MemAvailable covers the pack plus the load margin, up to SMOKE_MEM_WAIT_S. The pack size is read from
# the pack the server will actually load (the worker's own need is the same sum of the same files).
PACK_MIB=$(pair_gxsh "docker exec $GXCT sh -c 'du -bc /cold/cold-L*.safetensors 2>/dev/null | tail -1'" 2>/dev/null | cut -f1)
case "${PACK_MIB:-}" in ''|*[!0-9]*) PACK_MIB=0 ;; esac
if [ "$PACK_MIB" -gt 0 ]; then
  PACK_MIB=$(( PACK_MIB / 1048576 ))
  want=$(( PACK_MIB + LOAD_MARGIN ))
  waited=0
  while :; do
    mem=$(gx_mem); avail=$(printf '%s' "$mem" | sed -n 's/.*avail_mib=\([0-9]*\).*/\1/p')
    [ "${avail:-0}" -ge "$want" ] && break
    [ "$waited" -ge "${SMOKE_MEM_WAIT_S:-120}" ] && break
    echo "SMOKE_MEM_WAIT waited=${waited}s avail_mib=$avail want_mib=$want $(printf '%s' "$mem" | sed 's/^avail_mib=[0-9]* //')"
    sleep 10; waited=$(( waited + 10 ))
  done
  echo "SMOKE_MEM_READY avail_mib=${avail:-0} want_mib=$want pack_mib=$PACK_MIB margin_mib=$LOAD_MARGIN waited=${waited}s"
fi
pair_step "SMOKE_SERVER_START pair=$PAIR ct=$GXCT build=$BUILD_ID buckets=$SMOKE_BUCKETS timeout=${SMOKE_SERVER_TIMEOUT}s"
srv_out=$(GX=$GX CT=$GXCT GX_DEV=${GX_DEV:-rocep1s0f0} GX_GID=$GX_GID PORT=$PORT \
          SERVER_BUCKETS=$SMOKE_BUCKETS SEAM_SERVER_TIMEOUT=$SMOKE_SERVER_TIMEOUT \
          SEAM_SLACK_GATE=$SMOKE_SLACK_GATE SEAM_SLAB_SLOTS=$SMOKE_SLAB_SLOTS \
          SEAM_ENV="$(pair_server_env "$BUILD_ID")" \
          "$ENSURE" --fresh --buckets "$SMOKE_BUCKETS" 2>&1)
srv_rc=$?
[ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$srv_out" >> "$PAIR_LOG"
printf '%s\n' "$srv_out" | grep -E '^(SEAM_ENSURE|SEAM_SLACK|SEAM_NO_MEMORY|SEAM_NO_PACK)' | sed 's/^/SMOKE_SERVER /'
# The worker refuses to load a pack that does not fit in MemAvailable and exits 3 -- but the ensure
# wrapper reports "died" (rc 2) for that too, so the verdict has to be read from the line, not from
# rc alone. Otherwise a box that is merely full is reported as a bad release and rolled back.
case "$srv_out" in
  *SEAM_NO_MEMORY*)
    fail "memory server_no_memory $(printf '%s' "$srv_out" | grep -m1 SEAM_NO_MEMORY)" ;;
  *SEAM_NO_PACK*)
    fail "no_pack $(printf '%s' "$srv_out" | grep -m1 SEAM_NO_PACK)" ;;
esac
case "$srv_rc" in
  0) ;;
  3) line=$(printf '%s' "$srv_out" | grep -m1 PREFLIGHT_FAIL)
     case "$line" in
       # the box, not the release, said no: the ticket keeps the code and says so (memory_fail)
       *gx-mem*) fail "memory server_preflight $line" ;;
       *)        fail "server_preflight_refused $line" ;;
     esac
     ;;
  4) fail "memory server_slack $(printf '%s' "$srv_out" | grep -m1 SEAM_SLACK)"
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
# This smoke started that worker, so it owns it: the pid goes to state so an interrupted ticket can
# stop exactly this process instead of whatever a later ticket loaded (see pair_server_stop). Without
# it a ticket killed at 23:32 left the pack loaded and the 23:54 smoke refused `pair_busy gx_procs=2`.
if pair_server_record "$BUILD_ID"; then
  echo "SMOKE_SERVER_PID $(sed -n 's/.* pid=\([0-9]*\).*/\1/p' "$(pair_server_pidfile)")"
else
  echo "SMOKE_SERVER_PID unrecorded"
fi

loaded=$(pair_gxsh "docker exec $GXCT sh -c \"grep -m1 '^SEAM_SERVER_LOADED' /tmp/seam_server.log\"" 2>/dev/null | tail -1)
echo "SMOKE_SERVER_LOADED $loaded"
case "$loaded" in
  *"build=$BUILD_ID"*) ;;
  *) fail "server_build_mismatch loaded=${loaded:-none} want=$BUILD_ID" ;;
esac
echo "SMOKE_MEM_SERVER $(gx_mem)"

# The kernel the server actually serves on. cuda-exl3 is imported from the release's vendored tree
# (qmesh/exl3_moe_cuda.py), whose .so is untracked and therefore has to be built per cut: a release
# whose build stage skipped that imports nothing, prints SEAM_MOE_NOEXT and serves every layer on the
# mixed-K module -- correct output at a fraction of the rate, i.e. exactly the kind of release that
# must not reach a pair. On a memory-tight box a few layers may legitimately refuse the restack
# (SEAM_MOE_NOFIT); the validated run had 2 of 40, so allow up to a tenth and no more.
kernel=$(pair_gxsh "docker exec $GXCT sh -c \"grep -E '^(SEAM_MOE_NOEXT|SEAM_PACK_KERNEL)' /tmp/seam_server.log | tail -3\"" 2>/dev/null)
printf '%s\n' "$kernel" | grep -v '^$' | sed 's/^/SMOKE_KERNEL /'
case "$kernel" in
  *SEAM_MOE_NOEXT*) fail "moe_kernel_missing $(printf '%s' "$kernel" | grep -m1 SEAM_MOE_NOEXT | cut -c1-240)" ;;
esac
kv=$(printf '%s\n' "$kernel" | sed -n 's/^SEAM_PACK_KERNEL .*cudaexl3_layers=\([0-9]*\) mixedk_layers=\([0-9]*\) of=\([0-9]*\).*/\1 \2 \3/p' | tail -1)
read -r cx_n mx_n nl_n <<<"$kv"
if [ -z "${nl_n:-}" ]; then
  fail "moe_kernel_unknown $(printf '%s' "$kernel" | tail -1 | cut -c1-240)"
fi
if [ $((cx_n * 10)) -lt $((nl_n * 9)) ]; then
  fail "moe_kernel_mostly_mixedk cudaexl3_layers=$cx_n mixedk_layers=$mx_n of=$nl_n (the release's cuda-exl3 fits fewer layers than it did when it was validated)"
fi

# The box's own verdict on the load, on every smoke: avail after the pack, the swap the load pushed
# to, and the slack the server reserved. A pair that passes here with 1.4 GiB and 3.7 GiB of swap is a
# pair whose next failure will look like a CUDA abort (2026-11-01 16:44, GX1).
printf '%s\n' "$srv_out" | grep -E '^SEAM_SERIAL|^SEAM_SLACK|^SEAM_SERVER_SLAB' | sed 's/^/SMOKE_SERVER /'

pair_step "SMOKE_ARM pair=$PAIR rows=1 new=$SMOKE_NEW"
# The arm's own log: a client that hangs (a silent JIT rebuild, a stale lock baton, a seam handshake
# that never answers) produces no output at all by the time timeout(1) kills it, and "rc=124" then
# says nothing about why. Keep the bytes, and on failure the server's log tail next to them.
ARM_LOG=${SMOKE_ARM_LOG:-$PAIR_STATE_DIR/logs/smoke-arm-$PAIR-$(date -u +%Y%m%dT%H%M%SZ).log}
mkdir -p "$(dirname "$ARM_LOG")" 2>/dev/null || true
ARM_IDS_OUT=${ARM_IDS_OUT:-${ARM_LOG%.log}.ids.json}
arm_out=$(ARM_IDS_OUT="$ARM_IDS_OUT" \
          "$HERE/pair_arm.sh" "$PAIR" --rows 1 --max-new "$SMOKE_NEW" --cache-tokens "$SMOKE_CACHE_TOKENS" \
          --timeout "$SMOKE_ARM_TIMEOUT" --ids 2>&1)
arm_rc=$?
printf '%s\n' "$arm_out" > "$ARM_LOG" 2>/dev/null || true
[ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$arm_out" >> "$PAIR_LOG"
server_tail() {
  local log=${SEAM_LOG:-/tmp/seam_server.log}
  echo "SMOKE_SERVER_TAIL ct=$GXCT log=$log"
  pair_gxsh "docker exec $GXCT sh -c 'tail -40 $log 2>/dev/null; echo ---; grep -c . $log 2>/dev/null'" 2>&1 | tail -45
  echo "SMOKE_EXT_CACHE $(pair_gxsh "docker exec $GXCT sh -c 'ls -l /pair/extcache/exllamav3_ext/lock /pair/extcache/exllamav3_ext/*.so 2>/dev/null'" 2>&1 | head -3)"
  echo "SMOKE_P7_EXT_CACHE $(docker exec "$P7CT" sh -c 'ls -l /pair/extcache/exllamav3_ext/lock /pair/extcache/exllamav3_ext/*.so 2>/dev/null' 2>&1 | head -3)"
}
if [ "$arm_rc" != 0 ]; then
  echo "SMOKE_ARM_LOG $ARM_LOG ($(wc -c < "$ARM_LOG" 2>/dev/null || echo 0) bytes)"
  server_tail
  # A client killed for memory (the box swapped the pack out from under it) is a capacity failure:
  # classify it so the ticket keeps the release instead of rolling good code back.
  case "$(printf '%s\n' "$arm_out"; cat "$ARM_LOG" 2>/dev/null)" in
    *"out of memory"*|*"OutOfMemory"*|*"Cannot allocate memory"*|*"unspecified launch failure"*)
      fail "memory arm rc=$arm_rc $(printf '%s' "$arm_out" | grep -m1 -iE 'out of memory|OutOfMemory|Cannot allocate memory|unspecified launch failure' | cut -c1-200)" ;;
  esac
  fail "arm rc=$arm_rc $(printf '%s' "$arm_out" | grep -m1 -E 'peer layout mismatch|remote invalid request|Traceback|Error|error' | cut -c1-300)"
fi
case "$arm_out" in
  *"peer layout mismatch"*|*"remote invalid request"*)
    fail "wire_mismatch $(printf '%s' "$arm_out" | grep -m1 -E 'peer layout mismatch|remote invalid request' | cut -c1-300)" ;;
esac

# The grant, not the request, is the shape the pair runs: the server advertises its --buckets and the
# client adopts the intersection, so a client asking for less than the server grants silently caps
# every prefill chunk at its own largest bucket. That is what happened on 23:32 (server 1,6,64,8192,
# client default topped out at 4096 and contained neither 8192 nor 256, grant 1,6,64, chunk_rows=64,
# 1.6 s/token): the smoke measured a pair that was not the shipped one. A smoke that cannot see the
# shipping width has nothing to say about the release.
grant=$(printf '%s\n' "$arm_out" | sed -n 's/.*accepted=\([0-9][0-9,]*\).*/\1/p' | head -1)
grant_line=$(python3 "$HERE/pair_bucket_grant.py" "$grant" "$SMOKE_BUCKETS"); grant_rc=$?
case "$grant_rc" in
  0) echo "SMOKE_BUCKETS_GRANT $grant_line" ;;
  2) # No negotiated line: a client that never reached the server. The arm's own rc/rows checks below
     # already fail the smoke; inventing a grant verdict here would report the wrong cause.
     echo "SMOKE_BUCKETS_GRANT unknown want=$SMOKE_BUCKETS" ;;
  *) fail "bucket_grant $grant_line (a prefill chunk splits at the largest granted bucket; the arm must ask for the pair's list)" ;;
esac

# The rows are read from the arm's log, by path: `python3 - <<'PY'` owns stdin with its own source, so a
# piped arm_out read as empty and this check reported "arm_no_tokens NONE" for an arm whose log held a
# full 8-token row (23:29:19 on dev). The parser is pair_arm_rows.py so that it can be tested.
row=$(python3 "$HERE/pair_arm_rows.py" "$ARM_LOG")
printf '%s\n' "$row" | sed -n 's/^i=/SMOKE_ROW /p'
new_tokens=$(printf '%s\n' "$row" | sed -n 's/^NEW //p')
[ "${new_tokens:-0}" -gt 0 ] || fail "arm_no_tokens $(printf '%s\n' "$row" | head -1)"
# The ids are the promotion's proof that a decode really ran: a wrong build can still emit fluent
# text, but not the same ids. Missing ids fail the smoke unless SMOKE_IDS_OPTIONAL=1 (a release
# whose driver cannot report them), because prod may not be reached without them.
ids_lines=$(printf '%s\n' "$arm_out" | sed -n 's/^ARM_IDS /SMOKE_ARM_IDS /p')
ids_n=$(printf '%s\n' "$ids_lines" |
        awk '{for (i = 1; i <= NF; i++) if ($i ~ /^n=/) s += substr($i, 3) + 0} END {print s + 0}')
ids_n=${ids_n:-0}
if [ -n "$ids_lines" ]; then printf '%s\n' "$ids_lines"; else echo "SMOKE_ARM_IDS none"; fi
echo "SMOKE_ARM_LOG $ARM_LOG"
echo "SMOKE_ARM_IDS_OUT $ARM_IDS_OUT"
# One line the ticket can record without guessing which arm came first: the total across arms and a
# commitment over every ARM_IDS line (arm, count, per-arm hash), so two smokes of one release can be
# told apart even when a single arm's hash coincides.
if [ "$ids_n" -gt 0 ]; then
  ids_commit=$(printf '%s\n' "$ids_lines" | LC_ALL=C sort | sha256sum | cut -d' ' -f1)
  echo "SMOKE_ARM_IDS_TOTAL n=$ids_n sha256=$ids_commit"
fi
if [ "$ids_n" -le 0 ] && [ "${SMOKE_IDS_OPTIONAL:-0}" != 1 ]; then
  fail "arm_no_ids (no ARM_IDS line from the driver; a release that cannot report ids cannot be promoted, SMOKE_IDS_OPTIONAL=1 skips this)"
fi
# The prefill arm: the number the owner's canary bar is stated in. One load, one 32k prompt, one
# chunked prefill through the release's seam pieces, engine tok/s as reported by the same driver the
# 837.6/908.0 were taken with. Tokens without a rate (an aborted arm) can not pass: the abort is the
# failure mode this exists to catch, so both are read from the same line.
pair_step "SMOKE_PREFILL pair=$PAIR depth=$SMOKE_PREFILL_DEPTH min_tps=$SMOKE_PREFILL_MIN_TPS timeout=${SMOKE_PREFILL_TIMEOUT}s"
echo "SMOKE_MEM_PREFILL_BEFORE $(gx_mem)"
pf_log=$PAIR_STATE_DIR/logs/smoke-prefill-$PAIR-$(date -u +%Y%m%dT%H%M%SZ).log
mkdir -p "$(dirname "$pf_log")" 2>/dev/null || true
pf_out=$(ARM_IDS_OUT="" "$HERE/pair_arm.sh" "$PAIR" --prefill "$SMOKE_PREFILL_DEPTH" \
         --timeout "$SMOKE_PREFILL_TIMEOUT" 2>&1)
pf_rc=$?
printf '%s\n' "$pf_out" > "$pf_log" 2>/dev/null || true
[ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$pf_out" >> "$PAIR_LOG"
pf_line=$(printf '%s\n' "$pf_out" | sed -n "s/^PREFILL_ARM depth=$SMOKE_PREFILL_DEPTH .*/&/p" | tail -1)
if [ "$pf_rc" != 0 ] || [ -z "$pf_line" ]; then
  echo "SMOKE_PREFILL_LOG $pf_log"
  server_tail
  case "$(printf '%s\n' "$pf_out")" in
    *"out of memory"*|*"OutOfMemory"*|*"Cannot allocate memory"*|*"unspecified launch failure"*)
      fail "memory prefill rc=$pf_rc $(printf '%s' "$pf_out" | grep -m1 -iE 'out of memory|OutOfMemory|Cannot allocate memory|unspecified launch failure' | cut -c1-200)" ;;
  esac
  fail "prefill rc=$pf_rc $(printf '%s\n' "$pf_out" | grep -m1 -E 'Traceback|Error|error|MEM_ABORT' | cut -c1-300)"
fi
pf_kv=$(printf '%s\n' "$pf_line" | tr ' ' '\n' | sed -n 's/^\(tok_per_s\|aborted\|prefill_s\|shrunk\)=//p' | tr '\n' ' ')
pf_tps=$(printf '%s\n' "$pf_line" | sed -n 's/.*tok_per_s=\([0-9.]*\).*/\1/p')
pf_ab=$(printf '%s\n' "$pf_line" | sed -n 's/.*aborted=\([0-9]*\).*/\1/p')
echo "SMOKE_PREFILL_LOG $pf_log"
echo "SMOKE_PREFILL depth=$SMOKE_PREFILL_DEPTH ${pf_kv% } line_rc=0"
echo "SMOKE_MEM_PREFILL_AFTER $(gx_mem)"
[ "${pf_ab:-1}" = 0 ] || fail "prefill_aborted depth=$SMOKE_PREFILL_DEPTH aborted=$pf_ab line=$pf_line"
case "$(printf '%s\n' "$pf_out")" in
  *"SEAM_COMPUTE_FAIL"*)
    fail "prefill_compute_fail $(printf '%s' "$pf_out" | grep -m1 SEAM_COMPUTE_FAIL | cut -c1-240)" ;;
esac
[ -n "${pf_tps:-}" ] || fail "prefill_no_rate depth=$SMOKE_PREFILL_DEPTH line=$pf_line"
# awk, not test: tok/s is fractional and the gate is a real threshold, not a string compare.
awk -v got="$pf_tps" -v want="$SMOKE_PREFILL_MIN_TPS" 'BEGIN { exit !(got + 0 >= want + 0) }' \
  || fail "prefill_slow depth=$SMOKE_PREFILL_DEPTH tok_per_s=$pf_tps min=$SMOKE_PREFILL_MIN_TPS"
echo "SMOKE_PREFILL_OK depth=$SMOKE_PREFILL_DEPTH tok_per_s=$pf_tps min=$SMOKE_PREFILL_MIN_TPS aborts=0"
SMOKE_VERDICT=1
[ "${PAIR_SMOKE_KEEP_SERVER:-0}" = 1 ] || rm -f "$(pair_server_pidfile)"
echo "SMOKE_OK $PAIR $SHA $BUILD_ID"
