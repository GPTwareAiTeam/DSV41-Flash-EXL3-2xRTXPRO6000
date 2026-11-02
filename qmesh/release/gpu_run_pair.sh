#!/usr/bin/env bash
# Priority GPU queue for one P7 GPU (QGPUS) and its GX node. Usage:
#   gpu_run.sh <agent> <max_minutes> <command...>
# Priority comes from gpu_priority.txt (one agent name per line, top = first). Unlisted agents go last.
# Same priority: first come first served. The GPU is released as soon as the command exits.
#
# Deadline handling (soft, not a guillotine):
#   The command gets $GPU_RUN_STATUS (a file path). Write progress into it at any time, e.g.
#       echo "eta_min=6 doing=nsys profile arm 3/4" > "$GPU_RUN_STATUS"
#   At <max_minutes>, the job is extended (no kill) if its latest eta_min <= EXTEND_MAX_MIN (default 10),
#   as long as the total stays under HARD_CAP_MIN (default max_minutes + 30). Extensions repeat while
#   the ETA keeps qualifying. The supervisor can also grant time by writing minutes into
#   gpu_queue/extend-<agent> (consumed on read). With no qualifying ETA or grant, the script waits
#   GRACE_SEC (default 90) for one to appear, then kills the job.
#
# Release guard (qmesh/release/pair_check.sh): the GPU is free but the pair may still be serving a
# different release on its P7 and GX halves -- that is exactly how a pair hangs with `peer layout
# mismatch` or dies with `remote invalid request`. So before a job starts, the pair is checked
# against its manifest, and a job is refused (exit 4 drift, 5 check error) instead of poisoning the
# queue.
#
# Who this applies to: a ticket that names one of the pair's own containers (`seam-p7-dev`,
# `seam-p7-prod`, `seam-gx2-dev`, `seam-gx1-prod`) on its command line, and the update jobs
# themselves. A ticket that names no pair container -- an experiment in its own container, a
# different GPU, no `/pair` mount -- cannot be affected by that drift and is not held up by it: on
# 2026-11-02 the whole queue stalled behind a repair because every ticket was checked. `PAIR_CHECK_SKIP=1`
# still turns the check off for a command that reaches a pair indirectly, and `PAIR_CHECK_FORCE=1`
# turns it on for one that does not name the container it will use.
set -u
AGENT=${1:?agent}; MAXMIN=${2:?minutes}; shift 2
BASE=${QBASE:?QBASE}
Q=$BASE/gpu_queue; LOCK=$BASE/gpu.lock; PRIO=$BASE/gpu_priority.txt; LOG=$BASE/gpu_queue.log
EXTEND_MAX_MIN=${EXTEND_MAX_MIN:-10}; HARD_CAP_MIN=${HARD_CAP_MIN:-$((MAXMIN + 30))}; GRACE_SEC=${GRACE_SEC:-90}
mkdir -p "$Q"
log() { echo "$(date +%T) $*" >> "$LOG"; }
SCRIPTS=""
for _c in "$BASE/../../scripts" "$BASE/../scripts"; do
  [ -x "$_c/pair_check.sh" ] && { SCRIPTS=$(cd "$_c" && pwd); break; }
done
# check_release: 0 run, 4 drift (refuse), 5 check error (refuse). Transient ssh failures are retried
# three times 10 s apart: a 10 s network blip must not silently run a job on a stale pair.
# The pair's containers, as pairs.conf names them: a ticket that names none of these is not
# touching the pair's release.
PAIR_CONTAINERS="seam-p7-dev seam-p7-prod seam-gx2-dev seam-gx1-prod"
# Named on the command line, not reached through some script we cannot read: that is the line
# between "this ticket can hurt the pair" and "this ticket cannot".
names_pair() {
  local a c
  for a in "$@"; do
    for c in $PAIR_CONTAINERS; do
      case "$a" in *"$c"*) return 0 ;; esac
    done
  done
  return 1
}
check_release() {
  [ -n "${PAIR_CHECK_SKIP:-}" ] && return 0
  if [ -z "${PAIR_CHECK_FORCE:-}" ] && ! names_pair "$@" \
     && [ "$AGENT" != PairUpdate ] && [ "$AGENT" != PairUpdateRun ]; then
    log "PAIR_CHECK_SKIP release guard not applicable: $AGENT names no pair container"
    return 0
  fi
  [ "$AGENT" = PairUpdate ] && return 0
  [ "$AGENT" = PairUpdateRun ] && return 0
  local name; name=$(basename "$BASE")
  if [ -z "$SCRIPTS" ]; then
    log "PAIR_DRIFT $name no_scripts (pair_check.sh not found above $BASE) refuse=$AGENT"
    return 4
  fi
  local out rc i
  for i in 1 2 3; do
    out=$("$SCRIPTS/pair_check.sh" "$name" 2>&1); rc=$?
    case $rc in
      0) return 0 ;;
      4) log "PAIR_DRIFT $name refuse=$AGENT :: $(printf '%s' "$out" | grep -E 'DRIFT' | sed 's/^ *//' | tr '\n' ';')"
         return 4 ;;
      *) sleep 10 ;;
    esac
  done
  log "PAIR_CHECK_ERROR $name rc=$rc refuse=$AGENT :: $(printf '%s' "$out" | tail -3 | tr '\n' ';')"
  return 5
}
gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i ${QGPUS:?QGPUS})" ]; }
rank=$(grep -nx "$AGENT" "$PRIO" 2>/dev/null | cut -d: -f1); rank=${rank:-99}
ticket="$Q/$(printf %02d "$rank")-$(date +%s%N)-$$-$AGENT"
export GPU_RUN_STATUS="$Q/status-$AGENT-$$"
GRANT="$Q/extend-$AGENT"
touch "$ticket"; rm -f "$GPU_RUN_STATUS"
trap 'rm -f "$ticket" "$GPU_RUN_STATUS"' EXIT
log "queued $AGENT rank=$rank pid=$$"
while :; do
  # drop tickets whose owner process is gone
  for t in "$Q"/[0-9]*; do
    [ -e "$t" ] || continue
    p=$(basename "$t" | cut -d- -f3)
    kill -0 "$p" 2>/dev/null || rm -f "$t"
  done
  # Head = lowest ticket name (rank prefix, then enqueue time). Every waiter must use this same rule:
  # waiters running different script versions must agree on the head or the queue deadlocks.
  head=$(ls "$Q" | grep -E '^[0-9]' | sort | head -n1)
  if [ "$Q/$head" = "$ticket" ]; then
    exec 9>"$LOCK"
    flock 9
    gpu_busy || {
      check_release "$@"; grc=$?
      [ "$grc" = 0 ] && break
      flock -u 9
      log "refused $AGENT release_check_rc=$grc (not started)"
      exit "$grc"
    }
    flock -u 9
  fi
  sleep 5
done
start=$(date +%s); deadline=$((start + MAXMIN * 60)); hard=$((start + HARD_CAP_MIN * 60))
# The hard cap is part of the ticket's contract, not just this loop's business: pair_gx_gate_take
# refuses a claim taken outside a ticket and never lets one run past the ticket's cap, so hand the
# number to the command and write it into the ticket file (a claim taken by a child reads it from
# either). A claim past the cap is exactly the orphan that made the Integrator ticket fail on
# 2026-11-02 after GxPort2 held gx2.gate 26 minutes from outside the queue.
export GPU_TICKET_HARDCAP=$hard
printf 'hardcap=%s\n' "$hard" >>"$ticket" 2>/dev/null
log "start $AGENT max=${MAXMIN}m hardcap=${HARD_CAP_MIN}m load=$(cut -d" " -f1 /proc/loadavg)"
setsid "$@" & cpid=$!
grace_until=0
while kill -0 "$cpid" 2>/dev/null; do
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    eta=$(grep -o 'eta_min=[0-9]*' "$GPU_RUN_STATUS" 2>/dev/null | tail -n1 | cut -d= -f2)
    doing=$(grep -o 'doing=.*' "$GPU_RUN_STATUS" 2>/dev/null | tail -n1)
    grant=$(cat "$GRANT" 2>/dev/null); rm -f "$GRANT"
    if [ -n "$grant" ] && [ "$now" -lt "$hard" ]; then
      deadline=$((now + grant * 60)); [ "$deadline" -gt "$hard" ] && deadline=$hard; grace_until=0
      log "extend $AGENT +${grant}m (supervisor grant) $doing"
    elif [ -n "$eta" ] && [ "$eta" -le "$EXTEND_MAX_MIN" ] && [ "$now" -lt "$hard" ]; then
      deadline=$((now + (eta + 2) * 60)); [ "$deadline" -gt "$hard" ] && deadline=$hard; grace_until=0
      log "extend $AGENT +$((eta + 2))m (eta=${eta}m) $doing"
    elif [ "$now" -ge "$hard" ]; then
      log "kill $AGENT (hard cap ${HARD_CAP_MIN}m) eta=${eta:-none} $doing"; break
    elif [ "$grace_until" -eq 0 ]; then
      grace_until=$((now + GRACE_SEC))
      log "deadline $AGENT eta=${eta:-none} $doing — waiting ${GRACE_SEC}s for eta/grant"
    elif [ "$now" -ge "$grace_until" ]; then
      log "kill $AGENT (no qualifying eta/grant) eta=${eta:-none} $doing"; break
    fi
  fi
  sleep 5
done
if kill -0 "$cpid" 2>/dev/null; then
  kill -TERM -- -"$cpid" 2>/dev/null; sleep 30; kill -KILL -- -"$cpid" 2>/dev/null
fi
wait "$cpid" 2>/dev/null; rc=$?
# container-side runners can outlive the host command: wait for the GPU to drain before releasing
for _ in $(seq 1 60); do gpu_busy || break; sleep 5; done
log "end $AGENT rc=$rc elapsed=$((($(date +%s) - start) / 60))m gpu_clear=$(gpu_busy && echo NO || echo yes) load=$(cut -d" " -f1 /proc/loadavg)"
exit $rc
