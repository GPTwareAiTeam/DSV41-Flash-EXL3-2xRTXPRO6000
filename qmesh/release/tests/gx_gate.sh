#!/bin/bash
# gx_gate.sh -- the GX box claim, as the release scripts and an experiment both read it.
#
#   ./tests/gx_gate.sh
#
# A GX box carries one cold pack. A release build, a release smoke and an experiment that has the pack
# loaded all want it, and two of them at once has already OOM-killed a build and buried a measurement
# under somebody else's hundreds of MiB. The claim is a file: "<owner> pid=… until=<epoch>", free when
# missing or past. The properties that matter:
#   * a stale claim must not wedge the box (until is authoritative, not existence),
#   * a foreign live claim is waited out and never overwritten (two jobs must not both think they won),
#   * a job drops the claim only if the line is still its own (it must not release somebody else's).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); REL=$(dirname "$HERE")
W=$(mktemp -d /tmp/pairgate.XXXXXX)
export PAIR_GATE_DIR=$W
# shellcheck source=../pair_lib.sh
. "$REL/pair_lib.sh"
PAIR_HOME=$W; PAIR_DRY=0; GXCT=seam-gx2-dev; PAIR=dev
f=$W/gx2.gate
fails=0
want() {  # <case> <expected-substring> <actual>
  if printf '%s' "$3" | grep -q -- "$2"; then echo "OK   $1"; else echo "FAIL $1 (wanted '$2', got '$3')"; fails=$((fails + 1)); fi
}
eq() { if [ "$2" = "$3" ]; then echo "OK   $1"; else echo "FAIL $1 (wanted '$2', got '$3')"; fails=$((fails + 1)); fi; }

eq "the gate names the box from the container" "$W/gx2.gate" "$(pair_gx_gate_file)"
eq "no file means free" free "$(pair_gx_gate_state)"

printf 'owner=Experiment pid=999999 until=%s\n' "$(( $(date +%s) + 600 ))" > "$f"
want "a live foreign claim reads as taken" '^Experiment until=[0-9]*$' "$(pair_gx_gate_state)"
if pair_gx_gate_take mine 5 0; then echo "FAIL take_foreign (took a live foreign claim)"; fails=$((fails + 1));
else echo "OK   take_foreign (refused)"; fi
want "take_foreign left the foreign line alone" '^owner=Experiment ' "$(cat "$f" | tr '\n' ' ')"

printf 'owner=Experiment pid=999999 until=%s\n' "$(( $(date +%s) - 1 ))" > "$f"
eq "an expired claim is free" free "$(pair_gx_gate_state)"
if pair_gx_gate_take mine 5 0; then echo "OK   take_expired"; else echo "FAIL take_expired"; fails=$((fails + 1)); fi
want "take_expired wrote our line" '^owner=mine pid='"$$"' until=' "$(cat "$f" | tr '\n' ' ')"

printf 'owner=mine pid=1 until=%s\n' "$(( $(date +%s) + 600 ))" > "$f"
pair_gx_gate_down
[ -f "$f" ] && echo "OK   down_foreign_kept" || { echo "FAIL down_foreign_kept (released somebody else's claim)"; fails=$((fails + 1)); }

printf 'owner=mine pid=%s until=%s\n' "$$" "$(( $(date +%s) + 600 ))" > "$f"
pair_gx_gate_down
[ -f "$f" ] && { echo "FAIL down_own_removed (claim still there)"; fails=$((fails + 1)); } || echo "OK   down_own_removed"

# A build that is killed cannot clean up: the stamp has to expire on its own.
printf 'owner=build pid=777 until=%s\n' "$(( $(date +%s) + 1 ))" > "$f"
sleep 2
eq "a dead job's claim expires by itself" free "$(pair_gx_gate_state)"
pair_gx_gate_down

# A dry run writes nothing: the dry-run harness must not claim a real box.
rm -f "$f"; PAIR_DRY=1; pair_gx_gate_take dry 5 0; PAIR_DRY=0
[ -f "$f" ] && { echo "FAIL dry_writes_nothing (a dry run claimed the box)"; fails=$((fails + 1)); } || echo "OK   dry_writes_nothing"

rm -rf "$W"
echo "RESULT $( [ "$fails" = 0 ] && echo PASS || echo FAIL ) fails=$fails"
[ "$fails" = 0 ]
