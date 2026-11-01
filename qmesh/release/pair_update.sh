#!/usr/bin/env bash
# pair_update.sh -- queue an update of one pair, to land right after the job that is running now.
#
#   pair_update.sh <pair> [sha] [--max-min N] [--flip-only]
#
# --flip-only: the release was already cut, wired and built out of band by pair_prepare.sh, so the
# ticket only flips `current`, restarts the server, smokes and verifies (minutes, not the hour a
# torch JIT build takes). Without it the ticket still does everything itself.
#
# The update is not run here: it is a ticket on the pair's own GPU queue, so it waits for the pair's
# current job to finish and never collides with one (one GPU job per pair, always through the queue).
# PairUpdate is put on line 1 of gpu_priority.txt first, so it goes next even if other agents are
# already waiting -- their tickets keep the rank they were created with, so nobody is reordered.
#
# Prints ENQUEUED <pair> agent=PairUpdate pid=... ticket=... log=...
# Follow it with:  scripts/pair_status.sh <pair>   and   tail -f .../pairs/logs/latest-update-<pair>.log
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_update.sh <pair> [sha] [--max-min N]}
shift
SHA=""; MINS=${PAIR_UPDATE_MAX_MIN:-120}; FLIP_ONLY=${PAIR_FLIP_ONLY:-0}
while [ $# -gt 0 ]; do
  case "$1" in
    --max-min) MINS=$2; shift ;;
    --flip-only) FLIP_ONLY=1 ;;
    -*) pair_die "unknown flag $1" ;;
    *) SHA=$1 ;;
  esac
  shift
done
pair_load "$PAIR_ARG"

pair_ensure_priority PairUpdate
export PAIR_FLIP_ONLY=$FLIP_ONLY
pair_step "UPDATE_ENQUEUE pair=$PAIR sha=${SHA:-<branch tip>} max=${MINS}m flip_only=$FLIP_ONLY"
cmd=("$HERE/pair_update_run.sh" "$PAIR_ARG" ${SHA:+"$SHA"})
pair_enqueue PairUpdate "$MINS" "${cmd[@]}"
