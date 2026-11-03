#!/usr/bin/env bash
# run_ab.sh <tag> -- one ticket: the K map on the release client, then the instrumented arms.
#
# K <= 5 only: the DSpark drafter drafts one block of dspark_block_size = 5 positions per forward and
# the Generator clamps num_draft_tokens to it (caps["max_draft_size"]), so K = 6/8 would just repeat K = 5.
# Phase B is skipped when the K map aborts on its health probe: a paging box describes nothing.
set -uo pipefail
TAG=${1:?usage: run_ab.sh <tag>}
DIR=/home/treadall/recipe-artifacts/dspark-tpr
bash "$DIR/km_sweep.sh" "$TAG" 3,4,5
kmrc=$?
echo "$(date +%H:%M:%S) PHASE_A rc=$kmrc" >> "$DIR/logs/SUMMARY.$TAG"
if [ "$kmrc" -ne 0 ]; then
  echo "$(date +%H:%M:%S) RUN_AB_ABORT phase=B reason=phase_a_rc_$kmrc" >> "$DIR/logs/SUMMARY.$TAG"
  exit 3
fi
bash "$DIR/phase_b.sh" "$TAG"
echo "$(date +%H:%M:%S) RUN_AB_DONE tag=$TAG" >> "$DIR/logs/SUMMARY.$TAG"
