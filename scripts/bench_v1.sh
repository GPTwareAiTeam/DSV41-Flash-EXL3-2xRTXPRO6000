#!/usr/bin/env bash
# Single benchmark run of the 2-rank TP2 runtime: launch both ranks, wait, report.
#
# The host artifact directory ($EXL3_ARTIFACTS) is assumed to be mounted at /artifacts
# inside the container; set CT_ARTIFACTS if you mount it somewhere else.
#
# Usage: scripts/bench_v1.sh [-e KEY=VAL ...]
# Env:   EXL3_ARTIFACTS (default ./artifacts), CONTAINER (tp2-ctr), REPO_DIR (/workspace),
#        TAG, MAX_NEW (128), COMM (p2p), DSPARK (1), NUM_DRAFT (5), BATCH (1), DRY_RUN (0)
set -euo pipefail

EXL3_ARTIFACTS=${EXL3_ARTIFACTS:-./artifacts}
CT_ARTIFACTS=${CT_ARTIFACTS:-/artifacts}
CONTAINER=${CONTAINER:-tp2-ctr}
REPO_DIR=${REPO_DIR:-/workspace}
TAG=${TAG:-bench_$(date +%Y%m%d_%H%M%S)}
MAX_NEW=${MAX_NEW:-128}
COMM=${COMM:-p2p}
DSPARK=${DSPARK:-1}
NUM_DRAFT=${NUM_DRAFT:-5}
BATCH=${BATCH:-1}
DRY_RUN=${DRY_RUN:-0}

EXTRA=("$@")
LOGS_HOST="$EXL3_ARTIFACTS/tp2logs"
LOGS_CT="$CT_ARTIFACTS/tp2logs"
RV_HOST="$LOGS_HOST/${TAG}_rendezvous"
RDV_HOST="$LOGS_HOST/${TAG}_p2p"
RV_CT="$LOGS_CT/${TAG}_rendezvous"
RDV_CT="$LOGS_CT/${TAG}_p2p"

mkdir -p "$LOGS_HOST"
rm -rf "$RV_HOST" "$RDV_HOST"
mkdir -p "$RV_HOST" "$RDV_HOST"

echo "tag=$TAG comm=$COMM dspark=$DSPARK(num_draft=$NUM_DRAFT) batch=$BATCH max_new=$MAX_NEW"
echo "logs: $LOGS_HOST/${TAG}_rank{0,1}.log"

for r in 0 1; do
  cmd=(docker exec -d
    -e RANK="$r" -e WORLD_SIZE=2
    -e INIT_METHOD="file://$RV_CT" -e P2P_RDV="$RDV_CT"
    -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions
    -e TOKENS_OUT="$LOGS_CT/${TAG}_tokens_rank$r"
    -e PROFILE_TRACE="$LOGS_CT/${TAG}_trace_rank$r.json"
    -e COMM="$COMM" -e DSPARK="$DSPARK" -e NUM_DRAFT="$NUM_DRAFT" -e MAX_NEW="$MAX_NEW"
    -e BATCH="$BATCH" -e IGNORE_EOS=1 -e WARM=1
    -e EXL3_WHOLE_GRAPH=1 -e EXL3_DSV4_B1_BATCH=1
    "${EXTRA[@]}"
    "$CONTAINER" bash -c
    "cd $REPO_DIR && python3 -m qmesh.tp2.run_tp2 > $LOGS_CT/${TAG}_rank$r.log 2>&1")
  if [ "$DRY_RUN" = 1 ]; then printf '%q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
done

if [ "$DRY_RUN" = 1 ]; then
  echo "dry run: not waiting"
  exit 0
fi

echo "waiting for both ranks"
while docker exec "$CONTAINER" pgrep -f '[q]mesh.tp2.run_tp2' >/dev/null 2>&1; do
  sleep 5
done

status=0
for r in 0 1; do
  log="$LOGS_HOST/${TAG}_rank$r.log"
  if [ ! -s "$log" ]; then
    echo "rank $r: no log at $log"; status=1; continue
  fi
  if grep -qE 'Traceback|CUDA out of memory|RuntimeError' "$log"; then
    echo "rank $r: FAILED (see $log)"
    grep -E 'Error|error' "$log" | tail -3
    status=1
  fi
done
[ "$status" = 0 ] || exit "$status"

echo
echo "rank 0 timing"
grep -E '^\[[^]]*\] (TIMING|SPEC|MEM)' "$LOGS_HOST/${TAG}_rank0.log" || true
echo
echo "summary (rank 0, ms/token and tok/s from the runner's own effective rate)"
grep -oE 'TIMING [0-9]+ total=[0-9.]+s eff_tok_s=[0-9.]+' "$LOGS_HOST/${TAG}_rank0.log" |
  awk '{ split($3,t,"="); split($4,e,"="); printf "  stream %s: %s s, %.2f tok/s, %.2f ms/token\n", $2, t[2], e[2], 1000/e[2] }'

echo
echo "token identity across ranks:"
for r in 0 1; do
  [ -f "$LOGS_HOST/${TAG}_tokens_rank$r.json" ] || { echo "  rank $r: no token file"; continue; }
done
if command -v cmp >/dev/null && [ -f "$LOGS_HOST/${TAG}_tokens_rank0.json" ] && [ -f "$LOGS_HOST/${TAG}_tokens_rank1.json" ]; then
  if cmp -s "$LOGS_HOST/${TAG}_tokens_rank0.json" "$LOGS_HOST/${TAG}_tokens_rank1.json"; then
    echo "  identical"
  else
    echo "  DIFFER -- do not report this run's numbers"
    status=1
  fi
fi
exit "$status"
