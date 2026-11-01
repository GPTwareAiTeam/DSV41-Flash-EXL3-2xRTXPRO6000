#!/usr/bin/env bash
# pair_arm.sh -- run client decode rows through a pair's P7 container, on the release it is serving.
#
#   pair_arm.sh <pair> [--rows N] [--max-new N] [--warmup N] [--cache-tokens N]
#                      [--prompts PATH] [--checkpoint PATH] [--timeout S] [--out FILE]
#
# Prints whatever qmesh/e2e_prompts.py prints (one JSON object per prompt on stdout, diagnostics on
# stderr) and returns the client's exit code, or 124 on timeout. The timeout is not politeness: a
# client against a mismatched or dead seam server blocks forever instead of failing, and a smoke
# step that can hang is worse than one that fails.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_arm.sh <pair> [--rows N] ...}
shift
ROWS=1; MAX_NEW=8; WARMUP=0; CACHE=4096; ARM_TIMEOUT=300; OUT=""
PROMPTS=${PAIR_ARM_PROMPTS:-/artifacts/2x-rtx-tp2/dtrain/prompts_held.json}
CHECKPOINT=${PAIR_ARM_CHECKPOINT:-/models/dsv41/exl3-2.8535-p7hot-hc1}
while [ $# -gt 0 ]; do
  case "$1" in
    --rows) ROWS=$2; shift ;;
    --max-new) MAX_NEW=$2; shift ;;
    --warmup) WARMUP=$2; shift ;;
    --cache-tokens) CACHE=$2; shift ;;
    --prompts) PROMPTS=$2; shift ;;
    --checkpoint) CHECKPOINT=$2; shift ;;
    --timeout) ARM_TIMEOUT=$2; shift ;;
    --out) OUT=$2; shift ;;
    *) pair_die "unknown flag $1" ;;
  esac
  shift
done
pair_load "$PAIR_ARG"

envs=(
  -e "CUDA_VISIBLE_DEVICES=$GPU"
  -e "REMOTE_TRANSPORT=seam"
  -e "SEAM_DEV=$SEAM_DEV" -e "SEAM_GID=$SEAM_GID" -e "SEAM_PEER=$SEAM_PEER"
  -e "SEAM_PORT=$PORT" -e "SEAM_BUCKETS=$BUCKETS"
  -e "MAX_PROMPTS=$ROWS" -e "MAX_NEW=$MAX_NEW" -e "WARMUP=$WARMUP" -e "WARMUP_NEW=$MAX_NEW"
  -e "CACHE_TOKENS=$CACHE" -e "PROMPTS=$PROMPTS" -e "CHECKPOINT=$CHECKPOINT"
)
[ -n "$OUT" ] && envs+=(-e "OUT=$OUT")

if [ "$PAIR_DRY" = 1 ]; then
  echo "DRY docker exec ${envs[*]} $P7CT bash -lc 'cd /src/exllamav3 && timeout $ARM_TIMEOUT python3 -u qmesh/e2e_prompts.py'"
  exit 0
fi
timeout "$ARM_TIMEOUT" docker exec "${envs[@]}" "$P7CT" \
  bash -lc "cd /src/exllamav3 && python3 -u qmesh/e2e_prompts.py"
