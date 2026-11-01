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
#
# `--ids` runs the same driver through pair_arm_ids.py, which additionally prints one
# `ARM_IDS arm=… n=… sha256=… first=…` line per arm (the ids the engine generated) and writes them
# to $ARM_IDS_OUT. Everything else is unchanged, so a row recorded with `--ids` compares equal.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_arm.sh <pair> [--rows N] ...}
shift
ROWS=1; MAX_NEW=8; WARMUP=0; CACHE=4096; ARM_TIMEOUT=300; OUT=""; IDS=0
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
    --ids) IDS=1 ;;
    *) pair_die "unknown flag $1" ;;
  esac
  shift
done
pair_load "$PAIR_ARG"
# The client half must name the release it is running, or the server's session handshake accepts a
# mixed pair (`SEAM_SESSION_START ... build=unknown proto=3`, 2026-11-01). A released tree has no
# .git, so qes.build_id() falls back to <tree>/qmesh/BUILD_ID (written by pair_cut.sh); exporting it
# here makes the env the first answer on both halves, which is the pair-level check that matters.
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' \
  "$PAIR_ROOT/current/RELEASE.json" 2>/dev/null) || BUILD_ID=""

envs=(
  -e "CUDA_VISIBLE_DEVICES=$GPU"
  -e "REMOTE_TRANSPORT=seam"
  -e "SEAM_DEV=$SEAM_DEV" -e "SEAM_GID=$SEAM_GID" -e "SEAM_PEER=$SEAM_PEER"
  # no SEAM_BUCKETS: the client's own default is the superset and the server narrows it to what it
  # granted. Pinning the server's list here would silently cap every prefill chunk at its max row.
  -e "MAX_PROMPTS=$ROWS" -e "MAX_NEW=$MAX_NEW" -e "WARMUP=$WARMUP" -e "WARMUP_NEW=$MAX_NEW"
  -e "CACHE_TOKENS=$CACHE" -e "PROMPTS=$PROMPTS" -e "CHECKPOINT=$CHECKPOINT"
)
[ -n "$BUILD_ID" ] && envs+=(-e "SEAM_BUILD_ID=$BUILD_ID")
[ -n "$OUT" ] && envs+=(-e "OUT=$(pair_ct_path "$OUT")")
# The driver has to be a real file, not stdin: the client's CPU-host module starts its worker with
# multiprocessing "spawn", which re-executes the parent's main file by path -- with `python3 -` that
# path is "<stdin>", the child dies on FileNotFoundError and the parent sees only EOFError. The
# artifact tree is mounted in the container, so the driver is written there and run by path.
if [ "$IDS" = 1 ]; then
  drv_host=$PAIR_STATE_DIR/arm/pair_arm_ids.py
  mkdir -p "$(dirname "$drv_host")"
  cp "$HERE/pair_arm_ids.py" "$drv_host"
  drv=$(pair_ct_path "$drv_host")
  [ -n "${ARM_IDS_OUT:-}" ] && envs+=(-e "ARM_IDS_OUT=$(pair_ct_path "$ARM_IDS_OUT")")
fi

if [ "$PAIR_DRY" = 1 ]; then
  echo "DRY docker exec ${envs[*]} $P7CT bash -lc 'cd /src/exllamav3 && timeout $ARM_TIMEOUT python3 -u ${drv:-qmesh/e2e_prompts.py}'"
  exit 0
fi
timeout "$ARM_TIMEOUT" docker exec "${envs[@]}" "$P7CT" \
  bash -lc "cd /src/exllamav3 && python3 -u ${drv:-qmesh/e2e_prompts.py}"
