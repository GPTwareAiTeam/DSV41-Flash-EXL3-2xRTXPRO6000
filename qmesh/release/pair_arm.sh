#!/usr/bin/env bash
# pair_arm.sh -- run a client measurement through a pair's P7 container, on the release it serves.
#
#   pair_arm.sh <pair> [--rows N] [--max-new N] [--warmup N] [--cache-tokens N]
#                      [--prompts PATH] [--checkpoint PATH] [--timeout S] [--out FILE]
#                      [--prefill DEPTH]
#
# Decode (the default): prints whatever qmesh/e2e_prompts.py prints (one JSON object per prompt on
# stdout, diagnostics on stderr) and returns the client's exit code, or 124 on timeout. The timeout is
# not politeness: a client against a mismatched or dead seam server blocks forever instead of failing,
# and a smoke step that can hang is worse than one that fails.
#
# `--prefill DEPTH` runs qmesh/mem_budget.py instead, with PREFILL_ARMS=DEPTH:8192 -- one arm at that
# depth, host-timed prefill and engine tok/s, which is the instrument the prefill numbers were taken
# with and what the smoke gates dev on.
#
# `--ids` runs the same driver through pair_arm_ids.py, which additionally prints one
# `ARM_IDS arm=… n=… sha256=… first=…` line per arm (the ids the engine generated) and writes them
# to $ARM_IDS_OUT. Everything else is unchanged, so a row recorded with `--ids` compares equal.
#
# The environment below is the shipping client config and is not inherited from the caller: the
# smoke, a manual arm and a promotion probe all have to run the pair's real shape, or their numbers
# describe something nobody serves.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_arm.sh <pair> [--rows N] ...}
shift
ROWS=1; MAX_NEW=8; WARMUP=0; CACHE=1048576; ARM_TIMEOUT=300; OUT=""; IDS=0
PROMPTS=${PAIR_ARM_PROMPTS:-/artifacts/2x-rtx-tp2/dtrain/prompts_held.json}
# The hot pack both pairs ship: 4460 cold experts per layer, drafter HC1. Its placement hash must be
# the one the GX box's cold pack carries (844251e7…), or the server's Hello refuses the session.
CHECKPOINT=${PAIR_ARM_CHECKPOINT:-/models/dsv41/exl3-p7hot-4460-hc1}
CORPUS=${PAIR_ARM_CORPUS:-/artifacts/2x-rtx-tp2/corpus/prose_24k.txt}
PREFILL_DEPTH=""; SRC=""
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
    --prefill) PREFILL_DEPTH=$2; shift ;;
    --ids) IDS=1 ;;
    --src) SRC=$2; shift ;;
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


# Which tree the client imports, and therefore which extension cache it may write. torch's version
# hash covers the *include path spelling*, not the content, so a client that imports a second
# checkout of the same sources rebuilds the extension in place -- in the cache it was pointed at. On
# 2026-11-03 20:59:31 a client running /artifacts/dspark-tpr/src with the pair's own
# TORCH_EXTENSIONS_DIR=/pair/extcache rebuilt exllamav3_ext.so in the pair's release cache (5780205b
# replacing the stamped 04a61140), and `pair_serve.sh` then refused the pair as DRIFT for an hour
# while every later client import rebuilt or hung on it. So the release cache
# (/pair/extcache, the release's own ext/) is for the release tree only; any other --src gets a
# cache of its own, under the pair's state dir, named by the tree so two agents cannot collide.
SRC=${SRC:-/src/exllamav3}
case "$SRC" in
  /src/exllamav3|/src/exllamav3/|/pair/current/src) EXT_DIR=/pair/extcache ;;
  *)
    ext_tag=$(printf '%s' "$SRC" | sed 's#^/*##; s#/*$##; s#[^A-Za-z0-9._-]#_#g')
    mkdir -p "$PAIR_STATE_DIR/extcache/$ext_tag"
    EXT_DIR=$(pair_ct_path "$PAIR_STATE_DIR/extcache/$ext_tag")
    echo "ARM_SRC_ISOLATED src=$SRC extcache=$EXT_DIR (not the release cache)"
    ;;
esac
envs=(
  # Stated, not inherited: the container's env is the release's, and a --src above must not be able
  # to reach the release cache through an inherited variable. For the release tree these are the two
  # values the container already carries, so the shipped exec is unchanged.
  -e "PYTHONPATH=$SRC" -e "TORCH_EXTENSIONS_DIR=$EXT_DIR"
  # pairs.conf's `gpu` is a *host* index: the container spec spends it on --gpus device=<gpu>, so
  # inside the container that one card is device 0 again. Passing the host index here asked dev's
  # container (which had every GPU) for GPU 1 and asked a scoped one for a GPU it cannot see.
  -e "CUDA_VISIBLE_DEVICES=0"
  -e "REMOTE_TRANSPORT=seam"
  -e "SEAM_DEV=$SEAM_DEV" -e "SEAM_GID=$SEAM_GID" -e "SEAM_PEER=$SEAM_PEER"
  # SEAM_BUCKETS is a *request*: the server grants the intersection of this list and its own
  # (--buckets, from pairs.conf). Leaving it unset used the client's fallback default
  # (1,6,64,256,512,1024,2048,4096), which does not contain the shipping prefill width 8192: the
  # intersection was 1,6,64, so the 23:32 dev arm negotiated chunk_rows=64 and split every 8192-row
  # prefill chunk into 128 calls of 64 rows (accepted=1,6,64, 1.6 s/token decode). Ask for the pair's
  # own list and the grant is exact.
  -e "SEAM_BUCKETS=$BUCKETS"
  -e "MAX_PROMPTS=$ROWS" -e "MAX_NEW=$MAX_NEW" -e "WARMUP=$WARMUP" -e "WARMUP_NEW=$MAX_NEW"
  -e "CACHE_TOKENS=$CACHE" -e "PROMPTS=$PROMPTS" -e "CHECKPOINT=$CHECKPOINT"
)
[ -n "$BUILD_ID" ] && envs+=(-e "SEAM_BUILD_ID=$BUILD_ID")
# The shipping client config, one list. Every knob here was in the run the notes quote (4460 packs,
# drafter on, 1M FP4 KV, 8192-row chunks with adaptive chunking, seam buckets up to the grant); a
# release that needs a different value changes this list, so the smoke checks what ships.
envs+=(
  -e "DRAFT_CHECKPOINT=$CHECKPOINT" -e "DSPARK=1" -e "NUM_DRAFT=5"
  -e "KV_BITS=4" -e "EXL3_WHOLE_GRAPH=0" -e "MAX_CHUNK=8192"
  -e "EXL3_MOE_SPLIT_FUSED=0"
  -e "EXL3_SEAM_MARK=1" -e "EXL3_SEAM_MARK_LAYERS=8"
  -e "EXL3_SEAM_PROF=1" -e "EXL3_SEAM_PROF_ROWS=2560"
  -e "PREFILL_ADAPTIVE=1" -e "PREFILL_ADAPTIVE_LOG=1"
  -e "SEAM_ROWS=8192"
  -e "SEAM_NS=1" -e "SEAM_OVERLAP=1" -e "SEAM_COPY_STREAM=1"
)

# PAIR_ARM_EXTRA_ENVS: space-separated K=V, each appended as `-e K=V` after the list above (docker
# keeps the last one, so this can also override a shipping knob). It exists for measurement-only
# instrumentation -- e.g. EXL3_SEAM_MARK_MIN_ROWS lowered so a 6-row decode call is not skipped by the
# seam timeline's 512-row floor -- and must stay empty for anything whose numbers are quoted as the
# shipped shape. Unset, the exec is byte-for-byte the one it was before.
for kv in ${PAIR_ARM_EXTRA_ENVS:-}; do envs+=(-e "$kv"); done
if [ -n "$PREFILL_DEPTH" ]; then
  envs+=(-e "PREFILL_ARMS=$PREFILL_DEPTH:8192" -e "CORPUS=$CORPUS"
         -e "ARM_MAX_S=${PAIR_ARM_ARM_MAX_S:-1200}" -e "LOAD_ONLY=0")
  drv=qmesh/mem_budget.py
fi
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
  echo "DRY docker exec ${envs[*]} $P7CT bash -lc 'cd $SRC && timeout $ARM_TIMEOUT python3 -u ${drv:-qmesh/e2e_prompts.py}'"
  exit 0
fi
timeout "$ARM_TIMEOUT" docker exec "${envs[@]}" "$P7CT" \
  bash -lc "cd $SRC && python3 -u ${drv:-qmesh/e2e_prompts.py}"
