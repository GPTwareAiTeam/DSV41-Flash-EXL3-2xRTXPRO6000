#!/usr/bin/env bash
# pair_build_ext.sh -- build the exllamav3 torch extension for one release, per container.
#
#   pair_build_ext.sh <pair> <sha> [--side p7|gx|both]
#
# The cache is per release (releases/<sha>/ext) and per container image (P7 torch 2.14, GX torch 2.11,
# and 12.0 vs 12.1 arch), so each side builds its own: a cache shared between containers is how one
# container's stale objects became another container's "fresh" build.
#
# Guards are the same as pair_build_lib.sh: no server in this pair's container, no compiler already
# running on the box, and on GX enough free memory that no pack is loading. A build killed mid-JIT
# leaves torch's `lock` baton behind, and every later import then sleeps 3 x 600 s on it -- the baton
# is cleared here, before the build, never during someone else's run.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

PAIR_ARG=${1:?usage: pair_build_ext.sh <pair> <sha> [--side p7|gx|both]}
SHA=${2:?usage: pair_build_ext.sh <pair> <sha> [--side p7|gx|both]}
shift 2
SIDES=both
while [ $# -gt 0 ]; do
  case "$1" in
    --side) SIDES=${2:?--side needs p7|gx|both}; shift ;;
    *) pair_die "unknown flag $1" ;;
  esac
  shift
done
pair_load "$PAIR_ARG"
REL=$(pair_release_dir "$SHA")
[ -d "$REL/src" ] || pair_die "release ${SHA:0:8} is not cut at $REL"
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json")
JOBS=${MAX_JOBS:-8}; [ "$JOBS" -le 8 ] || JOBS=8
EXT_TIMEOUT=${EXT_TIMEOUT:-1800}
GX_BUILD_AVAIL_GIB=${GX_BUILD_AVAIL_GIB:-60}
MOUNT=${PAIR_MOUNT:-/pair}
# Dry run (stub-ssh harness only): say what would be built, write nothing (see pair_build_lib.sh).
if [ "$PAIR_DRY" = 1 ]; then
  for side in p7 gx; do
    [ "$SIDES" = both ] || [ "$SIDES" = "$side" ] || continue
    ct=$([ "$side" = p7 ] && echo "$P7CT" || echo "$GXCT")
    sm=$([ "$side" = p7 ] && echo "$P7_ARCH" || echo "$GX_ARCH")
    pair_step "DRY EXT_BUILD $side $ct arch=$(pair_cuda_arch_for "$sm") jobs=$JOBS build=$BUILD_ID"
  done
  exit 0
fi

side_docker() {  # <side> <args...>
  local side=$1; shift
  if [ "$side" = p7 ]; then docker "$@"; else pair_gxsh "docker $(printf '%q ' "$@")"; fi
}
side_hostsh() { local side=$1; if [ "$side" = p7 ]; then bash -c "$2"; else pair_gxsh "$2"; fi; }

build_side() {
  local side=$1 ct sm archlist out rc
  if [ "$side" = p7 ]; then ct=$P7CT; sm=$P7_ARCH; else ct=$GXCT; sm=$GX_ARCH; fi
  archlist=$(pair_cuda_arch_for "$sm")
  local rel_c="$MOUNT/releases/$SHA"

  [ "$(side_docker "$side" inspect -f '{{.State.Running}}' "$ct" 2>/dev/null)" = true ] \
    || { pair_step "EXT_SKIP $side $ct not_running"; return 1; }
  [ -z "$(side_docker "$side" exec "$ct" pgrep -f worker_seam_server.py 2>/dev/null)" ] \
    || { pair_step "EXT_REFUSE $side server_running_in=$ct"; return 1; }
  local builds
  builds=$(side_hostsh "$side" "pgrep -c -x 'ninja|cc1plus|nvcc|cicc' 2>/dev/null || true")
  local waited=0 cap=${PAIR_BUILD_WAIT_SECS:-1800}
  while [ "${builds:-0}" != 0 ] && [ "$waited" -lt "$cap" ]; do
    pair_step "EXT_WAIT $side compiler_running=$builds waited=${waited}s (another build owns the host)"
    sleep 30; waited=$((waited + 30))
    builds=$(side_hostsh "$side" "pgrep -c -x 'ninja|cc1plus|nvcc|cicc' 2>/dev/null || true")
  done
  [ "${builds:-0}" = 0 ] || { pair_step "EXT_REFUSE $side compiler_running=$builds waited=${waited}s"; return 1; }
  if [ "$side" = gx ]; then
    local avail
    avail=$(side_hostsh gx "awk '/^MemAvailable:/{print int(\$2/1048576)}' /proc/meminfo")
    [ "${avail:-0}" -ge "$GX_BUILD_AVAIL_GIB" ] \
      || { pair_step "EXT_REFUSE gx mem_avail=${avail:-?}GiB need=$GX_BUILD_AVAIL_GIB"; return 1; }
  fi

  # stale baton: only when nothing is building and no server of ours runs (both checked above)
  side_docker "$side" exec "$ct" sh -c "mkdir -p $rel_c/ext/exllamav3_ext; rm -f $rel_c/ext/exllamav3_ext/lock" \
    || { pair_step "EXT_FAIL $side lock_clear"; return 1; }

  pair_step "EXT_BUILD $side $ct arch=$archlist jobs=$JOBS build=$BUILD_ID"
  # PYTHONPATH points at the release tree itself, not at /src/exllamav3 (= /pair/current/src):
  # the cache must be built from this release even before `current` is flipped to it.
  local inner="cd $rel_c/src && nice -n 15 env MAX_JOBS=$JOBS NINJAFLAGS=-j$JOBS MAKEFLAGS=-j$JOBS \
PYTHONPATH=$rel_c/src TORCH_CUDA_ARCH_LIST=$archlist TORCH_EXTENSIONS_DIR=$rel_c/ext \
EXPECT_SRC=$rel_c/src/exllamav3 timeout $EXT_TIMEOUT python3 -u -"
  if [ "$side" = p7 ]; then
    out=$(docker exec -i "$ct" bash -lc "$inner" < "$HERE/pair_ext_probe.py" 2>&1); rc=$?
  else
    out=$(pair_gxsh "docker exec -i $(printf %q "$ct") bash -lc $(printf %q "$inner")" \
          < "$HERE/pair_ext_probe.py" 2>&1); rc=$?
  fi
  [ -n "${PAIR_LOG:-}" ] && printf '%s\n' "$out" >> "$PAIR_LOG"
  if [ "$rc" != 0 ] || ! printf '%s' "$out" | grep -q '^EXT_BUILD_OK '; then
    pair_step "EXT_FAIL $side rc=$rc $(printf '%s' "$out" | grep -E 'Error|error:|FAILED|EXT_' | tail -4 | tr '\n' ' ')"
    return 1
  fi
  local so sha_so
  so=$(printf '%s\n' "$out" | sed -n 's/^EXT_SO //p' | tail -1)
  [ -n "$so" ] || { pair_step "EXT_FAIL $side no_so_path"; return 1; }
  if [ "$side" = p7 ]; then
    sha_so=$(sha256sum "$REL/ext/exllamav3_ext/$(basename "$so")" 2>/dev/null | cut -d' ' -f1)
  else
    sha_so=$(pair_gxsh "sha256sum '$REL/ext/exllamav3_ext/$(basename "$so")' 2>/dev/null" | cut -d' ' -f1)
  fi
  [ -n "$sha_so" ] || { pair_step "EXT_FAIL $side so_not_in_release_cache"; return 1; }
  local torchv
  torchv=$(side_docker "$side" exec "$ct" python3 -c 'import torch;print(torch.__version__)' 2>/dev/null | tail -1)
  local stamp
  stamp=$(python3 - "$BUILD_ID" "$SHA" "$archlist" "$torchv" "$sha_so" "$side" <<'PY'
import datetime, json, sys
bid, sha, archs, torchv, so, side = sys.argv[1:7]
print(json.dumps({
    "build_id": bid, "source_sha": sha,
    "arch_list": archs.split(";") if ";" in archs else archs.split(","),
    "torch": torchv, "sha256": so, "host": side,
    "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, indent=2, sort_keys=True))
PY
)
  if [ "$side" = p7 ]; then
    printf '%s\n' "$stamp" > "$REL/ext/BUILD_STAMP.json"
  else
    printf '%s\n' "$stamp" | pair_gxsh "cat > '$REL/ext/BUILD_STAMP.json'"
  fi
  pair_step "EXT_OK $side torch=$torchv arch=$archlist so=$(basename "$so") sha256=$sha_so"
  # `ninja -n` in the cache: a non-empty build plan means a later runtime import would rebuild the
  # extension *under a live server* (the SIGBUS that has taken pairs down), so say so now.
  local ndry
  ndry=$(side_docker "$side" exec "$ct" sh -c "cd $rel_c/ext/exllamav3_ext && ninja -n 2>&1" 2>/dev/null \
         | grep -E '^\[' | head -3 | tr '\n' ';')
  if [ -n "$ndry" ]; then
    pair_step "EXT_NINJA_STALE $side ninja_would_run=$ndry"
  else
    pair_step "EXT_NINJA_CLEAN $side"
  fi
  return 0
}

rc=0
case "$SIDES" in
  both) build_side p7 || rc=1; [ "$rc" = 0 ] && { build_side gx || rc=1; } ;;
  p7) build_side p7 || rc=1 ;;
  gx) build_side gx || rc=1 ;;
  *) pair_die "bad --side $SIDES" ;;
esac
exit $rc
