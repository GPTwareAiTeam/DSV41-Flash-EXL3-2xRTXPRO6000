#!/usr/bin/env bash
# pair_build_lib.sh -- build qmesh/native/bin/libqmesh_expert_seam.<arch>.so for one release.
#
#   pair_build_lib.sh <pair> <sha> [--side p7|gx|both] [--force]
#   pair_build_lib.sh verify <pair> <sha> [--side p7|gx|both]
#
# The library is built *inside* the release's own tree, from the release's own sources, and lands in
# that release's bin/ next to a .so.stamp that names the commit it came from. A shared library is not
# byte-reproducible across containers (compiler, headers, torch), so the stamp -- not a cross-box
# .so hash -- is what pair_check compares. Nothing is ever copied between boxes: each box builds for
# its own arch (x86_64/sm_120 on P7, aarch64/sm_121 on GX).
#
# Guards, because an in-place build under a live server has already killed a box (SIGBUS):
#   * no worker_seam_server.py in this pair's own container,
#   * no compiler already active anywhere on the box (ninja/cc1plus/nvcc/cicc),
#   * on GX: MemAvailable above GX_BUILD_AVAIL_GIB (a loading server holds the pack -> low avail).
# The build runs niced with one nvcc; it writes to a temp dir and mv(1)s into place.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=pair_lib.sh
. "$HERE/pair_lib.sh"

MODE=build
if [ "${1:-}" = verify ]; then MODE=verify; shift; fi
PAIR_ARG=${1:?usage: pair_build_lib.sh [verify] <pair> <sha> [--side p7|gx|both]}
SHA=${2:?usage: pair_build_lib.sh [verify] <pair> <sha> [--side p7|gx|both]}
shift 2
SIDES="both"; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --side) SIDES=${2:?--side needs p7|gx|both}; shift ;;
    --force) FORCE=1 ;;
    *) pair_die "unknown flag $1" ;;
  esac
  shift
done
pair_load "$PAIR_ARG"
REL=$(pair_release_dir "$SHA")
[ -d "$REL/src" ] || pair_die "release ${SHA:0:8} is not cut at $REL (run pair_cut.sh first)"
BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' "$REL/RELEASE.json") \
  || pair_die "no RELEASE.json in $REL"
SEAM_FP=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["seam"]["fingerprint"])' "$REL/RELEASE.json")
GX_BUILD_AVAIL_GIB=${GX_BUILD_AVAIL_GIB:-60}
# Dry run (stub-ssh harness only): say what would be built, write nothing. A manifest is refused
# without real stamps, so a PAIR_DRY run can never leave a pair looking updated.
if [ "$PAIR_DRY" = 1 ]; then
  for side in p7 gx; do
    [ "$SIDES" = both ] || [ "$SIDES" = "$side" ] || continue
    ct=$([ "$side" = p7 ] && echo "$P7CT" || echo "$GXCT")
    sm=$([ "$side" = p7 ] && echo "$P7_ARCH" || echo "$GX_ARCH")
    pair_step "DRY LIB_BUILD $side $ct arch=$(pair_arch_for "$sm") sm=$sm build=$BUILD_ID"
  done
  exit 0
fi

side_ct() { [ "$1" = p7 ] && echo "$P7CT" || echo "$GXCT"; }
side_sm() { [ "$1" = p7 ] && echo "$P7_ARCH" || echo "$GX_ARCH"; }
side_libdir() { [ "$1" = p7 ] && echo /usr/lib/x86_64-linux-gnu || echo /usr/lib/aarch64-linux-gnu; }
# One transport for "run this docker command on the side's host": P7 is local, GX goes through ssh.
side_docker() {  # <side> <args...>
  local side=$1; shift
  if [ "$side" = p7 ]; then
    docker "$@"
  else
    pair_gxsh "docker $(printf '%q ' "$@")"
  fi
}
side_hostsh() {  # <side> <shell-string>
  local side=$1
  if [ "$side" = p7 ]; then bash -c "$2"; else pair_gxsh "$2"; fi
}

build_side() {
  local side=$1 ct arch sm libdir inner out
  ct=$(side_ct "$side"); sm=$(side_sm "$side")
  arch=$(pair_arch_for "$sm"); libdir=$(side_libdir "$side")
  local lib="src/qmesh/native/bin/$PAIR_LIB.$arch.so"

  # ---- guards ------------------------------------------------------------------------------------
  local running
  running=$(side_docker "$side" inspect -f '{{.State.Running}}' "$ct" 2>/dev/null)
  [ "$running" = true ] || { pair_step "LIB_SKIP $side $ct not_running"; return 1; }
  if [ -n "$(side_docker "$side" exec "$ct" pgrep -f worker_seam_server.py 2>/dev/null)" ]; then
    pair_step "LIB_REFUSE $side server_running_in=$ct (never build under a live server)"; return 1
  fi
  local builds
  builds=$(side_hostsh "$side" "pgrep -c -x 'ninja|cc1plus|nvcc|cicc' 2>/dev/null || true")
  local waited=0 cap=${PAIR_BUILD_WAIT_SECS:-1800}
  while [ "${builds:-0}" != 0 ] && [ "$waited" -lt "$cap" ]; do
    pair_step "LIB_WAIT $side compiler_running=$builds waited=${waited}s (another build owns the host)"
    sleep 30; waited=$((waited + 30))
    builds=$(side_hostsh "$side" "pgrep -c -x 'ninja|cc1plus|nvcc|cicc' 2>/dev/null || true")
  done
  [ "${builds:-0}" = 0 ] || { pair_step "LIB_REFUSE $side compiler_running=$builds waited=${waited}s"; return 1; }
  if [ "$side" = gx ]; then
    local avail
    avail=$(side_hostsh gx "awk '/^MemAvailable:/{print int(\$2/1048576)}' /proc/meminfo")
    [ "${avail:-0}" -ge "$GX_BUILD_AVAIL_GIB" ] \
      || { pair_step "LIB_REFUSE gx mem_avail=${avail:-?}GiB need=$GX_BUILD_AVAIL_GIB (a server is loading)"; return 1; }
  fi

  if [ "$MODE" = verify ]; then
    local stamp="src/qmesh/native/bin/$PAIR_LIB.$arch.so.stamp" got want
    if [ "$side" = p7 ]; then
      got=$(cat "$REL/$stamp" 2>/dev/null)
      want=$(sha256sum "$REL/$lib" 2>/dev/null | cut -d' ' -f1)
    else
      got=$(pair_gxsh "cat '$REL/$stamp' 2>/dev/null")
      want=$(pair_gxsh "sha256sum '$REL/$lib' 2>/dev/null" | cut -d' ' -f1)
    fi
    python3 - "$got" "$want" "$BUILD_ID" "$SHA" "$arch" "$side" <<'PY' || return 1
import json, sys
got, want, bid, sha, arch, side = sys.argv[1:7]
try:
    d = json.loads(got)
except Exception:
    print(f"LIB_STAMP_MISSING {side} {arch}"); sys.exit(1)
bad = [k for k, v in (("build_id", bid), ("source_sha", sha), ("arch", arch)) if d.get(k) != v]
if bad:
    print(f"LIB_STAMP_MISMATCH {side} {' '.join(f'{k}={d.get(k)}' for k in bad)}"); sys.exit(1)
if want and d.get("sha256") != want:
    print(f"LIB_SHA_MISMATCH {side} on_disk={want} stamp={d.get('sha256')}"); sys.exit(1)
print(f"LIB_OK {side} {arch} sha256={d.get('sha256')} build={d.get('build_id')}")
PY
    return $?
  fi

  # ---- build -------------------------------------------------------------------------------------
  local tmp="/pair/releases/$SHA/.lib-$arch-$$"
  local flags="-O2 -std=c++17 -arch=$sm -Xcompiler -fPIC -shared -x cu qmesh_expert_seam.cu -o $tmp/$PAIR_LIB.$arch.so -libverbs -lcuda -lcudart -Xlinker -rpath=$libdir -I."
  inner="set -e; cd /pair/releases/$SHA/src/qmesh/native; rm -rf $tmp; mkdir -p $tmp; \
         nice -n 15 env MAX_JOBS=8 /usr/local/cuda/bin/nvcc $flags; \
         mv -f $tmp/$PAIR_LIB.$arch.so bin/$PAIR_LIB.$arch.so; rmdir $tmp; \
         sha256sum bin/$PAIR_LIB.$arch.so"
  pair_step "LIB_BUILD $side $ct arch=$arch sm=$sm build=$BUILD_ID"
  out=$(side_docker "$side" exec "$ct" bash -lc "$inner" 2>&1)
  local rc=$?
  if [ "$rc" != 0 ]; then
    pair_step "LIB_FAIL $side rc=$rc $(printf '%s' "$out" | tail -5 | tr '\n' ' ')"
    return 1
  fi
  local sha_lib
  sha_lib=$(printf '%s\n' "$out" | awk '/^[0-9a-f]{64} /{print $1}' | tail -1)
  [ -n "$sha_lib" ] || { pair_step "LIB_FAIL $side no_sha_in_output"; return 1; }

  local cc
  cc=$(side_docker "$side" exec "$ct" /usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -1 | tr -s ' ')
  local stamp
  stamp=$(python3 - "$BUILD_ID" "$SHA" "$arch" "$sm" "$SEAM_FP" "$sha_lib" "$cc" "$flags" "$side" <<'PY'
import datetime, json, sys
bid, sha, arch, sm, fp, libsha, cc, flags, side = sys.argv[1:10]
print(json.dumps({
    "build_id": bid, "source_sha": sha, "arch": arch, "sm": sm,
    "seam": fp, "sha256": libsha, "compiler": cc, "flags": flags,
    "host": side, "built_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, indent=2, sort_keys=True))
PY
)
  local stamp_path="$REL/src/qmesh/native/bin/$PAIR_LIB.$arch.so.stamp"
  if [ "$side" = p7 ]; then
    printf '%s\n' "$stamp" > "$stamp_path" || { pair_step "LIB_FAIL $side stamp_write"; return 1; }
    chmod 644 "$stamp_path"
  else
    printf '%s\n' "$stamp" | pair_gxsh "cat > '$stamp_path'" || { pair_step "LIB_FAIL gx stamp_write"; return 1; }
    pair_gxsh "chmod 644 '$stamp_path'"
  fi
  pair_step "LIB_OK $side arch=$arch sha256=$sha_lib build=$BUILD_ID stamp=$(basename "$stamp_path")"
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
