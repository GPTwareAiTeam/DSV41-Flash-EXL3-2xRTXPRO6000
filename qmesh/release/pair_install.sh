#!/usr/bin/env bash
# pair_install.sh -- put release tooling into the pairs' script dir, atomically, and only when no
# queue ticket is in flight.
#
#   pair_install.sh [--src <dir>] [--dst <dir>] <file>...
#
# Why the guard: bash reads a script file incrementally, so replacing one *in place* while a ticket is
# executing it makes the reader pick up shifted lines from the new text. On 2026-11-03 01:29Z a prod
# prefill run exited 1 with `ir_arm_ids.py: command not found` and `drv_host: unbound variable` from
# line numbers that had already run, because another agent copied a patched pair_arm.sh over the file
# while the ticket had it open; the measurement was otherwise fine (792.5 tok/s) and the run was
# thrown away. A rename would have kept the old inode alive for the reader, a copy does not, and
# neither is safe to do under a live ticket -- so this refuses instead of relying on the writer.
#
# The destination defaults to the artifacts script dir next to this checkout's artifact root; the
# files come from the repo tree (qmesh/release/) unless --src says otherwise. Every file is syntax
# checked before it moves, and the move is a rename so a *reader* that starts after it gets either
# the whole old file or the whole new one, never a mix.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)

die() { echo "INSTALL_FAIL $*" >&2; exit 2; }
step() { echo "INSTALL $*"; }

SRC=""
DST=""
files=()
while [ $# -gt 0 ]; do
  case "$1" in
    --src) SRC=${2:?--src needs a dir}; shift ;;
    --dst) DST=${2:?--dst needs a dir}; shift ;;
    -*) die "unknown flag $1" ;;
    *) files+=("$1") ;;
  esac
  shift
done
[ "${#files[@]}" -gt 0 ] || die "usage: pair_install.sh [--src <dir>] [--dst <dir>] <file>..."

if [ -z "$SRC" ]; then
  # Prefer the release tree: the installed copies must equal what a release carries, never the
  # other way round.
  tree=${PAIR_RELEASE_TREE:-/home/treadall/recipe-pair-release}
  [ -d "$tree/qmesh/release" ] && SRC=$tree/qmesh/release || SRC=$HERE
fi
[ -d "$SRC" ] || die "src $SRC is not a directory"

if [ -z "$DST" ]; then
  # <artifacts>/scripts; this file may be run from the repo or from its installed copy.
  for cand in "$HERE" "$(dirname "$HERE")/scripts" "/home/treadall/recipe-artifacts/gpu0-gx1/scripts"; do
    [ -d "$cand" ] && [ -f "$cand/pair_lib.sh" ] && DST=$cand && break
  done
fi
[ -n "$DST" ] && [ -d "$DST" ] || die "cannot resolve the installed script dir (pass --dst)"

# One queue per pair; a live pid in <queue>/running owns the box and owns the scripts it is reading.
for q in dev prod; do
  for root in "$(dirname "$DST")/queue" /home/treadall/recipe-artifacts/gpu0-gx1/queue; do
    rf=$root/$q/running
    [ -f "$rf" ] || continue
    pid=$(tr ' ' '\n' < "$rf" | sed -n 's/^pid=\([0-9]*\)$/\1/p' | head -1)
    agent=$(tr ' ' '\n' < "$rf" | sed -n 's/^agent=\([^ ]*\)$/\1/p' | head -1)
    [ -n "$pid" ] || continue
    if kill -0 "$pid" 2>/dev/null; then
      die "ticket_in_flight queue=$q agent=${agent:-?} pid=$pid -- installs wait for the queue to be idle"
    fi
  done
done
step "queue_idle src=$SRC dst=$DST"

n=0
for f in "${files[@]}"; do
  src=$SRC/$f
  [ -f "$src" ] || die "missing $src"
  case "$f" in
    *.sh) bash -n "$src" || die "syntax $f" ;;
    *.py) python3 -m py_compile "$src" || die "syntax $f" ;;
  esac
  tmp=$DST/.$f.new.$$
  cp -- "$src" "$tmp" || die "copy $f"
  case "$f" in *.py) ;; *) chmod +x "$tmp" ;; esac
  mv -f -- "$tmp" "$DST/$f" || die "rename $f"
  cmp -s -- "$src" "$DST/$f" || die "verify $f"
  step "installed $f"
  n=$((n+1))
done
echo "INSTALL_OK n=$n dst=$DST"
