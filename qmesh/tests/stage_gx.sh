#!/usr/bin/env bash
# Stage this worktree onto the box that runs the cold experts.
#
# The GX trees are *copies* (rsynced, no .git), not shares: a change committed on P7 is invisible to a
# container until someone stages it. That is how a ranking change can be "landed" and still not be
# running on the GX -- and why every driver that trusts container code stages first and then proves,
# by sha, that the container is looking at the same bytes.
#
#   GX=treadall@10.0.4.33 CX_CT=seam-gx2-4460 bash qmesh/tests/stage_gx.sh
#
# Not copied: .git, qmesh/native/obj (build intermediates), qmesh/native/bin (the aarch64 libraries
# are built in the container's own CUDA image, and a stale P7-side binary must never overwrite them),
# __pycache__. Exit 0 staged and verified, 1 rsync failed, 2 a staged file does not match.
set -uo pipefail
GX=${GX:-treadall@10.0.4.33}
CT=${CX_CT:-seam-gx2-4460}
SRC=${SRC:-/home/treadall/recipe-seam-robust}
DST=${DST:-/home/treadall/recipe-seam-robust}
CTWT=${CTWT:-/src/exllamav3}

rsync -a --exclude .git --exclude qmesh/native/obj --exclude qmesh/native/bin --exclude __pycache__ \
  "$SRC/" "$GX:$DST/" || { echo "STAGE FAIL rsync $SRC/ -> $GX:$DST/"; exit 1; }

# The files whose behaviour the seam's two halves have to agree on. A stale one of these is exactly
# the class of bug this script exists for: the transport cannot tell an old numbering from a new one.
bad=0
for f in qmesh/worker_seam_server.py qmesh/placement.py qmesh/native/qes.py \
         qmesh/tests/cx_direct_check.py qmesh/tests/placement_sha_refusal.py \
         qmesh/tests/refusal_stub.sh qmesh/native/qmesh_expert_seam.cu; do
  l=$(sha256sum "$SRC/$f" | cut -c1-16)
  r=$(ssh -o BatchMode=yes "$GX" "docker exec $CT sha256sum $CTWT/$f" 2>/dev/null | cut -c1-16)
  if [ "$l" = "$r" ]; then
    echo "STAGE $f sha=$l"
  else
    echo "STAGE FAIL $f local=$l ct=${r:-missing}"
    bad=1
  fi
done
# The libraries are built in the container, so their shas are the container's business; print them so
# a log can be matched against the run that produced a number.
ssh -o BatchMode=yes "$GX" "docker exec $CT bash -lc 'cd $CTWT/qmesh/native/bin && sha256sum * 2>/dev/null'" \
  | awk 'NF == 2 {printf "STAGE lib %s sha=%.16s\n", $2, $1}'
[ "$bad" = 0 ] || exit 2
echo "STAGE OK host=$GX ct=$CT tree=$CTWT"
