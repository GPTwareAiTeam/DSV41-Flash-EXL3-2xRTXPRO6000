#!/bin/bash
# P7-only: verify-round cost at 6/12/24 rows with the MoE removed (null remote). No GX server involved.
set -u
CACHE=/artifacts/dspark-tpr/extcache
[ -n "$(ls -A $CACHE/exllamav3_ext 2>/dev/null)" ] || { echo "ABORT reason=ext_cache_missing"; exit 4; }
echo "eta_min=12 doing=attn_rows W=6/12/24 null remote" > "${GPU_RUN_STATUS:-/tmp/gpu_run_status}"
docker exec \
  -e CUDA_VISIBLE_DEVICES=1 \
  -e PYTHONPATH=/artifacts/dspark-tpr/src \
  -e TORCH_EXTENSIONS_DIR=$CACHE \
  -e EXL3_DSPARK_GATE=thr -e EXL3_DSPARK_CONF=0 \
  -e EXL3_MOE_SPLIT_FUSED=0 -e EXL3_MOE_CPU_SWAP=0 \
  -e DSPARK_BLOCK=23 -e KS=5,11,23 -e ROUNDS=8 -e WARM=2 \
  seam-p7-dev bash -lc 'cd /tmp && python3 -u /artifacts/dspark-tpr/src/qmesh/attn_rows.py'
