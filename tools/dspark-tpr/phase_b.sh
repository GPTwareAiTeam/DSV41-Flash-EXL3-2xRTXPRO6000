#!/usr/bin/env bash
# phase_b.sh -- instrumented arms for the tpr ticket: per-position verify table (EXL3_DSPARK_POSSTATS)
# and the pair's own round cost per verify row (EXL3_ROUND_PROF width_fit).
#
#   phase_b.sh <tag>
#
# Client tree: /artifacts/dspark-tpr/src (host /home/treadall/recipe-artifacts/dspark-tpr/src), the
# release d330dcf5 tree overlaid with the DSparkTune worktree. The server keeps serving d330dcf5 and
# the protocol is unchanged; only the client half is instrumented, which is where the gate lives.
#
# Outputs under $OUTDIR: <tag>.jsonl (all rows+SUMMARY), <tag>-<arm>.log, SUMMARY.<tag>.
set -uo pipefail
TAG=${1:?usage: phase_b.sh <tag>}
OUTDIR=/home/treadall/recipe-artifacts/dspark-tpr/logs
ARM=/home/treadall/recipe-dspark-tpr/qmesh/release/pair_arm.sh
SRC=/home/treadall/recipe-artifacts/dspark-tpr/src   # host path; pair_arm maps it to /artifacts/...
mkdir -p "$OUTDIR"
# A --src arm must never build over the pair's release extension: a different source path gives torch a
# different hash, so load() compiles fresh, and landing that in /pair/extcache put dev into DRIFT at
# 20:59:31. Own cache, prebuilt out of band by prebuild_ext.py, and refuse to run without it.
EXT_CACHE=/artifacts/dspark-tpr/extcache
[ -n "$(ls -A "$EXT_CACHE"/exllamav3_ext 2>/dev/null)" ] || {
  echo "ABORT reason=ext_cache_missing path=$EXT_CACHE (run prebuild_ext.py first)"; exit 4; }
J=$OUTDIR/$TAG.jsonl
S=$OUTDIR/SUMMARY.$TAG
: > "$S"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$S"; }
# One gate for every ticket that judges this server -- canonical fields, the log read inside the
# container, VmSwap and swap-growth terms: see gate_gx.sh. Sourced, so a fix lands in one place.
HERE=/home/treadall/recipe-artifacts/dspark-tpr
source "$HERE/gate_gx.sh"
# Same box-health contract as km_sweep: the pack lives on GX2 and a paging box invalidates the arm.
GX=$(awk '!/^#/ && NF && $1=="dev"{print $8; exit}' /home/treadall/recipe-dspark-tpr/qmesh/release/pairs.conf)
SWAP_GROW_MIB=${SWAP_GROW_MIB:-256}
# The dev server writes its own log at /tmp/seam_server.log on GX2 (truncated per start), and it says
# what state the box is in: SEAM_SERVER_SLAB at load with avail/swapped, then WARNs if the pack is
# paging. A loaded GX always leaves only ~0.3-1.0 GiB available (116 GiB pack on a 121.6 GiB box), so
# free memory is not the test; the server's own verdict plus swap growth since its load line is.
gxstat() {
  ssh -o BatchMode=yes -o ConnectTimeout=5 "$GX" \
    "awk '/^MemAvailable|^SwapFree|^SwapTotal/{k=\$1; sub(/:\$/,\"\",k); printf \"%s=%s \", k, \$2}' /proc/meminfo; \
     echo; grep -E 'SEAM_SERVER_SLAB |SEAM_SERVER_SLAB_WARN|SEAM_SERVER_SWAP_WARN|SEAM_SLACK_WARN' /tmp/seam_server.log 2>/dev/null | tail -3" \
    2>/dev/null || echo "MemAvailable=unknown"
}
say "PHASEB_START tag=$TAG src=$SRC"
# Same health gate as km_sweep: an 8-token arm that comes back over 300 ms/token is a paging box.
probe() {
  local attempt dp log
  for attempt in 1 2; do
    log=$OUTDIR/$TAG-probe$([ "$attempt" = 1 ] || echo "-r$attempt").log
    echo "eta_min=1 doing=$TAG probe" > "$GPU_RUN_STATUS"
    bash "$ARM" dev --rows 1 --max-new 8 --num-draft 5 --timeout 600 --src "$SRC" \
      --env TORCH_EXTENSIONS_DIR=/artifacts/dspark-tpr/extcache \
        --out "$OUTDIR/$TAG-probe.jsonl" > "$log" 2>&1
    dp=$(grep -oE '"decode_ms_per_tok": *[0-9.]+' "$log" | head -1 | grep -oE '[0-9.]+$')
    say "PROBE attempt=$attempt src decode_ms_per_tok=${dp:-none} pos=$(grep -c '"pos": {' "$log")"
    if [ -z "${dp:-}" ]; then
      say "PROBE_FAIL attempt=$attempt no row -- see $log"
      if [ "$attempt" = 1 ] && grep -qE "No CUDA GPUs are available|CUDA driver version is insufficient|NVIDIA driver" "$log"; then
        say "PROBE_RETRY attempt=1 reason=no-gpu-in-container (transient)"; sleep 25; continue
      fi
      return 1
    fi
    awk -v d="$dp" 'BEGIN{exit !(d > 300)}' && { say "PROBE_DEGRADED ${dp}ms/token > 300"; return 1; }
    return 0
  done
  return 1
}
gx_gate || { say "PHASEB_ABORT tag=$TAG reason=gx_mem"; exit 3; }
probe || { say "PHASEB_ABORT tag=$TAG reason=probe"; exit 3; }
run() {  # run <arm-tag> [pair_arm args...]
  local t=$1; shift
  local log=$OUTDIR/$TAG-$t.log
  echo "eta_min=4 doing=$TAG $t" > "$GPU_RUN_STATUS"
  say "ARM_START tag=$t"
  say "GX_MEM $t $(gxmem)"
  bash "$ARM" dev --rows 3 --max-new 256 --num-draft 5 --timeout 900 --src "$SRC" \
      --env TORCH_EXTENSIONS_DIR=/artifacts/dspark-tpr/extcache \
      --out "$J" "$@" > "$log" 2>&1
  local rc=$?
  say "ARM_END tag=$t rc=$rc log=$log"
  grep -E "^SUMMARY" "$log" | sed "s/^/  [$t] /" | tee -a "$S" >/dev/null
  if [ $rc -ne 0 ]; then
    say "ARM_FAIL tag=$t rc=$rc tail=$(grep -viE '^(warn|info)' "$log" | tail -n 3 | tr '\n' ' | ')"
  fi
}
# 1. The shipping gate (auto, conf 0.5) with the per-position table: where does the window die, and is
#    the score it dies on calibrated?
run pos-auto --env EXL3_DSPARK_POSSTATS=1
# 2. A fixed threshold at the calibrator's default 0.5: the same table without the adaptive LO pin.
run pos-thr50 --env EXL3_DSPARK_POSSTATS=1 --env EXL3_DSPARK_GATE=thr --env EXL3_DSPARK_CONF=0.5
# 3. Width fit: threshold 0 forces the widest window, so every verify width actually appears and
#    roundprof's width_fit gives this pair's intercept_ms and slope_ms_per_row -> EXL3_DSPARK_EV_COST.
run prof-wide --env EXL3_ROUND_PROF=1 --env EXL3_DSPARK_GATE=thr --env EXL3_DSPARK_CONF=0
# 4. The candidate the cost fit points at: never truncate (conf 0 keeps the full K=5 window). Same
#    session and same client tree as pos-auto, so it is a direct A/B of one knob.
run pos-thr0 --env EXL3_DSPARK_POSSTATS=1 --env EXL3_DSPARK_GATE=thr --env EXL3_DSPARK_CONF=0
say "PHASEB_DONE tag=$TAG"
