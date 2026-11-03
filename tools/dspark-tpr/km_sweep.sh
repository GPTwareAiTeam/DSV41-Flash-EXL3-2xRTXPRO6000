#!/usr/bin/env bash
# km_sweep.sh -- one config's K map on the dev pair, all arms in one ticket.
#
#   km_sweep.sh <tag> <k1,k2,...> [extra pair_arm.sh args...]
#
# Every arm is the SHIPPING client env (qmesh/release/pair_arm.sh; its override flags defaulted) with
# exactly one knob changed: NUM_DRAFT, plus whatever the caller passes (e.g. --env EXL3_DSPARK_GATE=ev).
# Prompts: the shipping held list's first three -- code (Kotlin rate limiter), prose (lighthouse
# story), structured (release notes) -- and, separately, the 32k prose passage the 17:58 baseline
# used (/artifacts/ppl-4460/prose_32k.json), 256 greedy tokens each.
#
# Outputs under $OUTDIR:
#   <tag>.jsonl        every row and SUMMARY line (one JSON object per line, appended in order)
#   <tag>-<arm>.log    the full client stdout/stderr of that arm
#   SUMMARY.<tag>      one "ARM" line per arm with its SUMMARY line, so the table is greppable
set -uo pipefail
TAG=${1:?usage: km_sweep.sh <tag> <k1,k2,...> [extra pair_arm args]}
KS=${2:?ks}
shift 2
EXTRA=("$@")
OUTDIR=/home/treadall/recipe-artifacts/dspark-tpr/logs
ARM=/home/treadall/recipe-dspark-tpr/qmesh/release/pair_arm.sh
# Prompts are read *inside* the container, so this is the container path (/artifacts is the host's
# recipe-artifacts); the host path made every 32k arm die on FileNotFoundError (kmC, 20:33-20:35).
PROMPTS_32K=/artifacts/ppl-4460/prose_32k.json
mkdir -p "$OUTDIR"
J=$OUTDIR/$TAG.jsonl
S=$OUTDIR/SUMMARY.$TAG
: > "$S"
NK=$(echo "$KS" | tr ',' '\n' | wc -l)
REMAIN=$(( NK * 2 + 1 ))
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$S"; }
# The dev server's cold pack lives on GX2, so that box's free memory decides whether an arm measures
# the engine or the pager: kmC ran with 213 MiB available and 1.46 GiB swapped out, which is why its
# numbers are a direction and not a result. Print it per arm and refuse to start one on a tight box.
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
gx_gate() {
  local st m avail tot free swapu base
  st=$(gxstat)
  m=$(printf '%s\n' "$st" | head -1)
  avail=$(printf '%s' "$m" | grep -oE 'MemAvailable=[0-9]+' | cut -d= -f2)
  tot=$(printf '%s' "$m" | grep -oE 'SwapTotal=[0-9]+' | cut -d= -f2)
  free=$(printf '%s' "$m" | grep -oE 'SwapFree=[0-9]+' | cut -d= -f2)
  swapu=$(( (${tot:-0} - ${free:-0}) / 1024 ))
  base=$(printf '%s' "$st" | grep -oE 'swapped_mib=[0-9]+' | tail -1 | cut -d= -f2)
  say "GX_MEM avail_kb=${avail:-?} swap_used_mib=$swapu swapped_at_load=${base:-none} $(printf '%s' "$st" | grep -oE 'SEAM_SERVER_SLAB_WARN|SEAM_SERVER_SWAP_WARN|SEAM_SLACK_WARN' | tr '\n' ',')"
  if printf '%s' "$st" | grep -qE 'SEAM_SERVER_SLAB_WARN|SEAM_SERVER_SWAP_WARN|SEAM_SLACK_WARN'; then
    say "GX_TIGHT the running server logged a paging WARN: requeue after it is reloaded"
    return 1
  fi
  if [ -n "${base:-}" ] && [ "$swapu" -gt $((base + SWAP_GROW_MIB)) ]; then
    say "GX_TIGHT swap_used_mib=$swapu is $((swapu - base)) MiB over the load line's ${base} (limit ${SWAP_GROW_MIB})"
    return 1
  fi
  return 0
}
say "SWEEP_START tag=$TAG ks=$KS extra=${EXTRA[*]-} arm=$ARM"
gx_gate || { say "SWEEP_ABORT tag=$TAG reason=gx_mem"; exit 3; }
# A health gate before the sweep spends the ticket. On 2026-11-02 20:11 the same 8-token arm the
# 11:46 canary ran at ttft 1.4 s / 111 ms-per-token came back at 21 s / 2685 ms-per-token: the GX
# box was paging and every number after it would have described a sick box, not a K. A single 8-token
# arm is 3 s when healthy, so gate on it: >300 ms/token means requeue, not measure.
probe() {
  # Two attempts: on 2026-11-02 20:30 the first probe after a pair server restart died in
  # model.load with "No CUDA GPUs are available" -- the client container had come up without the
  # device while the host's nvidia nodes were being republished. That is an infrastructure race,
  # not a sick box: wait and retry once before calling it degraded.
  local attempt dp log
  for attempt in 1 2; do
    log=$OUTDIR/$TAG-probe$([ "$attempt" = 1 ] || echo "-r$attempt").log
    echo "eta_min=1 doing=$TAG probe" > "$GPU_RUN_STATUS"
    bash "$ARM" dev --rows 1 --max-new 8 --num-draft 5 --timeout 600 \
        --out "$OUTDIR/$TAG-probe.jsonl" ${EXTRA[@]+"${EXTRA[@]}"} > "$log" 2>&1
    dp=$(grep -oE '"decode_ms_per_tok": *[0-9.]+' "$log" | head -1 | grep -oE '[0-9.]+$')
    say "PROBE attempt=$attempt decode_ms_per_tok=${dp:-none} ttft_ms=$(grep -oE '"ttft_ms": *[0-9.]+' "$log" | head -1 | grep -oE '[0-9.]+$')"
    if [ -z "${dp:-}" ]; then
      say "PROBE_FAIL attempt=$attempt no row -- see $log"
      if [ "$attempt" = 1 ] && grep -qE "No CUDA GPUs are available|CUDA driver version is insufficient|NVIDIA driver" "$log"; then
        say "PROBE_RETRY attempt=1 reason=no-gpu-in-container (transient)"; sleep 25; continue
      fi
      return 1
    fi
    awk -v d="$dp" 'BEGIN{exit !(d > 300)}' && { say "PROBE_DEGRADED ${dp}ms/token > 300: box is paging, not measuring"; return 1; }
    return 0
  done
  return 1
}
probe || { say "SWEEP_ABORT tag=$TAG reason=probe"; exit 3; }
run() {  # run <arm-tag> <k> <prompts-file|-> <rows>
  local tag=$1 k=$2 pf=$3 rows=$4
  [ "$pf" = - ] || tag=$tag-32k
  local log=$OUTDIR/$TAG-$tag.log
  REMAIN=$((REMAIN - 1))
  local eta=$(( (REMAIN * 50 + 59) / 60 ))
  echo "eta_min=$eta doing=$TAG $tag K=$k" > "$GPU_RUN_STATUS"
  say "ARM_START tag=$tag K=$k rows=$rows prompts=$([ "$pf" = - ] && echo held || echo 32k)"
  say "GX_MEM $tag $(gxmem)"
  if [ "$pf" = - ]; then
    bash "$ARM" dev --rows "$rows" --max-new 256 --num-draft "$k" --timeout 900 \
        --out "$J" ${EXTRA[@]+"${EXTRA[@]}"} > "$log" 2>&1
  else
    bash "$ARM" dev --rows 1 --max-new 256 --num-draft "$k" --timeout 1800 \
        --prompts "$pf" --out "$J" ${EXTRA[@]+"${EXTRA[@]}"} > "$log" 2>&1
  fi
  local rc=$?
  say "ARM_END tag=$tag rc=$rc log=$log"
  grep -E "^SUMMARY" "$log" | sed "s/^/  [$tag] /" | tee -a "$S" >/dev/null
  if [ $rc -ne 0 ]; then
    say "ARM_FAIL tag=$tag rc=$rc tail=$(grep -viE '^(warn|info)' "$log" | tail -n 3 | tr '\n' ' | ')"
  fi
}
FIRST=$(echo "$KS" | cut -d, -f1)
for k in $(echo "$KS" | tr ',' ' '); do
  run k$k "$k" - 3
  run k$k "$k" "$PROMPTS_32K" 1
done
# The first arm again, last: the same arm measured twice brackets any drift inside the ticket (the GX
# box's pack pages and other agents' work both move a single arm by percent; a K decision needs more).
run k${FIRST}r "$FIRST" - 3
say "SWEEP_DONE tag=$TAG"
