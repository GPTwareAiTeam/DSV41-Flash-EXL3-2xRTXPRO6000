#!/bin/bash
# gx_gate -- is the dev server on GX2 in a state where a decode measurement is quotable?
#
# SWAP GATE = OPTION B (owner decision, 2026-11-03). The worker's VmSwap *level* is never a refusal:
# the level, the box swap (SwapTotal-SwapFree), avail_mib, the worker pid, its VmRSS and the pack's own
# `swapped_mib=` from the load line are printed beside every reading as a WARNING, and the verdict is a
# *growth* over the measured window:
#
#   decode window   worker VmSwap growth > 128 MiB between --arm-start and --arm-end  -> REFUSE (rc=1)
#   prefill window  the same growth, flagged --prefill                                -> WARNING only
#   absolute        worker VmSwap > 2048 MiB                                          -> REFUSE (rc=1)
#                   (a clear "another tenant's pack" level; ~2 GiB is not one pack's own pages)
#   box_busy        --preload and avail_mib < PACK_MIB + PACK_MARGIN_MIB              -> REFUSE (rc=1)
#   box tight       avail_mib < 256 after the wait                                    -> REFUSE (rc=1)
#   no verdict      post-load avail_mib > ~2000 MiB (cannot be post-residency)        -> judge=no (rc=2)
#                   never a refusal: the run is fine, the *reading* is not comparable
#
# Why growth and not level: a load on this box legitimately lands some of the 116 GiB cold pack in swap
# (measured 344 MiB worker VmSwap with 1.1 GiB avail right after a fresh load, 22:04:29Z), and a pack
# page that was swapped once at load and never touched again costs nothing at decode time. What makes a
# decode number meaningless is the worker *re-faulting* pages while it generates, which is exactly
# growth inside the window. Swap optimization itself is deferred by the owner; do not spend time on it.
#
# Modes (one server round trip each; every reading prints the canonical line):
#   gate_gx.sh                     pre-flight verdict for a serving worker (the line other gate readers
#                                  already parse; refuses only on the rules above)
#   gate_gx.sh --preload           is the box free enough to load a pack at all (box_busy/pack+margin)
#   gate_gx.sh --arm-start TAG     baseline sample, written to $GX_STATE_DIR/TAG.pre; always rc=0
#                                  (rc=1 only when the box is unreachable: no baseline, nothing to judge)
#   gate_gx.sh --arm-end TAG [--prefill]
#                                  post sample; prints growth vs TAG.pre and the verdict
#
# Canonical line (agreed with GxPort2; unchanged key set, plus worker_vmrss_mib):
#   GX_MEM avail_mib=… box_swap_mib=… swap_total_mib=… worker_pid=… worker_vmswap_mib=…
#          slab_swap_base_mib=… swap_growth_mib=… slab_warn=… box_warn=… warn_last=…
#          worker_age_min=… ct_age_min=… ws_rows=… ws_env=… tune=… slab_buckets=… rows_max=…
#          ws_mib=… worker_vmrss_mib=… note=…
# `swap_growth_mib` here is box swap minus the load line's `swapped_mib=`; it is recorded for context
# and no longer judges anything. The window growth is on the GX_ARM lines.
#
# Sources, all read on GX2, all with stdin detached (`</dev/null` -- a docker exec inherits the stdin of
# a `ssh host bash -s` payload and would eat the rest of it):
#   avail/box/total swap   /proc/meminfo
#   worker_pid, VmSwap, VmRSS  the anchored worker (`^python3 -u …/worker_seam_server[.]py`, head -1 --
#                          the wrapper is excluded), host /proc/<pid>/status. This is the HOST pid; the
#                          container pid differs and changes on reload, the host pid does not.
#   slab_swap_base_mib     `swapped_mib=` of the last `SEAM_SERVER_SLAB ` line in the server's
#                          /tmp/seam_server.log INSIDE the container (the worker's stdout is redirected
#                          into that file; the host /tmp and `docker logs` are both empty). `?` means the
#                          file is missing, never "no baseline".
#   slab_warn / box_warn   whole-file counts (`grep -ac`): both are load-time lines, and this server writes
#                          one SEAM_COMPUTE line per request, so a tail-scoped read finds none on any
#                          server older than 400 requests -- exactly the case that matters.
#   warn_last              the same warns from the 400-line tail, for context only; never a verdict.
SLACK_ABS_MIB=${SLACK_ABS_MIB:-2048}
SLACK_GROWTH_MIB=${SLACK_GROWTH_MIB:-128}
SLACK_MIN_MIB=${SLACK_MIN_MIB:-256}
LEVEL_WARN_MIB=${LEVEL_WARN_MIB:-256}
POST_MAX_MIB=${POST_MAX_MIB:-2000}
PACK_MIB=${PACK_MIB:-116000}
PACK_MARGIN_MIB=${PACK_MARGIN_MIB:-2048}
CT=${CT:-seam-gx2-dev}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GX=${GX:-$(awk '!/^#/ && NF && $1=="dev"{print $8; exit}' \
  /home/treadall/recipe-dspark-tpr/qmesh/release/pairs.conf)}
GX_STATE_DIR=${GX_STATE_DIR:-$HERE/logs/gx_state}

gx_read() {
  local st
  if [ -n "${GX_ST_OVERRIDE:-}" ]; then st=$GX_ST_OVERRIDE; else
  st=$(ssh -o BatchMode=yes -o ConnectTimeout=6 "$GX" "CT=$CT bash -s" \
        < "$HERE/gate_gx_remote.sh") || { UNREACH=1; return 1; }; fi
  UNREACH=
  local mem
  mem=$(printf '%s\n' "$st" | head -1)
  # Keys are unique in this payload, so match them anywhere on the line; a missing key is 0, never a
  # syntax error (the arithmetic below must not die on an old probe script).
  num() { local v; v=$(printf '%s' "$st" | grep -oE "$1=[0-9]+" | cut -d= -f2 | head -1); echo "${v:-0}"; }
  avail=$(( $(num MemAvailable) / 1024 ))
  tot=$(( $(num SwapTotal) / 1024 ))
  free=$(( $(num SwapFree) / 1024 ))
  box=$(( tot - free ))
  wp=$(printf '%s' "$st" | grep -oE 'worker_pid=[0-9]*' | cut -d= -f2 | head -1)
  wpv=$(( $(num worker_vmswap_kb) / 1024 ))
  wrss=$(( $(num worker_vmrss_kb) / 1024 ))
  was=$(printf '%s' "$st" | grep -oE '^worker_age_s=[0-9]+' | cut -d= -f2)
  cta=$(printf '%s' "$st" | grep -oE '^ct_age_s=[0-9]+' | cut -d= -f2)
  base=$(printf '%s' "$st" | grep 'SEAM_SERVER_SLAB ' | tail -1 | grep -oE 'swapped_mib=[0-9]+' | cut -d= -f2)
  if [ -n "${base:-}" ]; then growth=$(( box - base )); else growth=; base=; fi
  slab_warn=$(printf '%s' "$st" | grep -oE '^slab_warn=[0-9]*' | cut -d= -f2)
  box_warn=$(printf '%s' "$st" | grep -oE '^box_warn=[0-9]*' | cut -d= -f2)
  warn_last=$(printf '%s' "$st" | grep -oE '^warn_last=.*' | cut -d= -f2-)
  wsr=$(printf '%s' "$st" | grep 'SEAM_SERVER_SLAB ' | tail -1 | grep -oE 'ws_rows=[0-9]+' | cut -d= -f2)
  # The grant is recorded, not judged: a narrowed bucket list re-chunks prefill and says nothing about decode,
  # but a reader comparing two runs needs to see that the pieces were sized differently.
  local slabline
  slabline=$(printf '%s' "$st" | grep 'SEAM_SERVER_SLAB ' | tail -1)
  grant=$(printf '%s' "$slabline" | grep -oE 'buckets=[0-9,]+' | cut -d= -f2)
  rmax=$(printf '%s' "$slabline" | grep -oE 'rows_max=[0-9]+' | cut -d= -f2)
  wsmib=$(printf '%s' "$slabline" | grep -oE 'ws_mib=[0-9.]+' | cut -d= -f2)
  # With no server the SLAB line is the previous load's, so quoting its width as a serving shape would be the
  # exact mislabel the ws_env check exists to prevent. Mark it, do not print it.
  [ -z "${wp:-}" ] && wsr="stale(slab_line)"
  # ws_rows=0 is the auto path: the workspace is sized by the request, not by a piece (1880 MiB at an
  # 8192-row request vs 235 at 1024 and 58.8 at 256), so it is never a decode-shaped width -- which is why
  # the pair pins a width at all. Printing the 0 would invite exactly that misreading.
  [ "${wsr:-}" = "0" ] && wsr="auto(request-sized)"
  argv=$(printf '%s' "$st" | grep -oE '^worker_argv=.*' | cut -d= -f2-)
  unanch=$(printf '%s' "$st" | grep -oE '^worker_unanchored=[0-9]+' | cut -d= -f2)
  tcache=$(printf '%s' "$st" | grep -oE '^tune=.*' | cut -d= -f2-)
  wsenv=$(printf '%s' "$st" | grep -oE '^worker_ws_env=.*' | cut -d= -f2-)
  klay=$(printf '%s' "$st" | grep -oE '^kernel_layers=.*' | cut -d= -f2-)
  knoext=$(printf '%s' "$st" | grep -oE '^kernel_noext=[0-9]+' | cut -d= -f2)
  return 0
}

# The five things the owner wants beside every reading, on one line: current worker VmSwap level, box
# swap, avail, pid, and the pack's own swapped_mib from the load line.
gx_emit() {
  echo "GX_MEM avail_mib=$avail box_swap_mib=$box swap_total_mib=$tot worker_pid=${wp:-?}" \
       "worker_vmswap_mib=$wpv slab_swap_base_mib=${base:-?} swap_growth_mib=${growth:-?}" \
       "slab_warn=${slab_warn:-?} box_warn=${box_warn:-?}" \
       "worker_age_min=$(( ${was:-0} / 60 )) ct_age_min=$(( ${cta:-0} / 60 )) warn_last=${warn_last:-none}" \
       "ws_rows=${wsr:-?} ws_env=${wsenv:-?} tune=${tcache:-?}" \
       "slab_buckets=${grant:-?} rows_max=${rmax:-?} ws_mib=${wsmib:-?}" \
       "worker_vmrss_mib=${wrss:-?} note=worker_age_is_the_server_container_age_is_not"
  [ -n "${argv:-}" ] && echo "GX_SHAPE worker_argv=$argv kernel_layers=${klay:-?} kernel_noext=${knoext:-?}"
  # Levels are context, never verdicts (Option B). Printed whenever the worker has any swap at all, so a
  # reader never has to wonder whether it was zero or merely below a threshold.
  if [ "${wpv:-0}" -gt 0 ]; then
    echo "GX_WARN kind=level worker_vmswap_mib=$wpv worker_vmrss_mib=${wrss:-?} box_swap_mib=$box" \
         "avail_mib=$avail worker_pid=${wp:-?} slab_swap_base_mib=${base:-?}" \
         "note=level_only_not_a_refusal_growth_decides"
  fi
  [ "${slab_warn:-0}" != "0" ] && [ -n "${slab_warn:-}" ] &&
    echo "GX_WARN kind=slab_warn count=$slab_warn note=the_load_line_says_the_pack_did_not_fit"
  if [ "${box_warn:-0}" != "0" ] && [ -n "${box_warn:-}" ]; then
    echo "GX_NOTE box_warn=$box_warn: box-wide SWAP_WARN/SLACK_WARN lines (desktop pages included), not a refusal"
  fi
  return 0
}

# One key=value per line, for a later --arm-end to diff against.
gx_dump() {
  printf '%s\n' \
    "at_s=$(date +%s)" "avail_mib=$avail" "box_swap_mib=$box" "swap_total_mib=$tot" \
    "worker_pid=${wp:-}" "worker_vmswap_mib=$wpv" "worker_vmrss_mib=${wrss:-0}" \
    "slab_swap_base_mib=${base:-}" "swap_growth_mib=${growth:-}" \
    "slab_warn=${slab_warn:-0}" "box_warn=${box_warn:-0}" \
    "worker_age_min=$(( ${was:-0} / 60 ))" "ws_rows=${wsr:-?}" \
    "ws_env=${wsenv:-?}" "tune=${tcache:-?}" "kernel_layers=${klay:-?}" "kernel_noext=${knoext:-0}"
}
fld() { sed -n "s/^$2=//p" "$1" 2>/dev/null | tail -1; }
tag_ok() { case "$1" in ''|*/*|*..*) echo "GX_REFUSE bad_tag=$1"; return 1 ;; esac; }

gx_flight() {
  [ -n "${UNREACH:-}" ] && { echo "GX_MEM avail_mib=? unreachable=1"; return 1; }
  gx_emit
  # The env said cudaexl3; only the log says whether the layers ran it. A fallback load measures the mixed-K
  # module "at a fraction of the measured rate" (pair_lib), so its numbers are not the shipping path.
  if [ "${knoext:-0}" != "0" ]; then
    echo "GX_REFUSE kernel=fallback(mixed-k): SEAM_MOE_NOEXT in the current load, so this box is not serving" \
         "the cudaexl3 path the release pins; kernel_layers=${klay:-?}"; return 1
  fi
  if [ -z "${wp:-}" ]; then
    echo "GX_REFUSE server=down: no worker process, nothing to measure unanchored_matches=${unanch:-?}"
    return 1
  fi
  if [ "$wpv" -gt "$SLACK_ABS_MIB" ]; then
    echo "GX_REFUSE worker_vmswap_mib=$wpv > $SLACK_ABS_MIB: that is another tenant's pack level, not this" \
         "worker's own pages; box_swap_mib=$box avail_mib=$avail worker_pid=$wp"; return 1
  fi
  if [ "$avail" -lt "$SLACK_MIN_MIB" ]; then
    local waited=0 a2
    while [ "$waited" -lt "${GX_WAIT_S:-60}" ]; do
      sleep 10; waited=$(( waited + 10 ))
      a2=$(ssh -o BatchMode=yes -o ConnectTimeout=6 "$GX" \
             "awk '/^MemAvailable/{printf \"%d\", \$2 / 1024}' /proc/meminfo" </dev/null 2>/dev/null)
      [ -n "${a2:-}" ] && [ "$a2" -ge "$SLACK_MIN_MIB" ] && break
    done
    echo "BOX_READY waited_s=$waited avail_mib=${a2:-?} (a stopped server releases its pack asynchronously)"
    if [ -z "${a2:-}" ] || [ "$a2" -lt "$SLACK_MIN_MIB" ]; then
      echo "GX_REFUSE avail_mib=$avail < $SLACK_MIN_MIB after ${waited}s"; return 1
    fi
  fi
  # A serving worker whose box still has more than ~2 GiB available is not post-residency: the reader
  # must not trust the absolute numbers. The run is never refused for this, only the judgement.
  if [ "$avail" -gt "$POST_MAX_MIB" ]; then
    echo "GX_NOJUDGE avail_mib=$avail > $POST_MAX_MIB with worker_pid=$wp: cannot be post-residency," \
         "so readings here are not comparable to a loaded box (run may proceed)"
    return 2
  fi
  return 0
}

gx_preload() {
  [ -n "${UNREACH:-}" ] && { echo "GX_MEM avail_mib=? unreachable=1"; return 1; }
  gx_emit
  local want=$(( PACK_MIB + PACK_MARGIN_MIB ))
  if [ "$avail" -lt "$want" ]; then
    echo "BOX_BUSY box_busy=1 avail_mib=$avail want_mib=$want pack_mib=$PACK_MIB margin_mib=$PACK_MARGIN_MIB" \
         "(another tenant holds GX RAM; a load here is served from disk)"; return 1
  fi
  echo "GX_PRELOAD_OK avail_mib=$avail want_mib=$want"
  return 0
}

gx_arm_start() { # tag
  local tag=$1
  tag_ok "$tag" || return 1
  mkdir -p "$GX_STATE_DIR"
  if [ -n "${UNREACH:-}" ]; then
    echo "GX_ARM phase=pre tag=$tag unreachable=1 note=no_baseline"; return 1
  fi
  gx_dump > "$GX_STATE_DIR/$tag.pre"
  # The one thing a window cannot survive is the worker not being there: nothing to attribute a growth to.
  if [ -z "${wp:-}" ]; then
    echo "GX_ARM phase=pre tag=$tag worker_pid=? avail_mib=$avail box_swap_mib=$box"
    echo "GX_REFUSE phase=pre tag=$tag server=down: no worker to baseline, so the window cannot be judged"
    return 1
  fi
  echo "GX_ARM phase=pre tag=$tag avail_mib=$avail box_swap_mib=$box worker_pid=${wp:-?}" \
       "worker_vmswap_mib=$wpv worker_vmrss_mib=${wrss:-?} slab_swap_base_mib=${base:-?}" \
       "ws_rows=${wsr:-?} note=baseline_for_the_window"
  [ "${wpv:-0}" -gt 0 ] && echo "GX_WARN kind=level phase=pre tag=$tag worker_vmswap_mib=$wpv" \
       "box_swap_mib=$box avail_mib=$avail worker_pid=${wp:-?} slab_swap_base_mib=${base:-?} note=level_only"
  return 0
}

gx_arm_end() { # tag [--prefill]
  local tag=$1 pre=$GX_STATE_DIR/$1.pre decode=1
  [ "${2:-}" = "--prefill" ] && decode=0
  if [ -n "${UNREACH:-}" ]; then
    echo "GX_ARM phase=post tag=$tag unreachable=1 judge=no"; return 2
  fi
  gx_dump > "$GX_STATE_DIR/$tag.post"
  [[ "$tag" =~ (^|_)32k($|_) ]] && decode=0   # a 32k arm's window is a prefill window by construction
  echo "GX_ARM phase=post tag=$tag avail_mib=$avail box_swap_mib=$box worker_pid=${wp:-?}" \
       "worker_vmswap_mib=$wpv worker_vmrss_mib=${wrss:-?} slab_swap_base_mib=${base:-?} ws_rows=${wsr:-?}"
  if [ -z "${wp:-}" ]; then
    echo "GX_REFUSE phase=post tag=$tag worker=gone: the worker left inside the window" \
         "(avail_mib=$avail box_swap_mib=$box), so the arm measured a load, not a decode"
    return 1
  fi
  if [ ! -r "$pre" ]; then
    echo "GX_ARM_GROWTH tag=$tag window_s=? worker_vmswap_growth_mib=? box_swap_growth_mib=?" \
         "judge=no reason=no_baseline note=start_the_window_with_--arm-start"; return 2
  fi
  local pv pb pv_rss t0 t1 dg db
  pv=$(fld "$pre" worker_vmswap_mib); pb=$(fld "$pre" box_swap_mib)
  pv_rss=$(fld "$pre" worker_vmrss_mib); t0=$(fld "$pre" at_s); t1=$(date +%s)
  dg=$(( wpv - ${pv:-0} )); db=$(( box - ${pb:-0} )); win=$(( t1 - ${t0:-$t1} ))
  echo "GX_ARM_GROWTH tag=$tag window_s=$win worker_vmswap_mib_pre=${pv:-?} worker_vmswap_mib=$wpv" \
       "worker_vmswap_growth_mib=$dg box_swap_growth_mib=$db worker_vmrss_mib_pre=${pv_rss:-?}" \
       "worker_vmrss_mib=${wrss:-?} avail_mib=$avail worker_pid=${wp:-?} slab_swap_base_mib=${base:-?}"
  if [ "$dg" -gt "$SLACK_GROWTH_MIB" ] && [ "$decode" = 1 ]; then
    echo "GX_REFUSE phase=post tag=$tag worker_vmswap_growth_mib=$dg > $SLACK_GROWTH_MIB inside the decode" \
         "window: the worker re-faulted its pack while generating, so this arm measured the pager" \
         "(box_swap_mib=$box avail_mib=$avail worker_pid=$wp)"
    return 1
  fi
  if [ "$dg" -gt "$SLACK_GROWTH_MIB" ]; then
    echo "GX_WARN kind=window_growth phase=post tag=$tag window=prefill worker_vmswap_growth_mib=$dg" \
         "note=expected_for_a_32k_prefill_and_not_a_refusal (box_swap_growth_mib=$db)"
  fi
  if [ "$avail" -gt "$POST_MAX_MIB" ]; then
    echo "GX_NOJUDGE phase=post tag=$tag avail_mib=$avail > $POST_MAX_MIB: cannot be post-residency, reading not comparable"
    return 2
  fi
  echo "GX_ARM_OK tag=$tag worker_vmswap_growth_mib=$dg box_swap_growth_mib=$db window_s=$win"
  return 0
}

main() {
  local mode=${1:-flight} rc
  case "$mode" in
    flight)   gx_read; gx_flight; rc=$? ;;
    --preload|preload) gx_read; gx_preload; rc=$? ;;
    --arm-start) gx_read; gx_arm_start "${2:-?}"; rc=$? ;;
    --arm-end) gx_read; gx_arm_end "${2:-?}" "${3:-}"; rc=$? ;;
    *) echo "GX_REFUSE unknown_mode=$mode (flight|--preload|--arm-start TAG|--arm-end TAG [--prefill])"; rc=1 ;;
  esac
  echo "gate_rc=$rc"
  return $rc
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then main "$@"; fi
