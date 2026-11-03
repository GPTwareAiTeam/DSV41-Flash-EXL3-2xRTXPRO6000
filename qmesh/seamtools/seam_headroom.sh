#!/usr/bin/env bash
# The headroom census: one --fresh load per workspace shape (ws_rows 1024, the shipping geometry, and
# 256, a decode-sized piece), each with a decode arm and the 32k prefill, and every line that says how
# much memory the box and the worker had left -- plus a 10 s trajectory across all three phases.
#
# Why a separate ticket: the ws 2048 arm measured 1123.9 tok/s and then printed post-32k
# avail_mib=129 with swap_warn=1, which is fast and not shippable. Before any ws_rows above 1024
# is tested again, the owner wants the actual per-layer balance on a fresh load of the release we
# ship -- the pack is 113.65 GiB of a 121.6 GiB pool, so the whole question is what the load and
# the prefill each take out of the ~8 GiB that is not the pack.
#
# The lines this leaves in the evidence dir, all from the server's own log (arm3 tees it, prefixed
# with "DIRECT load1   ", so none of the greps are anchored):
#   SEAM_PACK_LOAD    need_mib, avail_mib before the pack, margin, fadvise, swapped
#   layer <N>:        per layer: GiB allocated, avail_mib, cached_mib, rss_peak, host rss/swap/trim
#   SEAM_HOST_MEM     phase=load_done rss/swap/anon/trim + avail_mib + box swap (the host bill)
#   SEAM_PACK_KERNEL  how many layers went to cudaexl3 and how many stayed mixed-K, avail/min
#   SEAM_SERVER_SLAB  slab_mib, ws_rows, ws_mib, rows_max, avail_mib, swapped_mib, fits
#   SEAM_SERVER_SWAP_WARN  present only when the load paged; its swapped_mib is the pack sliding
#   SLACK load1       after the 32k prefill: avail_mib, swap_free_mib, ws_rows, prefill, swap_warn
#
#   queue/run_dev.sh SeamRobust 40 bash <this script>     (two loads plus two prefills)
set -uo pipefail
. /home/treadall/recipe-artifacts/gpu0-gx1/scripts/pair_lib.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cx_pair_server.sh"
pair_load dev                       # GX=10.0.4.33, GXCT=seam-gx2-dev, PORT=47311, BUCKETS=1,6,64,8192
# The pair's own container (pairs.conf via `pair_load dev`): /cold is the pack, /src/exllamav3 is the
# release it serves from. seam-gx2-4460 no longer exists on either GX10 after the 11-02 reboot.
CT=${CX_CT:-$GXCT}
P7CT=${CX_P7CT:-seam-p7-dev}        # pairs.conf's dev client: /models, /artifacts, same release tree
WT=${CX_WT:-/home/treadall/recipe-seam-robust}   # after the release lands, point this at its tip
# A pair container renumbers its one card to 0 (`--gpus device=1` at start-up), so the client index is
# the container's, not the host's.
GPU=${CX_CUDA_VISIBLE:-0}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARM_SH="$HERE/seam_direct_arm3.sh"
STAMP=$(date +%m%d-%H%M%S)
AB=${CX_AB_BASE:-/home/treadall/recipe-artifacts/seamrobust/evidence/ab}/$STAMP
LOG=$AB/headroom.log
mkdir -p "$AB"; : > "$LOG"
# Two load shapes, because the question is whether the workspace geometry is what makes the worker
# swap: 1024 is today's shipping shape (measured fast, measured tight), 256 is a decode-sized piece.
# Each shape gets its own --fresh load, so the comparison is load-to-load and not arm-to-arm.
ROWS=${CX_HR_ROWS:-1024 256}
WS=                             # set per iteration; kept for the log lines that name the shape

say() { echo "$(date +%T) HR $*" | tee -a "$LOG"; }
status() { [ -n "${GPU_RUN_STATUS:-}" ] && printf 'eta_min=%s doing=%s\n' "$1" "$2" > "$GPU_RUN_STATUS"; return 0; }
gx_mem() {  # -> "avail_mib swapfree_mib"
  timeout 20 ssh -o BatchMode=yes "$GX" \
    'awk "/^MemAvailable:/{a=\$2} /^SwapFree:/{s=\$2} END{printf \"%d %d\", a/1024, s/1024}" /proc/meminfo' \
    2>/dev/null | tr -d '\r'
}
cleanup() {  # the kill-safe handback; see seam_direct_det5.sh for why each step is here
  local rc=$? i av sw st
  trap - EXIT INT TERM
  timeout 12 ssh -o BatchMode=yes "$GX" \
    "docker exec $CT sh -c 'pkill -f \"^python3 -u qmesh/worker_seam_server.py\" || exit 0; sleep 4; pkill -KILL -f \"^python3 -u qmesh/worker_seam_server.py\" || true'" \
    >/dev/null 2>&1
  st=$(pair_gx_gate_state)
  pair_gx_gate_down
  for i in 1 2 3; do
    read -r av sw < <(gx_mem)
    [ -n "${av:-}" ] && [ "$av" -ge "${CX_MIN_AVAIL:-118400}" ] && break   # the arm's own floor
    sleep 3
  done
  say "CLEANUP rc=$rc gx_avail_mib=${av:-?} swap_free_mib=${sw:-?} gate_was=${st:-none}"
  exit "$rc"
}
trap cleanup EXIT INT TERM

[ $# = 0 ] || { echo "usage: $0   (no arguments: every knob is an env var, see the header)" >&2; exit 2; }
status 20 "gx2 gate: headroom census, one fresh load per shape (ws_rows $ROWS)"
# Two loads plus two prefills need more than one: 45 min, and 10 min of waiting for the box to drain.
pair_gx_gate_take SeamRobust-hr "${CX_HR_GATE_MIN:-45}" 10 || { say "HR FAIL gate_unresolved"; exit 3; }
say "gate file=$(pair_gx_gate_file) owner=SeamRobust-hr"
say "script=$0 sha=$(sha256sum "$0" | cut -c1-16) arm_sha=$(sha256sum "$ARM_SH" | cut -c1-16) wt=$WT ws_rows=[$ROWS]"
say "box before stop: gx=[$(gx_mem)]"

cx_pair_server_stop | sed "s/^/HR pair   /" | tee -a "$LOG"
say "box after stop: gx=[$(gx_mem)]  (this is the number the load's SEAM_PACK_LOAD should match)"
# Attribute the census to a commit, not to "the tree on disk". The container has no .git (it holds a
# copy of a release), so the only way a number from this ticket can be matched to a release is to print
# the sha here, refuse a dirty worktree, and let cx_verify_tree.sh prove afterwards that every file in
# the container equals this worktree -- not seven of them, and without rewriting the release. A census taken from a half-applied worktree is exactly the kind of
# evidence that reads as measured and is not.
SRC=$WT
sha=$(git -C "$SRC" rev-parse HEAD 2>/dev/null)
dirty=$(git -C "$SRC" status --porcelain 2>/dev/null | grep -cv '^?? ')
say "STAGED src=$SRC branch=$(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null) sha=${sha:-none} dirty=$dirty want=${CX_STAGE_SHA:-any}"
[ -z "${CX_STAGE_SHA:-}" ] || [ "${CX_STAGE_SHA:0:8}" = "${sha:0:8}" ] ||
  { say "HR FAIL $SRC is at ${sha:-none}, not at the requested ${CX_STAGE_SHA}; the canary shipped another commit"; exit 15; }
[ "$dirty" = 0 ] || { say "HR FAIL $SRC has $dirty uncommitted files; a census cannot be attributed to ${sha:-none}"; exit 16; }
# Verification, not a copy: the container's /src/exllamav3 is a symlink into the pair's release, and
# writing another commit into a release directory is what makes pair_check call the pair drifted.
status 17 "verifying the container runs $SRC at $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null)"
SRC="$SRC" GX="$GX" CX_CT="$CT" bash "$HERE/cx_verify_tree.sh" 2>&1 | tee -a "$AB/verify.log" >> "$LOG" ||
  { say "HR FAIL the container is not running $sha; nothing it prints would be evidence"; exit 13; }
grep -ah '^VERIFY lib \|^VERIFY target ' "$AB/verify.log" | sed 's/^/HR /' | tee -a "$LOG"

# Two things a reader of this table would otherwise have to know from elsewhere.
#  - d330dcf5 has no prefix cache (a fresh Cache per client process, e2e_prompts.py:271), so the
#    second shape's load is not helped by the first one's prefixes and a faster load2 is not a
#    warm-cache artifact. Equally, nothing here benefits from one.
#  - the two shapes do not answer a prefill in the same number of pieces (see HR GRANT): ws_rows is
#    a ceiling on the *piece*, the client chunks a prefill at the largest granted bucket, so 256
#    rows turns one 8192-row chunk into 32 pieces and 1024 into 8. Prefill seconds therefore compare
#    weight re-reads, not throughput; the comparison this census is for is VmRSS/VmSwap/Pss and
#    SUnreclaim at each phase.
say "HR NOTE prefix_cache=none(d330dcf5) piece_split=min(ws_rows,rows_max) prefill_compare=swap_only"
rcs=""
fails=0
for WS in $ROWS; do
  E=$AB/H$WS
  mkdir -p "$E"
  status 12 "--fresh load at ws_rows $WS: load, decode arm, 32k prefill, 10 s sampling"
  # The shape here comes from SEAM_ENV, so it is unpinned by construction -- but pair_serve.sh,
  # pair_smoke.sh, pair_update_run.sh and the decode/roundprof tickets all take their shape from the
  # pair line's WS_ROWS (pairs.conf, default 1024) through pair_server_env. A row here can therefore
  # never be compared against a server that came up from one of those paths unless the pair line said
  # 256; HR GRANT is the line that settles which shape a load actually had.
  # CX_WT is passed, not inherited: arm3 verifies the container against ${CX_WT:-<robust worktree>},
  # so a driver that verified one tree and an arm that verified another would compare the container
  # twice against different bytes and the arm's exit 13 would blame the wrong commit.
  CX_WT="$WT" CX_GATE=own CX_ARM_TAG="H$WS" CX_EVIDENCE="$E" CX_LOADS=1 CX_IDS_RUNS=1 CX_WS_ROWS="$WS" \
  CX_DETERMINISTIC=0 CX_PREFILL=1 CX_WANT_DIGEST=any \
  CX_SERVER_PLACEMENT=/cold/placement.exl3.json \
  CX_PLACEMENT=/models/dsv41/hot4cold2/packs/placement.hot4460-cold2800x4-8100x2.json \
  CX_RANKED=0 CX_ORACLE=0 CX_REFUSAL=0 \
    bash "$ARM_SH" 2>&1 | tee -a "$AB/armH$WS.log" >> "$LOG"
  rcH=${PIPESTATUS[0]}
  rcs="$rcs $WS:$rcH"
  [ "$rcH" = 0 ] || fails=$((fails + 1))
  say "ARM H$WS rc=$rcH"

  status 3 "verdict ws_rows $WS"
  SRV=$E/server_load1.log            # server_log()'s archive of /tmp/seam_server.log; column 0
  PL=$E/pf32k_load1.log
  say "SOURCES ws_rows=$WS server=$(basename "$SRV") prefill=$(basename "$PL")"
  # Unanchored anyway: the copy in the ticket log is prefixed with "DIRECT load1   ", this one is not.
  grep -ah 'SEAM_PACK_LOAD ' "$SRV" | sed "s/^/HR PACKLOAD ws=$WS /" | tee -a "$LOG"
  grep -acE 'layer [0-9]+:' "$SRV" | sed "s/^/HR LAYERLINES ws=$WS /" | tee -a "$LOG"
  grep -ah 'SEAM_HOST_MEM ' "$SRV" | sed "s/^/HR HOSTMEM ws=$WS /" | tee -a "$LOG"
  grep -ah 'SEAM_HOST_MAP ' "$SRV" | sed "s/^/HR HOSTMAP ws=$WS /" | tee -a "$LOG"
  grep -ah 'SEAM_PACK_KERNEL ' "$SRV" | sed "s/^/HR KERNEL ws=$WS /" | tee -a "$LOG"
  # The two lines the workspace question is answered from: the pack's own slab/workspace bill, and the
  # allocator-internal split (free/total of the unified pool, allocated vs reserved) once per load.
  grep -ah 'SEAM_SERVER_SLAB ' "$SRV" | sed "s/^/HR SRVSLAB ws=$WS /" | tee -a "$LOG"
  # The grant, stated instead of inferred: with SEAM_WS_ROWS_AUTO=0 the server serves one request in
  # pieces of min(ws_rows, rows_max), so a 256-row server answers an 8192-row prefill chunk in 32
  # pieces where a 1024-row one answers it in 8. That is a different prefill shape as well as a
  # different workspace (the client's chunk_rows is the largest granted bucket, not ws_rows), which
  # is why the two arms' prefill seconds are not a throughput A/B. Printed so no reader has to hold
  # the bucket list in their head to notice it.
  read -r SRVWS RMAX <<< "$(grep -ah 'SEAM_SERVER_SLAB ' "$SRV" | tail -1 |
    sed -n 's/.* ws_rows=\([0-9]*\) .* rows_max=\([0-9]*\) .*/\1 \2/p')"
  PPC=$(( (${RMAX:-1} + ${SRVWS:-1} - 1) / ${SRVWS:-1} ))
  say "HR GRANT ws=$WS server_ws_rows=${SRVWS:-?} rows_max=${RMAX:-?} pieces_per_prefill_chunk=$PPC auto=0"
  grep -ah 'SEAM_SERVER_DEVMEM ' "$SRV" | sed "s/^/HR DEVMEM ws=$WS /" | tee -a "$LOG"
  grep -ah 'SLAB_SHIP \|SAMPLE_SHIP ' "$AB/armH$WS.log" | sed "s/.*DIRECT /HR /" | tee -a "$LOG"
  # Host samples around the load (clear box, 113 GiB resident) and after the prefill; SUnreclaim is
  # the slab nobody can hand back, which is the memory the pack load leaves unaccounted for.
  grep -ah 'DIRECT SLAB phase=' "$AB/armH$WS.log" | sed "s/.*SLAB /HR CENSUS ws=$WS /" | tee -a "$LOG"
  # The three phase points, each a synchronous sample: at load, after the decode arm, after 32k.
  grep -ah 'DIRECT PHASESAMP ' "$AB/armH$WS.log" | sed "s/.*DIRECT PHASESAMP /HR PHASESAMP ws=$WS /" | tee -a "$LOG"
  # The 10 s trajectory across load+decode+prefill: which consumer grows (Cached = page cache,
  # SUnreclaim = kernel slab, VmRSS/Pss = inside the worker, VmSwap = the worker's own pages).
  grep -ah 'SAMPLE_START \|SAMPLES load' "$AB/armH$WS.log" | sed "s/.*DIRECT /HR /;s/$/ ws=$WS/" | tee -a "$LOG"
  grep -ah 'DIRECT SAMP load1 ' "$AB/armH$WS.log" | sed "s/.*DIRECT SAMP load1 SAMP /HR SAMP ws=$WS /" | tee -a "$LOG"
  # From this load's own SEAM_SERVER_READY, never the tail: the container log is truncated by the
  # ensure path, but a restore that starts a server with >> would leave the previous load's warn at
  # the top of the file and a tail-based grep would quote it as this shape's.
  awk '/^SEAM_SERVER_READY/{n=NR} {a[NR]=$0} END{ if (n) for (i=n;i<=NR;i++) print a[i]; else for (i=1;i<=NR;i++) print a[i] }' "$SRV" |
    grep -ah 'SEAM_SERVER_SWAP_WARN ' | sed "s/^/HR SWAPWARN ws=$WS /" | tee -a "$LOG"
  # The piece count rides on the prefill line: the seconds there are 8192 tokens over PPC weight
  # passes, so nobody divides tokens by seconds and calls the result a shape comparison.
  grep -ah 'PREFILL_ARM ' "$PL" | sed "s/^/HR PREFILL ws=$WS pieces_per_chunk=$PPC /" | tee -a "$LOG"
done
# The comparison the owner reads first: worker swap and RSS at the three phases, per shape.
say "SHAPE TABLE (worker, from the phase samples)"
for WS in $ROWS; do
  grep -ah 'DIRECT PHASESAMP ' "$AB/armH$WS.log" 2>/dev/null |
    sed 's/.*DIRECT PHASESAMP //' |
    awk -v w="$WS" '
      { ph=$1; rss="-"; vsw="-"; av="-"
        for (i = 2; i <= NF; i++) if ($i ~ /^vm_rss_mib=/) { split($i, a, "="); rss = a[2] }
        for (i = 2; i <= NF; i++) if ($i ~ /^vm_swap_mib=/) { split($i, a, "="); vsw = a[2] }
        for (i = 2; i <= NF; i++) if ($i ~ /^avail_mib=/) { split($i, a, "="); av = a[2] }
        printf "HR SHAPE ws=%s phase=%s vm_rss_mib=%s vm_swap_mib=%s avail_mib=%s\n", w, ph, rss, vsw, av }' |
    tee -a "$LOG"
done
say "FINAL gx=[$(gx_mem)] rc=$rcs"
[ "$fails" = 0 ] || { say "HR FAIL arms failed:$rcs"; exit 24; }
say "HR PASS ws_rows=[$ROWS] census printed above; nothing above the shipping shape is tested until the owner reads it"
