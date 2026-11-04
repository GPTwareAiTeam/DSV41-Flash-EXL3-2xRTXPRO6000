#!/usr/bin/env bash
# promote_arm.sh -- one arm of the dev-pair release A/B, inside one queue ticket.
#
#   promote_arm.sh <sha> <tag> [restore_sha]
#
# Flip dev onto <sha> (flip-only: the release must already be prepared), prove which release the
# server and the client are actually on, then measure the two things a promotion needs in the same
# session as the other arm: a 256-row decode row (the shipped shape) and wiki2 quick8 ppl.
# `restore_sha` (optional) flips dev back to it at the end, so a losing candidate never stays on the
# box.
#
# Why flip-only pair_update_run.sh and not a hand-rolled symlink: the flip is where the release is
# *verified* (libs stamped for both arches, ext for both, container wiring, then a smoke row). A/B
# numbers from a release nobody flipped through that gate describe a tree, not a pair.
#
# Why the server stage exists at all: pair_arm.sh is a *client*. Against a dead or mismatched seam
# server it blocks instead of failing, so the arm has to own the box the way pair_smoke.sh does --
# take the GX claim, check both containers can see their GPU, prove the client import is a cache hit,
# load one worker from this release, and stop it before the claim is dropped.
set -uo pipefail
S=/home/treadall/recipe-artifacts/gpu0-gx1/scripts
. "$S/pair_lib.sh"

SHA_ARG=${1:?usage: promote_arm.sh <sha> <tag> [restore_sha]}
TAG=${2:?tag required}
RESTORE=${3:-}
pair_load dev
# Outputs must live where the containers can reach them: `--out` is handed to the client inside the
# P7 container, and pair_ct_path leaves an unmapped host path untouched, so a /tmp default writes a
# json nothing can open (and, worse, looks like a measurement that was taken).
OUT=${PROMOTE_OUT:-$PAIR_STATE_DIR/promote_ab}
mkdir -p "$OUT"
PY_CT=/artifacts/ppl-4460/panel_ppl_4460.py
Q_CT=/artifacts/2x-rtx-tp2/quality
ROWS=256
MAXNEW=256
GATE_LABEL="promote-ab-$PAIR"
SERVER_TIMEOUT=${PROMOTE_SERVER_TIMEOUT:-1200}

ENSURE=${SEAM_ENSURE:-}
if [ -z "$ENSURE" ]; then
  for c in "$PAIR_SCRIPTS/seam_server_ensure.sh" "$S/seam_server_ensure.sh"; do
    [ -x "$c" ] && { ENSURE=$c; break; }
  done
fi

step() { echo "$(date -u +%H:%M:%S) PROMOTE $*"; }
p7_clocks() { nvidia-smi --query-gpu=index,memory.used,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader | sed -n "$((GPU+1))p"; }
host_load() { awk '{printf "%.2f", $1}' /proc/loadavg; }
gx_state() {  # avail / box swap / worker VmSwap, one reading, one ssh round trip
  pair_gxsh "awk '/^MemAvailable:/{a=int(\$2/1024)} /^SwapTotal:/{t=int(\$2/1024)} /^SwapFree:/{f=int(\$2/1024)} END{printf \"avail_mib=%d box_swap_mib=%d\n\", a, t-f}' /proc/meminfo" 2>/dev/null | tail -1
}
workers() { pair_gxsh "docker exec $GXCT sh -c 'pgrep -x python3 -a | grep \"worker_seam_server[.]py\" | wc -l'" 2>/dev/null | tail -1; }

# The ensure launches the worker with `docker exec -d`; it writes no pidfile, so pair_server_stop -- which
# stops "the recorded worker" -- prints SERVER_STOP_SKIP and leaves 113 GiB resident. On 2026-11-04 at
# 04:03Z that leftover then made the next arm's flip refuse with SMOKE_FAIL memory box_busy avail_mib=840
# and cost the whole A/B. The gate is ours here, so any worker in our own GX container is ours to kill.
kill_worker() {
  local pids
  pids=$(pair_gxsh "docker exec $GXCT pgrep -x python3 -a" 2>/dev/null | tr -d '\r' \
         | awk '/worker_seam_server[.]py/ {print $1}' | tr '\n' ' ')
  [ -n "${pids// /}" ] || return 0
  pair_gxsh "docker exec $GXCT kill -9 $pids" >/dev/null 2>&1
  step "CLEANUP_KILL_WORKER tag=$TAG pids=$pids"
}

on_exit() {
  local rc=$?
  pair_server_stop >/dev/null 2>&1
  kill_worker >/dev/null 2>&1
  pair_gx_gate_down >/dev/null 2>&1
  step "CLEANUP tag=$TAG workers_left=$(workers) gate=$(pair_gx_gate_state 2>/dev/null | cut -c1-40)"
  return $rc
}
trap on_exit EXIT

SHA=$(git -C "$PAIR_REPO" rev-parse --verify "$SHA_ARG^{commit}" 2>/dev/null) \
  || { step "FAIL tag=$TAG no such commit: $SHA_ARG"; exit 2; }
step "ARM_START tag=$TAG sha=${SHA:0:12} host_load=$(host_load) p7=$(p7_clocks) gx=$(gx_state)"
# The stamp the archive fallback compares against: an archive written before this arm started belongs
# to someone else's load and must never be quoted as this one's fused coverage.
mkdir -p "$OUT"; touch "$OUT/arm_start_$TAG.stamp"

# ---- 0. prepare (out of band, idempotent) --------------------------------------------------------
if [ "${PROMOTE_PREPARE:-0}" = 1 ]; then
  step "PREPARE_START tag=$TAG sha=${SHA:0:12}"
  bash "$S/pair_prepare.sh" dev "$SHA" --side both >"$OUT/prepare_$TAG.log" 2>&1
  prc=$?
  grep -E 'PREPARE_|PREPARED|LIB_OK|EXT_OK|SEAM_OK' "$OUT/prepare_$TAG.log" | sed "s/^/PREP $TAG /" | cut -c1-200
  step "PREPARE_END tag=$TAG rc=$prc"
  [ "$prc" = 0 ] || { step "FAIL tag=$TAG prepare rc=$prc tail=$(tail -3 "$OUT/prepare_$TAG.log" | tr '\n' '|' | cut -c1-400)"; exit 7; }
fi
# ---- 1. flip (and prove it) ----------------------------------------------------------------------
step "FLIP_START tag=$TAG"
PAIR_FLIP_ONLY=1 "$S/pair_update_run.sh" dev "$SHA" >"$OUT/flip_$TAG.log" 2>&1
flip_rc=$?
grep -E 'UPDATE_(RELEASE|CONTAINERS|FLIP|SHA)|SMOKE_|FLIP_|UPDATE_SMOKE|PLACEMENT' "$OUT/flip_$TAG.log" \
  | sed "s/^/FLIP $TAG /" | cut -c1-200
step "FLIP_END tag=$TAG rc=$flip_rc host_load=$(host_load) gx=$(gx_state)"
[ "$flip_rc" = 0 ] || { step "FAIL tag=$TAG flip rc=$flip_rc tail=$(tail -3 "$OUT/flip_$TAG.log" | tr '\n' '|' | cut -c1-400)"; exit 3; }

# The flip's own smoke already ran the 32k prefill twice (SMOKE_PREFILL_ARMS, ratio-checked), so the
# prefill numbers are free -- but they are buried in the arm log the smoke names, which is why they are
# read back here rather than re-measured: PREFILL_ARM says what the client asked for and got, and
# SEAM_NEGOTIATED says what the worker *granted* (chunk_rows must be 8192, or the piece sizes -- and
# therefore the tier the MoE kernels picked -- are not the ones the run claims).
pf_arm=""
while read -r p; do [ -n "$p" ] && pf_arm=$p; done <<EOF
$(grep -h '^SMOKE_PREFILL_LOG ' "$OUT/flip_$TAG.log" 2>/dev/null | sed 's/.*SMOKE_PREFILL_LOG //')
EOF
if [ -n "$pf_arm" ] && [ -r "$pf_arm" ]; then
  grep -hE '^PREFILL_ARM |^SEAM_NEGOTIATED ' "$pf_arm" | sed "s/^/PREFILL_DETAIL tag=$TAG src=$(basename "$pf_arm") /" | cut -c1-430
else
  step "PREFILL_DETAIL tag=$TAG none (no readable SMOKE_PREFILL_LOG in the flip log)"
fi

# The one line that makes the arm real: the release the pair's manifest names, the sha `current`
# points at, and the build the worker was started with. A/B across two arms is meaningless if any of
# the three silently names the other arm's release.
cur=$(pair_current_sha "$PAIR_ROOT" 2>/dev/null); cur_gx=$(pair_current_sha "$PAIR_ROOT" 1 2>/dev/null)
man=$(pair_manifest_get release_sha 2>/dev/null)
srv=$(pair_gx_serving_build 2>/dev/null)
step "ARM_IDENTITY tag=$TAG sha=${SHA:0:12} current=${cur:0:12} current_gx=${cur_gx:0:12} manifest=${man:0:12} serving_build=${srv:-none}"
case "$cur" in "$SHA"*|"${SHA:0:12}"*) ;; *) step "FAIL tag=$TAG current=$cur is not the release we flipped to"; exit 4 ;; esac
case "$cur_gx" in "$SHA"*|"${SHA:0:12}"*) ;; *) step "FAIL tag=$TAG the GX half's current=$cur_gx is not the release we flipped to -- a split pair measures neither arm"; exit 4 ;; esac

# placement must be the ship layout on the arm we measure, or the numbers are another pack's
place=$(pair_cold_placement "$GXCT" 2>/dev/null)
step "ARM_PLACEMENT tag=$TAG sha=${place:-none}"
[ -n "$place" ] || step "WARN tag=$TAG placement read failed (a transport miss is not a pack verdict)"

BUILD_ID=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["build_id"])' \
            "$(pair_release_dir "$SHA")/RELEASE.json" 2>/dev/null) || BUILD_ID=""
[ -n "$BUILD_ID" ] || { step "FAIL tag=$TAG no build_id in the release we flipped to"; exit 4; }
step "ARM_BUILD tag=$TAG build_id=$BUILD_ID"

# ---- 2. own the box: claim, both containers, an import that is a hit, one worker load ------------
step "GATE_START tag=$TAG label=$GATE_LABEL"
if ! pair_gx_gate_take "$GATE_LABEL" $(( SERVER_TIMEOUT / 60 + 30 )) "${PROMOTE_GATE_WAIT_MIN:-25}"; then
  step "FAIL tag=$TAG gate_busy $(pair_gx_gate_state)"; exit 8
fi
step "GATE_HELD tag=$TAG $(cat "$(pair_gx_gate_file)")"
pair_ensure_nvml p7 "$P7CT" || { step "FAIL tag=$TAG nvml p7=$P7CT"; exit 8; }
pair_ensure_nvml gx "$GXCT" || { step "FAIL tag=$TAG nvml gx=$GXCT"; exit 8; }
for _side in p7 gx; do
  if [ "$_side" = p7 ]; then _ct=$P7CT; else _ct=$GXCT; fi
  _cmd='cd /tmp && timeout -k 5 90 python3 -c "import exllamav3.ext as e; print(\"EXT_HIT\", e.exllamav3_ext.__file__)"'
  # The cache is stated, not inherited: the containers default to /pair/scratch-ext since 2026-11-04,
  # and a probe without -e builds into it (90 s timeout, empty output, ext_client_miss -- the 02:29Z
  # cand arm). This is the same rule pair_arm.sh:107 and pair_smoke.sh's probe follow.
  if [ "$_side" = p7 ]; then _out=$(docker exec -e TORCH_EXTENSIONS_DIR=/pair/extcache "$_ct" sh -c "$_cmd" 2>&1); else _out=$(pair_gxsh "docker exec -e TORCH_EXTENSIONS_DIR=/pair/extcache $_ct sh -c '$_cmd'" 2>&1); fi
  case "$_out" in
    *EXT_HIT*) step "EXT_HIT tag=$TAG side=$_side $(printf '%s' "$_out" | grep -m1 EXT_HIT | cut -c1-120)" ;;
    *) step "FAIL tag=$TAG ext_client_miss side=$_side out=$(printf '%s' "$_out" | tail -2 | tr '\n' '|' | cut -c1-300)"; exit 9 ;;
  esac
done
[ -x "$ENSURE" ] || { step "FAIL tag=$TAG no seam_server_ensure (${ENSURE:-<not found>})"; exit 8; }
step "SERVER_START tag=$TAG build=$BUILD_ID buckets=$BUCKETS timeout=${SERVER_TIMEOUT}s gx=$(gx_state)"
srv_out=$(GX="$GX" CT="$GXCT" GX_DEV="${GX_DEV:-rocep1s0f0}" GX_GID="$GX_GID" PORT="$PORT" \
          SEAM_GATE_DIR="$(dirname "$(pair_gx_gate_file)")" \
          SERVER_BUCKETS="$BUCKETS" SEAM_SERVER_TIMEOUT="$SERVER_TIMEOUT" \
          SEAM_ENV="$(pair_server_env "$BUILD_ID")" \
          "$ENSURE" --fresh --buckets "$BUCKETS" 2>&1)
srv_rc=$?
printf '%s\n' "$srv_out" | grep -E '^(SEAM_ENSURE|SEAM_SLACK|SEAM_NO_MEMORY|SEAM_NO_PACK|PREFLIGHT_FAIL)' \
  | sed "s/^/SERVER $TAG /" | cut -c1-220
step "SERVER_END tag=$TAG rc=$srv_rc host_load=$(host_load) gx=$(gx_state) workers=$(workers)"
case "$srv_out" in
  *SEAM_NO_MEMORY*) step "FAIL tag=$TAG memory server_no_memory $(printf '%s' "$srv_out" | grep -m1 SEAM_NO_MEMORY)"; exit 10 ;;
  *SEAM_NO_PACK*)   step "FAIL tag=$TAG no_pack $(printf '%s' "$srv_out" | grep -m1 SEAM_NO_PACK)"; exit 10 ;;
esac
[ "$srv_rc" = 0 ] || { step "FAIL tag=$TAG server rc=$srv_rc $(printf '%s' "$srv_out" | grep -m1 -E 'SEAM_ENSURE|PREFLIGHT_FAIL')"; exit 10; }
# Record the worker we just started, exactly as pair_serve.sh:117 and pair_smoke.sh:347 do. The ensure
# launches it with `docker exec -d` and writes no pidfile, and pair_server_stop stops only "the recorded
# worker" -- so without this line on_exit left 113 GiB resident on 2026-11-04 04:03Z. pair_server_record
# also anchors on '^python3 -u qmesh/worker_seam_server.py', which is what keeps it off the wrapper pid.
if pair_server_record "$BUILD_ID"; then
  step "SERVER_RECORDED tag=$TAG $(cat "$(pair_server_pidfile)")"
else
  step "SERVER_RECORD_MISS tag=$TAG (no anchored worker pid; on_exit's kill_worker is the fallback)"
fi
# Which extension artifact the worker actually imported: the container default became
# /pair/scratch-ext on 2026-11-04, and the 02:07Z cand arm proved the cost of letting the worker
# inherit it (it ran a .so built inside the measurement window). ext_worker_check.sh states the
# live worker's /proc env and the .so behind /pair/extcache against the release's BUILD_STAMP.
"$S/ext_worker_check.sh" "$TAG" || exit 11
serving=$(pair_gx_serving_build 2>/dev/null)
case "$serving" in
  "$BUILD_ID"*) step "SERVER_BUILD_OK tag=$TAG serving=$serving" ;;
  *) step "FAIL tag=$TAG serving_build='${serving:-none}' is not $BUILD_ID -- the worker is another release's"; exit 10 ;;
esac

# PROMOTE_SKIP_RATE=1 measures quality only (a repeat or a paired ppl on a release whose decode row is
# already in the record): the same session, the same load path, one flip, no 4 minutes of row arm.
if [ "${PROMOTE_SKIP_RATE:-0}" = 1 ]; then
  step "RATE_SKIPPED tag=$TAG (PROMOTE_SKIP_RATE=1) host_load=$(host_load) p7=$(p7_clocks) gx=$(gx_state)"
  rate_rc=0
else
  # ---- 3. decode row at the shipped shape ----------------------------------------------------------
  step "RATE_START tag=$TAG rows=$ROWS max_new=$MAXNEW host_load=$(host_load) p7=$(p7_clocks) gx=$(gx_state)"
  ( cd "$S" && ARM_IDS_OUT="$OUT/ids_$TAG.json" timeout 1800 ./pair_arm.sh dev --ids --rows "$ROWS" \
      --max-new "$MAXNEW" --warmup 1 --timeout 1500 --out "$OUT/row_$TAG.jsonl" ) >"$OUT/rate_$TAG.log" 2>&1
  rate_rc=$?
  grep -E 'ARM_IDS|ms/tok|tok/s|SUMMARY|CLIENT_PLACEMENT' "$OUT/rate_$TAG.log" | sed "s/^/RATE $TAG /" | cut -c1-220
  # pair_arm writes its row summary as a `SUMMARY {...}` line in the jsonl, not to stdout, so an arm could
  # come back rc=0 with nothing quotable in the ticket log (exactly what happened at 04:03:43Z). State the
  # number here and make its absence a failed arm rather than a silent pass.
  rowsum=$(python3 -c 'import json,sys;L=[l for l in open(sys.argv[1]) if l.startswith("SUMMARY")];d=json.loads(L[-1][8:]) if L else None;ms=(d or {}).get("decode_ms_per_tok_mean") or 0;print("n=%s ms_per_tok=%.3f tok_per_s=%.2f ttft_ms=%.1f"%(d.get("n"),ms,(1000.0/ms) if ms else 0,d.get("ttft_ms_mean") or 0)) if d else 0' "$OUT/row_$TAG.jsonl" 2>/dev/null)
  if [ -n "$rowsum" ]; then
    step "DECODE_SUMMARY tag=$TAG $rowsum release=${SHA:0:12}"
  else
    step "DECODE_SUMMARY tag=$TAG missing (no SUMMARY line in row_$TAG.jsonl)"
    [ "$rate_rc" = 0 ] && rate_rc=6
  fi
fi
step "RATE_END tag=$TAG rc=$rate_rc host_load=$(host_load) p7=$(p7_clocks) gx=$(gx_state)"

# The fused counter, read from the *decode* load and nothing else. It is not in this position by
# accident: SERVER_START above is a --fresh worker with no prefill in front of it, so `fused_pct` here is
# a statement about rows=256 decode. Reading it after the flip instead would mix 8192-row prefill pieces
# into the same cumulative counters (the shipped gate is bm[g]==16, so those legitimately refuse the
# fused branch) and turn a 100 % decode number into an unreadable blend. Several lines are printed per
# load on a 4096-call throttle, so the last line of that log is the one to quote, and `knob=` is what
# separates "gate refused the shape" (knob=1 fused=0) from "the env never arrived" (knob=0). A build
# with no counter at all prints nothing here -- the ref arm predates the knob, so its absence is
# expected and is itself proof the two arms run different code.
# Ask the worker first, and believe nothing else. The counter is written by the worker's stderr, and
# the worker for *this* server is still up here (cleanup runs at the end), so /tmp/seam_server.log
# inside the GX container is this load's stderr: it is truncated on every worker start, which is what
# makes it this arm's and no one else's. The release-side archives are only a fallback, and only when
# they are newer than this arm's own start -- without that test the reader picked up a stale
# server---*.log from an earlier lane and printed its fused_pct=100.0 as this arm's evidence (ticket
# 3312781, both arms, while the ref build could not print the line at all).
fused_src=worker-stderr
live=$(pair_gxsh "docker exec $GXCT sh -c 'grep -h SEAM_MOE_CPP_FUSED_STATS /tmp/seam_server.log 2>/dev/null | tail -3'" 2>/dev/null)
nlive=$(printf '%s\n' "$live" | grep -cE 'SEAM_MOE_CPP_FUSED_STATS' || true)
if [ "${nlive:-0}" -gt 0 ]; then
  step "FUSED_SRC tag=$TAG src=gx:/tmp/seam_server.log lines=$nlive"
  printf '%s\n' "$live" | grep -hE 'SEAM_MOE_CPP_FUSED_STATS' | tail -2 \
    | sed "s/^/FUSED_DETAIL tag=$TAG src=worker-stderr /" | cut -c1-300
else
  fused_seen=0
  for l in $(ls -t "$PAIR_STATE_DIR"/logs/server---*.log 2>/dev/null | head -4); do
    [ "$l" -nt "$OUT/arm_start_$TAG.stamp" ] || continue
    n=$(grep -cE 'SEAM_MOE_CPP_FUSED_STATS' "$l" 2>/dev/null || true)
    [ "${n:-0}" -gt 0 ] || continue
    fused_seen=1
    step "FUSED_SRC tag=$TAG src=$(basename "$l") lines=$n"
    grep -hE 'SEAM_MOE_CPP_FUSED_STATS' "$l" | tail -2 | sed "s/^/FUSED_DETAIL tag=$TAG /" | cut -c1-300
  done
  [ "$fused_seen" = 1 ] || step "FUSED_SRC tag=$TAG none (worker stderr empty, no archive newer than this arm)"
fi

# ---- 4. quality: wiki2 quick8 against the ship reference ----------------------------------------
ckpt=$(pair_ckpt_resolve auto) || { step "FAIL tag=$TAG cannot resolve the client half for $(pair_cold_src "$GXCT")"; exit 5; }
step "PPL_START tag=$TAG ckpt=$ckpt build=$BUILD_ID host_load=$(host_load) gx=$(gx_state)"
# The container exposes the pair's card as device 0 (pairs.conf GPU= is the *host* index, which is 1):
# `-e CUDA_VISIBLE_DEVICES=$GPU` gave the client no GPU at all -- "RuntimeError: No CUDA GPUs are
# available", PPL_END rc=1 in 5 s at 04:03:45Z. pair_arm.sh:124 uses 0 for the same reason.
# Keep this comment *above* the command: inside the argument list its backslash continuation turned the
# comment into docker's argument vector and every option after it into a command ("-e: command not
# found", PPL_END rc=127 at 05:05:11Z).
timeout 2700 docker exec \
  -e "CUDA_VISIBLE_DEVICES=0" \
  -e "REMOTE_TRANSPORT=seam" \
  -e "SEAM_DEV=$SEAM_DEV" -e "SEAM_GID=$SEAM_GID" -e "SEAM_PEER=$SEAM_PEER" \
  -e "SEAM_BUCKETS=$BUCKETS" \
  -e "CHECKPOINT=$ckpt" -e "SEAM_BUILD_ID=$BUILD_ID" \
  -e "KV_BITS=4" -e "EXL3_WHOLE_GRAPH=0" -e "MAX_CHUNK=8192" \
  -e "EXL3_MOE_SPLIT_FUSED=0" \
  -e "SEAM_ROWS=8192" -e "SEAM_NS=1" -e "SEAM_OVERLAP=1" -e "SEAM_COPY_STREAM=1" \
  -e "WIKITEXT_ZIP=$Q_CT/wikitext-2-raw-v1.zip" \
  -e "PYTHONPATH=/src/exllamav3" \
  -e "TORCH_EXTENSIONS_DIR=/pair/extcache" \
  "$P7CT" bash -lc "cd /tmp && python3 -u $PY_CT --mode wiki2 --sanity --chunks 8 \
    --kv-bits 4 --out $(pair_ct_path "$OUT/ppl_$TAG.json")" \
  >"$OUT/ppl_$TAG.log" 2>&1
ppl_rc=$?
grep -E 'WIKI2 QUICK8|PANEL |WIKI2 RESULT' "$OUT/ppl_$TAG.log" | sed "s/^/PPL $TAG /" | cut -c1-200
if grep -qiE 'building extension|running ninja|torch\.utils\.cpp_extension' "$OUT/ppl_$TAG.log"; then
  step "FAIL tag=$TAG the client rebuilt an extension inside the arm -- numbers void"; ppl_rc=9
fi
step "PPL_END tag=$TAG rc=$ppl_rc host_load=$(host_load) gx=$(gx_state)"

# ---- 5. restore (optional) ----------------------------------------------------------------------
if [ -n "$RESTORE" ]; then
  rs=$(git -C "$PAIR_REPO" rev-parse --verify "$RESTORE^{commit}" 2>/dev/null) || rs=""
  if [ -n "$rs" ]; then
    step "RESTORE_START tag=$TAG to=${rs:0:12}"
    PAIR_FLIP_ONLY=1 "$S/pair_update_run.sh" dev "$rs" >"$OUT/restore_$TAG.log" 2>&1
    step "RESTORE_END tag=$TAG rc=$? current=$(pair_current_sha "$PAIR_ROOT" 2>/dev/null | cut -c1-12)"
  fi
fi

rc=0
[ "$rate_rc" = 0 ] || rc=6
[ "$ppl_rc" = 0 ] || rc=7
step "ARM_DONE tag=$TAG sha=${SHA:0:12} rc=$rc"
exit "$rc"
