#!/bin/bash
# ab_k35.sh -- the draft-window K A/B on the pair's served release.
#
# Arms (identical client, identical prompts, exactly one gate difference each):
#   fixed5  EXL3_DSPARK_GATE=thr  CONF=0   the full 5-draft window is always verified
#   auto    EXL3_DSPARK_GATE=auto CONF=0   the shipped content-adaptive threshold (hi=0 here, lo=0.25)
#   k35     EXL3_DSPARK_GATE=ev   CONF=0   the expected-value chooser with k in {3,5} priced and
#                                          1,2,4 priced out of the table (EV_KMIN=3, cost 1:1e9..)
# CONF=0 on all three: without it the `thr`/`auto` arms also stop early on low confidence, and the
# comparison would change two things at once. Under gate=ev CONF is inert (the cost table decides).
#
# Content arms: the whole 40-prompt held list, MAXNEW decode tokens each, one prompt at a time (the
# driver enqueues sequentially). Per-prompt rows are what make the five content medians possible; the
# labels are p0..p39 and the classification is by index (see ab_k35_summary.py).
# Order ABBA over the two threshold arms, k35 last in each pass.
# 32k arms: one long prompt, the same three policies, so the wider-window cost shows where the round is
# already dominated by attention over a long KV.
#
# Every arm is bracketed by the GX swap window (gate_gx.sh --arm-start / --arm-end, SWAP GATE = OPTION B
# in gate_gx.sh's header): an arm whose worker re-faulted >128 MiB of its own pack inside the decode
# window is written $OUT/<arm>.void and excluded, an arm whose box was demonstrably not post-residency is
# $OUT/<arm>.nojudge and excluded. The verdicts land in $OUT/<arm>.gate.txt.
#
# Identity: every arm runs through pair_arm.sh --ids, so $OUT/<arm>.ids.txt carries `ARM_IDS … sha256=`
# over the ids the engine actually generated, and the per-prompt `text` in the rows is the local tripwire.
#
# Usage:
#   ab_k35.sh [tag]                    the K A/B (queue it; do not run it ahead of the canary)
#   POSSTATS=1 ab_k35.sh [tag]         the per-position acceptance pass (needs --src: the release's own
#                                      driver has no pos_summary), fixed5 and k35 only
#   MAXNEW=128 NR=40 OVERRIDES: NR, MAXNEW, MAXNEW32K, SKIP32K=1, PROMPTS, PROMPTS_32K, OUT, PAIR_ARM
#   SRC_TREE=/artifacts/dspark-tpr/src  client tree for every arm (POSSTATS=1 implies it)
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
TAG=${1:-k35}
POSSTATS=${POSSTATS:-0}
OUT=${OUT:-$HERE/logs/$TAG}
PROMPTS=${PROMPTS:-/artifacts/2x-rtx-tp2/dtrain/prompts_held.json}
P32=${PROMPTS_32K:-/artifacts/ppl-4460/prose_32k.json}
ARM=${PAIR_ARM:-dev}
NR=${NR:-40}
MAXNEW=${MAXNEW:-128}
MAXNEW32K=${MAXNEW32K:-256}
SKIP32K=${SKIP32K:-0}
SRC_TREE=${SRC_TREE:-}
PAIR_ARM_SH=/home/treadall/recipe-artifacts/gpu0-gx1/scripts/pair_arm.sh
PAIRS=/home/treadall/recipe-artifacts/gpu0-gx1/pairs
E_FIXED="EXL3_DSPARK_GATE=thr EXL3_DSPARK_CONF=0"
E_AUTO="EXL3_DSPARK_GATE=auto EXL3_DSPARK_CONF=0"
E_K35="EXL3_DSPARK_GATE=ev EXL3_DSPARK_CONF=0 EXL3_DSPARK_EV_KMIN=3"\
" EXL3_DSPARK_EV_COST=1:1e9,2:1e9,3:20.7,4:1e9,5:22.2"
[ "$POSSTATS" = 1 ] && SRC_TREE=${SRC_TREE:-/artifacts/dspark-tpr/src}
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/arm.log"; }
gate_rc_of() { sed -n 's/^gate_rc=\([0-9]*\)$/\1/p' "$1" | tail -1; }
clock() { nvidia-smi --query-gpu=index,clocks.sm,temperature.gpu --format=csv,noheader \
          | grep -E '^1,' || echo "gpu1 clock unavailable"; }

# The box gate, before anything is measured: a paging worker changes routing, and a routing change makes
# every acceptance number here unquotable, not just slow. gate_gx.sh prints its verdict on `gate_rc=`.
bash "$HERE/gate_gx.sh" > "$OUT/gate.txt" 2>&1
grep -E '^(GX_MEM|GX_SHAPE|GX_WARN|GX_NOTE|GX_REFUSE|GX_NOJUDGE)' "$OUT/gate.txt"
rc=$(gate_rc_of "$OUT/gate.txt"); rc=${rc:-1}
[ "$rc" = 0 ] || [ "$rc" = 2 ] || { echo "ABORT reason=gx_gate gate_rc=$rc (see $OUT/gate.txt)" >&2; exit 3; }
[ "$rc" = 2 ] && say "AB_WARN gate_rc=2 (not post-residency): arms will be measured, the summary will exclude unjudgeable ones"
# Fresh load only: a worker that has already taken long prefills is a different memory shape.
age=$(sed -n 's/.*worker_age_min=\([0-9]*\).*/\1/p' "$OUT/gate.txt")
if [ -n "$age" ] && [ "$age" -gt 45 ]; then echo "ABORT reason=stale_worker worker_age_min=$age > 45" >&2; exit 5; fi
wsr=$(sed -n 's/.*ws_rows=\([^ ]*\).*/\1/p' "$OUT/gate.txt" | head -1)
sha_now() { readlink -f "$PAIRS/dev/current" 2>/dev/null | sed 's#.*/##' | cut -c1-12; }
REL=$(sha_now)
say "AB_START tag=$TAG release=${REL:-?} ws_rows=${wsr:-?} nr=$NR maxnew=$MAXNEW posstats=$POSSTATS src=${SRC_TREE:-release}"
echo "eta_min=30 doing=ab_k35 gate_ok start" > "${GPU_RUN_STATUS:-/tmp/gpu_run_status}"
LAST_WALL=30
eta() { echo "eta_min=$(( LAST_WALL * $1 / 60 + 1 )) doing=ab_k35 arms_left=$1" \
          > "${GPU_RUN_STATUS:-/tmp/gpu_run_status}"; }

run() { # run <arm> <envs> <prompts> <rows> <maxnew> [pair_arm.sh args]
  local a=$1 e=$2 p=$3 r=$4 n=$5; shift 5
  local t0=$SECONDS sflag=()
  [ -n "$SRC_TREE" ] && sflag=(--src "$SRC_TREE")
  bash "$HERE/gate_gx.sh" --arm-start "$a" > "$OUT/$a.gate.txt" 2>&1
  local g; g=$(gate_rc_of "$OUT/$a.gate.txt")
  if [ "${g:-1}" != 0 ]; then say "ARM $a SKIP gate_pre_rc=${g:-?}"; touch "$OUT/$a.void"; return 9; fi
  clock > "$OUT/$a.clock.pre"; uptime >> "$OUT/$a.clock.pre"
  ARM_IDS_OUT=$OUT/$a.ids.txt PAIR_ARM_EXTRA_ENVS="$e" PAIR_ARM_PROMPTS="$p" \
    bash "$PAIR_ARM_SH" "$ARM" --rows "$r" --max-new "$n" --warmup 0 --timeout 900 --ids \
      --out "$OUT/$a.rows.jsonl" ${sflag[@]+"${sflag[@]}"} "$@" > "$OUT/$a.stdout" 2> "$OUT/$a.stderr"
  local crc=$?
  clock > "$OUT/$a.clock.post"
  case "$a" in *32k*) bash "$HERE/gate_gx.sh" --arm-end "$a" --prefill >> "$OUT/$a.gate.txt" 2>&1 ;;
                 *)    bash "$HERE/gate_gx.sh" --arm-end "$a"           >> "$OUT/$a.gate.txt" 2>&1 ;; esac
  local g2; g2=$(gate_rc_of "$OUT/$a.gate.txt")
  case "${g2:-?}" in
    1) touch "$OUT/$a.void";    say "ARM $a VOID: swap growth refused (see $OUT/$a.gate.txt)" ;;
    2) touch "$OUT/$a.nojudge"; say "ARM $a NOJUDGE: not post-residency, excluded from the table" ;;
  esac
  [ "$crc" -ne 0 ] && touch "$OUT/$a.void"
  local rows; rows=$(grep -c '"label"' "$OUT/$a.rows.jsonl" 2>/dev/null); rows=${rows:-0}
  LAST_WALL=$(( SECONDS - t0 ))
  say "ARM $a rc=$crc rows=$rows wall_s=$LAST_WALL gate_post_rc=${g2:-?} ids=$(sed -n 's/.*sha256=\([0-9a-f]\{8\}\).*/\1/p' "$OUT/$a.ids.txt" 2>/dev/null | head -1)"
  echo "ARM $a rc=$crc rows=$rows wall_s=$LAST_WALL gate_post_rc=${g2:-?} $(clock | head -1)" >> "$OUT/arm.log"
  return $crc
}

# Discarded warm arms. There is no saved autotune file on GX2, so the first client after a server start
# pays the coop-GEMM tune inside its own window (the ~1.5 s round the 2xRTX notes record as a false
# 15-20% loss). One short run per shape family, never printed, so every timed arm measures a warm pair.
run "${TAG}_warm" "" 1 32 "$PROMPTS"
mv "$OUT/${TAG}_warm.rows.jsonl" "$OUT/warm.discarded.jsonl" 2>/dev/null
if [ "$SKIP32K" != 1 ]; then
  run "${TAG}_warm32k" "$E_FIXED" 1 32 "$P32"
  mv "$OUT/${TAG}_warm32k.rows.jsonl" "$OUT/warm32k.discarded.jsonl" 2>/dev/null
fi

if [ "$POSSTATS" = 1 ]; then
  # Per-position acceptance per content: only the shipped policy and the candidate, so the pass is two
  # arms. POSSTATS syncs once per verify round, so these arms are for shape, not for tok/s.
  # The --src tree has its own extension cache; a probe first, because a cache miss there would start a
  # CUDA build inside the arm and the ticket would be spent compiling instead of measuring.
  eta 3
  run "${TAG}_probe" "$E_FIXED" "$PROMPTS" 1 8
  if [ -f "$OUT/${TAG}_probe.void" ] || [ "$LAST_WALL" -gt 120 ]; then
    say "AB_ABORT reason=src_probe wall_s=$LAST_WALL (build or load problem in $SRC_TREE, see $OUT/${TAG}_probe.stderr)"
    exit 8
  fi
  mv "$OUT/${TAG}_probe.rows.jsonl" "$OUT/probe.discarded.jsonl" 2>/dev/null
  run "${TAG}_fixed5pos" "$E_FIXED EXL3_DSPARK_POSSTATS=1" "$PROMPTS" "$NR" "$MAXNEW"
  run "${TAG}_k35pos"    "$E_K35 EXL3_DSPARK_POSSTATS=1"   "$PROMPTS" "$NR" "$MAXNEW"
  python3 "$HERE/ab_k35_summary.py" "$OUT" 2>&1 | tail -60
  exit 0
fi

for pass in A B; do
  eta 6
  if [ "$pass" = A ]; then
    run "${TAG}_auto_p${pass}"   "$E_AUTO"  "$PROMPTS" "$NR" "$MAXNEW"
    run "${TAG}_fixed5_p${pass}" "$E_FIXED" "$PROMPTS" "$NR" "$MAXNEW"
  else
    run "${TAG}_fixed5_p${pass}" "$E_FIXED" "$PROMPTS" "$NR" "$MAXNEW"
    run "${TAG}_auto_p${pass}"   "$E_AUTO"  "$PROMPTS" "$NR" "$MAXNEW"
  fi
  run "${TAG}_k35_p${pass}" "$E_K35" "$PROMPTS" "$NR" "$MAXNEW"
  now=$(sha_now)
  if [ "$now" != "$REL" ]; then
    say "AB_ABORT release_flip $REL -> $now after pass $pass: the passes are not comparable"
    python3 "$HERE/ab_k35_summary.py" "$OUT" 2>&1 | tail -60
    exit 7
  fi
  eta 4
done
if [ "$SKIP32K" != 1 ]; then
  # 32k context: one long prompt per policy, so the wider window's cost shows where the round is already
  # dominated by attention. These are prefill-window arms: swap growth here is warned, not refused.
  eta 3
  run "${TAG}_fixed5_32k" "$E_FIXED" "$P32" 1 "$MAXNEW32K"
  run "${TAG}_auto_32k"   "$E_AUTO"  "$P32" 1 "$MAXNEW32K"
  run "${TAG}_k35_32k"    "$E_K35"   "$P32" 1 "$MAXNEW32K"
fi
python3 "$HERE/ab_k35_summary.py" "$OUT" 2>&1 | tail -60
