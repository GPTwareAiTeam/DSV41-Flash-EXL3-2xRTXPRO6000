#!/bin/bash
# ids.sh -- position-1 id-class measurement: one control arm and one EXL3_DSPARK_IDSTATS=1 arm, back to
# back, same session, same prompts, same everything else, so the capture's cost and its token identity can
# be judged against control. Greedy tokens must be identical across the two arms (the capture reads ids
# that were already compared; it changes nothing but a host sync), and the timing delta is the price of
# that sync.
#
#   bash ids.sh <tag>            # needs the dev pair, GPU1 only, queue/run_dev.sh
#
# Refuses on a paging GX server (gate_gx.sh), because a paged pack changes routing and therefore tokens.
set -u -o pipefail
TAG=${1:-ids}
HERE=/home/treadall/recipe-artifacts/dspark-tpr
SRC=$HERE/src
PROMPTS=${PROMPTS:-/artifacts/2x-rtx-tp2/dtrain/prompts_held.json}
OUT=$HERE/logs
mkdir -p "$OUT"
# A --src arm must never build over the pair's release extension: a different source path gives torch a
# different hash, so load() compiles fresh, and landing that in /pair/extcache put dev into DRIFT at
# 20:59:31. Own cache, prebuilt out of band by prebuild_ext.py, and refuse to run without it.
EXT_CACHE=/artifacts/dspark-tpr/extcache
[ -n "$(ls -A "$EXT_CACHE"/exllamav3_ext 2>/dev/null)" ] || {
  echo "ABORT reason=ext_cache_missing path=$EXT_CACHE (run prebuild_ext.py first)"; exit 4; }
say() { echo "$(date +%H:%M:%S) $*"; }

say "IDS_START tag=$TAG"
source "$HERE/gate_gx.sh"
gx_gate || { say "IDS_ABORT tag=$TAG reason=gx_paging"; exit 3; }

arm() {  # arm <name> <extra args...>
  local name=$1; shift
  local log=$OUT/$TAG-$name.log
  say "IDS_ARM tag=$TAG arm=$name log=$log"
  bash /home/treadall/recipe-dspark-tpr/qmesh/release/pair_arm.sh \
    --src "$SRC" --prompts "$PROMPTS" --rows 3 --max-new 256 --num-draft 5 \
    --env TORCH_EXTENSIONS_DIR=/artifacts/dspark-tpr/extcache \
    "$@" > "$log" 2>&1
  local rc=$?
  say "IDS_ARM_END tag=$TAG arm=$name rc=$rc"
  grep -E "^\[|SUMMARY " "$log" | tail -3
  [ $rc -eq 0 ] && cp "$log" "$OUT/$TAG-$name.jsonl" 2>/dev/null
  return $rc
}

# Both arms carry POSSTATS so p1 is comparable like-for-like (it is a histogram, no host read);
# only the second adds the id capture, which is the one that reads ints back to the host.
arm ctl --env EXL3_DSPARK_POSSTATS=1 || { say "IDS_FAIL tag=$TAG arm=ctl"; exit 2; }
arm ids --env EXL3_DSPARK_POSSTATS=1 --env EXL3_DSPARK_IDSTATS=1 || { say "IDS_FAIL tag=$TAG arm=ids"; exit 2; }

say "IDS_DONE tag=$TAG"
python3 "$HERE/ids_compare.py" "$OUT/$TAG-ctl.log" "$OUT/$TAG-ids.log" || true
