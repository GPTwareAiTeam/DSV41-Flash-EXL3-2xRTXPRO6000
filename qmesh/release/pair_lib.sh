#!/usr/bin/env bash
# pair_lib.sh -- shared helpers for the pair release/update system (see qmesh/notes/PairRelease.md).
# Sourced, never executed. Every script in qmesh/release sources it so that the pair layout, the ssh
# and docker quoting and the release identity live in exactly one place.
#
# Nothing here is allowed to guess: a missing pair, a missing manifest or an unreachable box is an
# error, never a default.

PAIR_HOME=${PAIR_HOME:-/home/treadall}
PAIR_REPO=${PAIR_REPO:-$PAIR_HOME/recipe-gpu0-gx1-v2}
PAIR_ART=${PAIR_ART:-$PAIR_HOME/recipe-artifacts/gpu0-gx1}
PAIR_SCRIPTS=${PAIR_SCRIPTS:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
PAIR_STATE=${PAIR_STATE:-$PAIR_ART/pairs}
PAIR_QUEUE=${PAIR_QUEUE:-$PAIR_ART/queue}
PAIR_CONF=${PAIR_CONF:-$PAIR_SCRIPTS/pairs.conf}
PAIR_BRANCH=${PAIR_BRANCH:-tools/pair-release}
PAIR_SSH=${PAIR_SSH:-ssh}
PAIR_DRY=${PAIR_DRY:-0}
PAIR_DRY_SSH=${PAIR_DRY_SSH:-}
PAIR_MOUNT=${PAIR_MOUNT:-/pair}                 # container path of the pair root
PAIR_SRC_LINK=${PAIR_SRC_LINK:-/src/exllamav3}  # container symlink -> /pair/current/src
PAIR_LIB=libqmesh_expert_seam                   # qmesh/native/bin/<this>.<arch>.so
PAIR_SEAM_FILES=(qmesh/native/qmesh_expert_seam.cu qmesh/native/qmesh_expert_seam.h qmesh/native/qes.py)
PAIR_VERIFY_PATHS=(qmesh exllamav3)             # what "the tree is the commit" covers
PAIR_SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=3)

pair_die() { echo "pair: $*" >&2; exit 2; }
pair_say() { echo "$(date '+%H:%M:%S') $*"; }
# pair_step: timestamped line, also appended to $PAIR_LOG when set (every job sets it).
pair_step() { say_line=$(printf '%s %s' "$(date '+%H:%M:%S')" "$*"); echo "$say_line"; [ -n "${PAIR_LOG:-}" ] && echo "$say_line" >> "$PAIR_LOG"; return 0; }

pair_load() {  # <prod|dev> -> PAIR P7CT GPU SEAM_DEV SEAM_GID GX_GID SEAM_PEER GX GXCT PAIR_ROOT PORT BUCKETS P7_ARCH GX_ARCH LOAD_MARGIN
  local p=${1:?usage: pair_load <pair>} conf=${2:-$PAIR_CONF} line
  [ -r "$conf" ] || pair_die "config not readable: $conf"
  line=$(awk -v p="$p" '!/^#/ && NF && $1==p {print; exit}' "$conf")
  [ -n "$line" ] || pair_die "pair '$p' is not in $conf"
  read -r PAIR P7CT GPU SEAM_DEV SEAM_GID GX_GID SEAM_PEER GX GXCT PAIR_ROOT PORT BUCKETS P7_ARCH GX_ARCH LOAD_MARGIN WS_ROWS <<<"$line"
  PAIR_ROOT=${PAIR_ROOT_OVERRIDE:-$PAIR_ROOT}
  # A relative root puts every release, log and stamp wherever the caller happened to stand: a
  # stubbed harness once grew four full source trees inside that repository way.
  case "$PAIR_ROOT" in
    /*) ;;
    *) pair_die "pair root must be absolute: '$PAIR_ROOT'" ;;
  esac
  # LOAD_MARGIN: what the pack load must leave behind, in MiB. Measured on both GX boxes, the load
  # consumes the pack plus ~1200 MiB and compute needs ~1600 MiB more, so the worker's own default
  # (2048) is a floor, not a promise -- the pair states its number here. GX2 has ~120585 MiB clean,
  # so 3072 passes by ~1150 MiB and the box still lands at the wall: the margin can only refuse a
  # dirtier box, it cannot make GX2 fit (qmesh/notes/PairRelease.md).
  LOAD_MARGIN=${LOAD_MARGIN:-2048}
  # WS_ROWS: the row piece a server is asked to take per call, pinned by the pair rather than left
  # to a per-image default. 1024 is the measured shape on both loaded GX boxes (512-row pieces cost
  # -7/-22% and a single 4096 costs 224-326 ms p50 with 3-5 s tails), and the picker that would
  # choose it adapts by itself only when a box has ~1 GiB of headroom, which neither has. "auto"
  # hands the decision back to whatever server binary the release ships.
  WS_ROWS=${WS_ROWS:-1024}
  if [ "$WS_ROWS" = auto ]; then
    WS_ENV=""
  else
    # AUTO=0 is the other half of the pin: without it a picker may widen the piece under memory
    # pressure on its own, and the pair would no longer run the shape its numbers were taken on.
    WS_ENV="SEAM_WS_ROWS=$WS_ROWS SEAM_WS_ROWS_AUTO=0"
  fi
  # MOE_ENV: the kernel the GX half serves the cold experts on. cuda-exl3 restacks a layer's cold
  # experts per bitrate group and is preloaded while the pack loads; without it (a release cut
  # without the vendored extension built -- its .so is untracked, so every cut needs
  # qmesh/build_cudaexl3.sh) the worker prints SEAM_MOE_NOEXT and every layer stays on the mixed-K
  # module, at a fraction of the measured rate. The smoke refuses to call such a release verified.
  # SEAM_CX_MARGIN_MIB is what must stay free after the load for the restack + workspace (the
  # worker's own default; the validated numbers were taken with it).
  CX_MARGIN_MIB=${CX_MARGIN_MIB:-1536}
  MOE_ENV="SEAM_MOE_KERNEL=cudaexl3 SEAM_MOE_PRELOAD=1 SEAM_CX_MARGIN_MIB=$CX_MARGIN_MIB"
  export PAIR P7CT GPU SEAM_DEV SEAM_GID GX_GID SEAM_PEER GX GXCT PAIR_ROOT PORT BUCKETS P7_ARCH GX_ARCH LOAD_MARGIN WS_ROWS WS_ENV CX_MARGIN_MIB MOE_ENV
  # PAIR_STATE_OVERRIDE lets the stub harnesses (@see tests/dry_run.sh) keep every log, ref and
  # restore note out of the real pairs/ tree: a dry run must never touch what a real one wrote.
  PAIR_STATE=${PAIR_STATE_OVERRIDE:-$PAIR_STATE}
  PAIR_STATE_DIR=$PAIR_STATE/$PAIR
  PAIR_MANIFEST=$PAIR_STATE/$PAIR.json
  export PAIR_STATE_DIR PAIR_MANIFEST
}

# The server half's environment as ONE string: the shipping config every number in the notes was
# taken on. Every starter (the smoke, the restore path, an operator) passes this and nothing else, so
# a server can not come up on a different kernel or piece size than the pair serves.
pair_server_env() {  # <build_id> -> "SEAM_BUILD_ID=… SEAM_LOAD_MARGIN_MIB=… <MOE_ENV> <WS_ENV>"
  printf 'SEAM_BUILD_ID=%s SEAM_LOAD_MARGIN_MIB=%s %s %s' "$1" "$LOAD_MARGIN" "$MOE_ENV" "$WS_ENV"
}

pair_arch_for() {  # <sm_120|sm_121> -> x86_64|aarch64  (lib name per side)
  case "$1" in sm_120) echo x86_64 ;; sm_121) echo aarch64 ;; *) echo "unknown" ;; esac
}
pair_cuda_arch_for() {  # <sm_120|sm_121> -> torch arch list entry
  case "$1" in sm_120) echo 12.0 ;; sm_121) echo 12.1 ;; *) echo "?" ;; esac
}

# ---- remote access ------------------------------------------------------------------------------
# Every GX command goes through pair_gxsh so a dry run can swap the transport (PAIR_DRY_SSH) and so
# no script ever hand-rolls ssh flags. stdin passes through untouched.
# Paths the containers can see. The P7 container mounts the pair root at /pair and the artifacts
# tree at /artifacts (containers/*.json); a host path handed to `docker exec -e` must be
# translated or the file a driver writes "does not exist" from inside.
pair_ct_path() {  # <host path> -> path inside the pair containers
  local p=$1 art=$PAIR_HOME/recipe-artifacts
  case "$p" in
    "$PAIR_ROOT"/*) echo "/pair/${p#"$PAIR_ROOT"/}" ;;
    # the mount is the whole artifacts tree, not the pair's subdirectory: /artifacts/gpu0-gx1/...
    "$art"/*) echo "/artifacts/${p#"$art"/}" ;;
    *) echo "$p" ;;
  esac
}
pair_gxsh() {
  if [ "$PAIR_DRY" = 1 ] && [ -n "$PAIR_DRY_SSH" ]; then
    "$PAIR_DRY_SSH" "${PAIR_SSH_OPTS[@]}" "$GX" "$1"
  else
    "$PAIR_SSH" "${PAIR_SSH_OPTS[@]}" "$GX" "$1"
  fi
}
pair_gxstdin() {
  if [ "$PAIR_DRY" = 1 ] && [ -n "$PAIR_DRY_SSH" ]; then
    "$PAIR_DRY_SSH" "${PAIR_SSH_OPTS[@]}" "$GX" bash -s
  else
    "$PAIR_SSH" "${PAIR_SSH_OPTS[@]}" "$GX" bash -s
  fi
}

# ---- GX box gate ---------------------------------------------------------------------------------
# A GX box carries one cold pack and one server, and two kinds of job want it: a release build (or a
# release smoke) and an experiment that has the pack loaded. They must not overlap -- a compile next to
# a loaded pack moves avail_mib by hundreds of MiB, which both OOM-kills builds and blurs a measurement
# that is about avail_mib. So the box is claimed the way the GPU is: a file, advisory and self-expiring,
# "<owner> pid=… until=<epoch>". A missing file, or a past `until`, means free, so a crashed job cannot
# wedge a box, and a reader never has to trust that the writer cleaned up.
pair_gx_gate_file() {  # -> <queue dir>/gx{N}.gate, N from the GX container name (seam-gx2-dev -> 2)
  local n=${GXCT#seam-gx}
  printf '%s\n' "${PAIR_GATE_DIR:-$PAIR_HOME/recipe-artifacts/gpu0-gx1/queue}/gx${n%%-*}.gate"
}
pair_gx_gate_state() {  # -> "free" or "<owner> until=<epoch>"; a past until is free
  local f o u; f=$(pair_gx_gate_file)
  [ -f "$f" ] || { echo free; return 0; }
  o=$(sed -n 's/^owner=\([^ ]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  u=$(sed -n 's/.*until=\([0-9][0-9]*\).*/\1/p' "$f" 2>/dev/null | head -1)
  case "${u:-}" in ''|*[!0-9]*) echo "foreign_bad_gate ${o:-?}"; return 0 ;; esac
  if [ "$u" -lt "$(date +%s)" ]; then echo free; else echo "${o:-?} until=$u"; fi
}
pair_gx_gate_up() {  # <label> <minutes>
  local f; f=$(pair_gx_gate_file)
  [ "$PAIR_DRY" = 1 ] && return 0
  mkdir -p "$(dirname "$f")" || return 1
  printf 'owner=%s pid=%s until=%s\n' "$1" "$$" "$(( $(date +%s) + ${2:-60} * 60 ))" >"$f"
}
pair_gx_gate_take() {  # <label> <minutes> <wait_minutes> -> 0 claimed, 1 still someone else's
  local st waited=0 limit=$(( ${3:-0} * 6 ))
  while :; do
    st=$(pair_gx_gate_state)
    case "$st" in
      free) pair_gx_gate_up "$1" "$2"; return 0 ;;
    esac
    [ "$waited" -lt "$limit" ] || return 1
    sleep 10; waited=$((waited + 1))
  done
}
pair_gx_gate_down() {  # drop the claim only if the line is still this process's
  local f; f=$(pair_gx_gate_file)
  [ "$PAIR_DRY" = 1 ] && return 0
  [ "$(sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' "$f" 2>/dev/null | head -1)" = "$$" ] && rm -f "$f"
  return 0
}

# ---- release identity ---------------------------------------------------------------------------
pair_release_dir() {  # <sha|sha-prefix> -> release dir; a prefix resolves only when it is unambiguous
  local d=$PAIR_ROOT/releases/$1 m
  if [ ! -d "$d" ]; then
    # A 12-char sha from a log is what humans type; the directory is named by the full one.
    m=$(ls -d "$PAIR_ROOT"/releases/"$1"* 2>/dev/null | head -2)
    if [ -n "$m" ] && [ "$(printf '%s\n' "$m" | wc -l)" = 1 ] && [ -d "$m" ]; then d=$m; fi
  fi
  echo "$d"
}
pair_current_sha() {  # <root> [remote 1] -> sha the `current` symlink points at ("" when absent)
  local root=$1 remote=${2:-0} t
  if [ "$remote" = 1 ]; then
    t=$(pair_gxsh "readlink -f $root/current 2>/dev/null || true")
  else
    t=$(readlink -f "$root/current" 2>/dev/null || true)
  fi
  # readlink -f echoes a missing path unchanged: only a real releases/<sha> counts as current
  case "$t" in "$root"/releases/*) basename "$t" ;; *) return 0 ;; esac
}
# Seam identity: the three files that define the wire. A release that changes any byte of them is a
# different protocol, whatever a constant says, so the fingerprint is the sha256 over their names and
# contents (same on P7 and GX for the same release).
pair_seam_fingerprint() {  # <tree>
  local tree=$1 f
  for f in "${PAIR_SEAM_FILES[@]}"; do
    [ -r "$tree/$f" ] || { echo missing; return 1; }
    printf '%s\0' "$f"; cat "$tree/$f"
  done | sha256sum | cut -c1-16
}
pair_seam_version() {  # <tree> -> PROTO_VERSION / QES_PROTO_VERSION when the tree carries one, else "-"
  local tree=$1 v=""
  v=$(sed -n 's/^PROTO_VERSION *= *\([0-9][0-9]*\).*/\1/p' "$tree/qmesh/native/qes.py" 2>/dev/null | head -1)
  [ -z "$v" ] && v=$(sed -n 's/^#define QES_PROTO_VERSION *\([0-9][0-9]*\).*/\1/p' "$tree/qmesh/native/qmesh_expert_seam.h" 2>/dev/null | head -1)
  echo "${v:--}"
}
# pair_canary_gate <sha> <evidence-file> -- prints "ok" when that sha passed its dev canary, else the
# reasons it did not. ONE definition, because three places must agree on what "dev passed" means:
# pair_update.sh (the fast refusal), pair_promote.sh (the only way in) and pair_update_run.sh (the
# script that actually flips, so queueing it directly is not a way around the gate).
pair_canary_gate() {
  python3 - "${2:?evidence file}" "${1:?sha}" <<'PY'
import json, sys
path, sha = sys.argv[1:3]
try:
    d = json.load(open(path))
except Exception as exc:                                                # noqa: BLE001
    print(f"unreadable {exc}")
    raise SystemExit(0)
bad = []
if d.get("release_sha") != sha:
    bad.append(f"sha={str(d.get('release_sha'))[:12]}")
if d.get("pair") != "dev":
    bad.append(f"pair={d.get('pair')}")
if not d.get("ok"):
    bad.append(f"ok=false at {d.get('failed_at_phase') or '?'} ({d.get('why') or 'no reason'})")
smoke = d.get("smoke") or {}
if not smoke.get("row"):
    bad.append("no_smoke_row")
try:
    n = int((smoke.get("ids") or {}).get("n") or 0)
except (TypeError, ValueError):
    n = 0
if n <= 0:
    bad.append("no_decode_ids")
print(" ".join(bad) if bad else "ok")
PY
}

# ---- manifests ----------------------------------------------------------------------------------
pair_manifest_get() {  # <field.path...> -> value on stdout ("" when absent); reads $PAIR_MANIFEST
  python3 - "$PAIR_MANIFEST" "$@" <<'PY'
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception:
    sys.exit(0)
for key in sys.argv[2:]:
    if not isinstance(d, dict):
        d = None
        break
    d = d.get(key)
    if d is None:
        break
if isinstance(d, (dict, list)):
    print(json.dumps(d))
elif d is not None:
    print(d)
PY
}

# ---- queue --------------------------------------------------------------------------------------
pair_run_script() { echo "$PAIR_QUEUE/run_$PAIR.sh"; }
# Enqueue one job on the pair's queue with the given agent name. The queue ranks agents by
# gpu_priority.txt in that pair's QBASE; PairUpdate is ranked so an update lands right after the
# job that is running now, ahead of every other waiter.
pair_enqueue() {  # <agent> <max_minutes> <cmd...>
  local agent=${1:?agent} mins=${2:?minutes}; shift 2
  # A ticket with no command would hold the GPU for zero seconds and report success; the queue
  # would look like it ran the update. Refuse instead -- and the caller must be the one that dies.
  [ "$#" -gt 0 ] || pair_die "pair_enqueue $agent: no command given (nothing would run)"
  local run q logdir log pid t i before after
  run=$(pair_run_script)
  [ -x "$run" ] || pair_die "queue runner missing: $run"
  q=$PAIR_QUEUE/$PAIR/gpu_queue
  logdir=$PAIR_STATE_DIR/logs
  mkdir -p "$logdir" "$q"
  log=$logdir/enqueue-$PAIR-$agent-$(date -u +%Y%m%dT%H%M%SZ).log
  # What is already queued, so the new ticket is found by difference and not by guessing its name.
  before=$(ls "$q" 2>/dev/null)
  # setsid: the ticket wrapper and everything it starts form their own process group, so cancelling
  # the job kills the whole group (the queue does kill -- -<pgid>) and nothing outlives its window.
  setsid nohup "$run" "$agent" "$mins" "$@" >"$log" 2>&1 </dev/null &
  pid=$!
  # Wait for the ticket so the caller reports the one it actually got. Match on the agent suffix and
  # on "not there before", never on $pid: the runner names the ticket with its own $$ and setsid(1)
  # may fork, so the name's pid is not $! -- matching it reported ENQUEUE_FAILED for tickets that
  # were queued and did run (2026-11-01, a probe ran while the caller was told nothing had).
  t=""
  for i in $(seq 1 40); do
    after=$(ls "$q" 2>/dev/null)
    t=$(comm -13 <(printf '%s\n' "$before" | sort) <(printf '%s\n' "$after" | sort) 2>/dev/null \
          | awk -v s="-$agent" 'length($0) > length(s) && substr($0, length($0) - length(s) + 1) == s' \
          | head -1)
    [ -n "$t" ] && break
    sleep 0.25
  done
  if [ -z "$t" ]; then
    echo "ENQUEUE_FAILED $PAIR agent=$agent pid=$pid log=$log"
    tail -5 "$log" 2>/dev/null
    return 1
  fi
  echo "ENQUEUED $PAIR agent=$agent pid=$pid ticket=$t log=$log"
}
# PairUpdate must land right after the job that is running now, ahead of every other waiter: the
# queue ranks agents by their line in gpu_priority.txt, so line 1 is enough. Tickets already taken
# keep the rank they were created with, so existing waiters are not reordered.
pair_ensure_priority() {  # <agent>
  local agent=${1:?agent} prio=$PAIR_QUEUE/$PAIR/gpu_priority.txt tmp
  mkdir -p "$(dirname "$prio")"
  tmp=$(mktemp "${prio}.XXXXXX") || pair_die "cannot write $prio"
  { echo "$agent"; [ -r "$prio" ] && grep -vx "$agent" "$prio" | grep -v '^$'; } > "$tmp"
  mv -f "$tmp" "$prio"
}
