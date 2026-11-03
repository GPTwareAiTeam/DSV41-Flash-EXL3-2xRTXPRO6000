#!/bin/bash
# gate_gx_remote.sh -- the GX2 half of gx_gate, run there as `ssh gx10-2 "CT=… bash -s" < gate_gx_remote.sh`.
# Everything the gate needs, printed as `key=value` lines; the host side parses. No docker stdout is mixed in:
# every docker call has stdin detached and stderr dropped.
CT=${CT:-seam-gx2-dev}
awk '/^MemAvailable|^SwapTotal|^SwapFree/{k=$1; sub(/:$/,"",k); printf "%s=%s ", k, $2}' /proc/meminfo; echo
# The worker's argv differs by how it was started: `python3 -u qmesh/worker_seam_server.py` from the source
# dir, or `python3 -u /pair/current/src/qmesh/worker_seam_server.py` from a release. An anchor that spells
# only the first misses the second and reports the server as down (or, worse, reports VmSwap 0 for it), so
# the path is allowed to be anything and the argv is printed for the load shape.
wp=$(pgrep -f '^python3 -u [^ ]*worker_seam_server[.]py' | head -1)
echo "worker_pid=${wp:-}"
if [ -n "${wp:-}" ]; then
  awk '/^VmSwap/{printf "worker_vmswap_kb=%s\n", $2} /^VmRSS/{printf "worker_vmrss_kb=%s\n", $2}' /proc/$wp/status
  echo "worker_argv=$(tr '\0' ' ' < /proc/$wp/cmdline 2>/dev/null)"
  # The worker's own clock, not the container's: pair_serve restarts the worker inside the running
  # container, so StartedAt can be an hour older than the server that is actually serving.
  echo "worker_age_s=$(ps -o etimes= -p $wp 2>/dev/null | tr -d ' ')"
else
  # A server process the anchor does not match is a gate bug, not a down server: say which it is. The
  # bracket is load-bearing -- this runs inside `bash -lc`, whose own argv contains the pattern, so
  # `pgrep -fc worker_seam_server` reports 1 on an empty container.
  echo "worker_unanchored=$(docker exec "$CT" pgrep -f '[w]orker_seam_server[.]py' </dev/null 2>/dev/null | wc -l)"
fi
ct_start=$(docker inspect -f '{{.State.StartedAt}}' "$CT" 2>/dev/null)
if [ -n "$ct_start" ]; then
  echo "ct_age_s=$(( $(date +%s) - $(date -d "$ct_start" +%s 2>/dev/null || date +%s) ))"
fi
docker exec "$CT" bash -lc 'grep -a "SEAM_SERVER_SLAB " /tmp/seam_server.log | tail -1' </dev/null 2>/dev/null
echo "slab_warn=$(docker exec "$CT" bash -lc 'grep -ac "SEAM_SERVER_SLAB_WARN" /tmp/seam_server.log' </dev/null 2>/dev/null | head -1)"
echo "box_warn=$(docker exec "$CT" bash -lc 'grep -acE "SEAM_SERVER_SWAP_WARN|SEAM_SLACK_WARN" /tmp/seam_server.log' </dev/null 2>/dev/null | head -1)"
echo "warn_last=$(docker exec "$CT" bash -lc 'tail -n 400 /tmp/seam_server.log' </dev/null 2>/dev/null | grep -aoE 'SEAM_SERVER_SLAB_WARN|SEAM_SERVER_SWAP_WARN|SEAM_SLACK_WARN' | tr '\n' ',')"
# The autotune pick file decides GEMM tiles, and its absence means every server start re-derives them.
# Resolved the way the loader resolves it -- EXLLAMAV3_TUNE_CACHE names a directory (append the file) or
# a file verbatim, else $XDG_CACHE_HOME/exllamav3/autotune, else $HOME/.cache/... -- so the probe says
# "absent" only when the server would really find nothing. Path, size and mtime are printed because
# presence alone does not mean the picks belong to this build: a version bump re-tunes silently. No
# python on purpose: a probe that imported the extension would create the file it is probing.
# The env that decides the load shape and the tune file is the serving worker's own: pair_serve starts it
# inside the container with the pair's env, and the host cannot read /proc/<pid>/environ of a container
# process (root-owned), so the probe runs inside the container and finds the worker by the container pid.
#   SEAM_WS_ROWS/_AUTO   pair_lib pins the width with the picker off, so 1024 is pinned, never chosen.
#   EXLLAMAV3_TUNE_CACHE resolved the way the loader does -- a directory gets the file appended, a file is
#                        used verbatim, else $XDG_CACHE_HOME/exllamav3/autotune, else $HOME/.cache/... --
#                        and printed with size and mtime because presence does not mean the picks belong
#                        to this build: a COOP_AUTOTUNE_VERSION bump re-tunes silently.
probe=$(cat <<'EOS'
cpid=$(pgrep -f '^python3 -u [^ ]*worker_seam_server[.]py' | head -1)
if [ -z "$cpid" ]; then echo "worker_ws_env=?"; echo "tune=unresolved no_worker"; exit 0; fi
env_of() { tr '\0' '\n' < /proc/$cpid/environ 2>/dev/null; }
echo "worker_ct_pid=$cpid"
# The width and the kernel family both come from the server's env, not from the client's argv: pair_serve
# composes SEAM_MOE_KERNEL/SEAM_MOE_PRELOAD via pair_server_env, so a served arm cannot see them from the
# pair_arm command line. Reading the worker's environ is the only place they are visible.
# What the worker was TOLD is not what it RAN: pair_lib sets SEAM_MOE_KERNEL=cudaexl3, but a missing or
# wrong-arch vendored extension makes every layer fall back to the mixed-K module and only the log says so
# (SEAM_MOE_NOEXT). The pack line carries the split, so the served family is quoted from the log.
pk=$(grep -a 'SEAM_PACK_KERNEL' /tmp/seam_server.log 2>/dev/null | tail -1)
if [ -n "$pk" ]; then
  cl=$(printf '%s' "$pk" | sed -n 's/.*cudaexl3_layers=\([0-9]*\).*/\1/p')
  ml=$(printf '%s' "$pk" | sed -n 's/.*mixedk_layers=\([0-9]*\).*/\1/p')
  echo "kernel_layers=cudaexl3:${cl:-?} mixedk:${ml:-?}"
else echo "kernel_layers=?"; fi
ld=$(grep -an 'SEAM_SERVER_LOADED' /tmp/seam_server.log 2>/dev/null | tail -1 | cut -d: -f1)
echo "kernel_noext=$(awk -v s=${ld:-1} 'NR>=s && /SEAM_MOE_NOEXT/' /tmp/seam_server.log 2>/dev/null | wc -l)"
echo "worker_ws_env=$(env_of | grep -E '^(SEAM_WS_ROWS(_AUTO)?|SEAM_MOE_KERNEL|SEAM_MOE_PRELOAD)=' | tr '\n' ',')"
f=$(env_of | sed -n 's/^EXLLAMAV3_TUNE_CACHE=//p' | tail -1)
xdg=$(env_of | sed -n 's/^XDG_CACHE_HOME=//p' | tail -1)
if [ -n "$f" ]; then
  if [ -d "$f" ]; then p="$f/coop_autotune_v1.bin"; else p="$f"; fi
else
  p="${xdg:-/root/.cache}/exllamav3/autotune/coop_autotune_v1.bin"
fi
if [ -e "$p" ]; then echo "tune=path=$p size_mtime=$(stat -c '%s %y' "$p" | tr ' ' '_')"; else echo "tune=absent path=$p"; fi
echo "tune_env=${f:-unset} xdg=${xdg:-unset}"
grep -rhoE 'COOP_AUTOTUNE_VERSION[^0-9]{1,8}[0-9]+' /src/exllamav3/ext /pair/current/src 2>/dev/null | head -1 | tr ' ' '_'
EOS
)
# Line-preserving: the host parser keys on `^tune=` / `^worker_ws_env=`, so joining these with spaces
# would hide them behind another key on the same line.
printf '%s\n' "$probe" | docker exec -i "$CT" bash -s 2>/dev/null
exit 0
