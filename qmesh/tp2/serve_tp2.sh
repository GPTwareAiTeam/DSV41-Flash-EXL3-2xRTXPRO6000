#!/usr/bin/env bash
# Launch or manage the two TP2 server ranks in an already running container.
set -euo pipefail

CONTAINER=${CONTAINER:-exl3-server}
REPO_DIR=${REPO_DIR:-/src/exllamav3}
STATE_DIR=${STATE_DIR:-/tmp/qmesh-tp2-server}
QMESH_CPUSET_RANK0=${QMESH_CPUSET_RANK0-2-5}
QMESH_CPUSET_RANK1=${QMESH_CPUSET_RANK1-10-13}
command=${1:-start}
if (($#)); then shift; fi

case "$command" in
  start)
    # Additional arguments are docker exec environment options (-e KEY=VALUE).
    extra=()
    while (($#)); do
      case "$1" in
        -e|--env)
          (($# >= 2)) || { echo "missing value for $1" >&2; exit 2; }
          extra+=("$1" "$2"); shift 2 ;;
        --env=*) extra+=("$1"); shift ;;
        *) echo "usage: $0 {start [-e KEY=VALUE ...]|stop|status}" >&2; exit 2 ;;
      esac
    done
    if docker exec "$CONTAINER" sh -c '
      for r in 0 1; do
        f="$1/rank$r.pid"
        if [ -r "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then exit 0; fi
      done
      exit 1
    ' sh "$STATE_DIR"; then
      echo "server already running in $CONTAINER; stop it first" >&2
      exit 1
    fi
    run_dir=$(docker exec "$CONTAINER" sh -c 'mkdir -p "$1" && mktemp -d "$1/run.XXXXXXXX"' sh "$STATE_DIR")
    env_args=(-e HOST="${HOST:-127.0.0.1}" -e PORT="${PORT:-8000}"
              -e DSPARK="${DSPARK:-1}" -e NUM_DRAFT="${NUM_DRAFT:-5}")
    # Forward exported runtime configuration when present. Explicit -e options win.
    while IFS='=' read -r name _; do
      case "$name" in
        OVERLAY|COMM|CTX_TOKENS|CACHE_TOKENS|CACHE_TOKENS_PER_SEQ|BATCH|MAX_BATCH|MAX_NEW|MODEL_ID|API_KEY|KV_BITS|MAX_CHUNK|DSPARK_*|EXL3_*|PREFILL_*|CHUNK_*|VRAM_RESERVE_MIB|TORCH_*|CUDA_VISIBLE_DEVICES|PYTHONPATH)
          env_args+=(-e "$name") ;;
      esac
    done < <(env)
    for rank in 0 1; do
      if ((rank == 0)); then cpuset=$QMESH_CPUSET_RANK0; else cpuset=$QMESH_CPUSET_RANK1; fi
      if ! docker exec -d -w "$REPO_DIR" "${env_args[@]}" "${extra[@]}" \
        -e RANK="$rank" -e WORLD_SIZE=2 -e INIT_METHOD="file://$run_dir/rendezvous" \
        -e P2P_RDV="$run_dir/p2p" "$CONTAINER" sh -c '
          mkdir -p "$1/p2p"
          echo $$ > "$2/rank$3.pid"
          if [ -n "$4" ]; then
            exec taskset -c "$4" python3 -m qmesh.tp2.serve_tp2 > "$1/rank$3.log" 2>&1
          else
            exec python3 -m qmesh.tp2.serve_tp2 > "$1/rank$3.log" 2>&1
          fi
        ' sh "$run_dir" "$STATE_DIR" "$rank" "$cpuset"; then
        echo "failed to launch rank $rank; stopping any started rank" >&2
        "$0" stop || true
        exit 1
      fi
    done
    alive=0
    for attempt in {1..10}; do
      sleep 1
      if docker exec "$CONTAINER" sh -c '
        for r in 0 1; do
          f="$1/rank$r.pid"
          [ -r "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null || exit 1
        done
      ' sh "$STATE_DIR"; then alive=1; break; fi
    done
    if ((!alive)); then
      echo "ranks exited during startup; log tails:" >&2
      docker exec "$CONTAINER" sh -c 'tail -n 20 "$1"/rank0.log "$1"/rank1.log' sh "$run_dir" >&2 || true
      exit 1
    fi
    echo "Started ranks in $CONTAINER; logs: $run_dir/rank{0,1}.log"
    ;;
  stop|status)
    (($# == 0)) || { echo "usage: $0 {start [-e KEY=VALUE ...]|stop|status}" >&2; exit 2; }
    docker exec "$CONTAINER" sh -c '
      running=0
      for r in 0 1; do
        f="$1/rank$r.pid"
        if [ -r "$f" ] && kill -0 "$(cat "$f")" 2>/dev/null; then
          pid=$(cat "$f")
          if [ "$2" = stop ] && [ "$r" = 0 ]; then
            kill "$pid"; echo "rank 0: stopping (pid $pid)"
          elif [ "$2" = stop ]; then
            echo "rank 1: awaiting coordinated shutdown (pid $pid)"
          else echo "rank $r: running (pid $pid)"; fi
          running=$((running + 1))
        else
          echo "rank $r: stopped"
        fi
      done
      if [ "$2" = status ] && [ "$running" -ne 2 ]; then exit 1; fi
    ' sh "$STATE_DIR" "$command"
    ;;
  *) echo "usage: $0 {start [-e KEY=VALUE ...]|stop|status}" >&2; exit 2 ;;
esac
