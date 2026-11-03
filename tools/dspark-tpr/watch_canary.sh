#!/bin/bash
# Wait (read-only) until the dev box goes quiet: no ticket holds it and no log is growing. Then print the
# last verdict lines so the next step (queueing ab_k35) is decided on the canary's actual outcome.
Q=/home/treadall/recipe-artifacts/gpu0-gx1/queue/dev
for i in $(seq 1 135); do
  run=$(cat "$Q/../running" 2>/dev/null; cat "$Q/running" 2>/dev/null)
  newest=$(ls -t "$Q"/logs 2>/dev/null | head -1)
  age=$(( $(date +%s) - $(stat -c %Y "$Q/logs/$newest" 2>/dev/null || echo 0) ))
  if [ -z "$run" ] && [ "$age" -ge 90 ]; then
    echo "BOX_QUIET waited_s=$((i*20)) newest=$newest log_age_s=$age"
    tail -25 "$Q/logs/$newest"
    exit 0
  fi
  sleep 20
done
echo "BOX_BUSY after 45min"; exit 1
