#!/bin/bash
# Queue the K-window A/B on the dev pair. Runs only after the canary: the box gate inside ab_k35.sh
# refuses a paging/loading worker, and the script itself refuses a worker older than 45 min.
set -u
PAIR_CHECK_FORCE=1 bash /home/treadall/recipe-artifacts/gpu0-gx1/queue/run_dev.sh DSparkTune 45 \
  bash /home/treadall/recipe-artifacts/dspark-tpr/ab_k35.sh "${1:-k35}" 2>&1 | tail -50
