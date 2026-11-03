#!/bin/bash
set -u
PAIR_CHECK_FORCE=1 bash /home/treadall/recipe-artifacts/gpu0-gx1/queue/run_dev.sh DSparkTune 20 \
  bash /home/treadall/recipe-artifacts/dspark-tpr/attn_run.sh 2>&1 | tail -45
