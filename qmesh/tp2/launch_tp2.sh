#!/usr/bin/env bash
# Launch the 2-rank TP runner in container exl3-tp2 (one process per GPU, NCCL over file:// rendezvous).
# Usage: qmesh/tp2/launch_tp2.sh <tag> [-e KEY=VAL ...]
# Logs: /artifacts/tp2logs/<tag>_rank{0,1}.log (host: /home/treadall/recipe-artifacts/2x-rtx-tp2/tp2logs)
set -euo pipefail
TAG="${1:?tag}"; shift
EXTRA=("$@")
RV="/artifacts/tp2logs/${TAG}_rendezvous"
rm -f "$RV"
P2P_RDV="/artifacts/tp2logs/${TAG}_rdv"
docker exec exl3-tp2 rm -rf "$P2P_RDV"
for r in 0 1; do
  docker exec -d -e RANK=$r -e WORLD_SIZE=2 -e INIT_METHOD="file://$RV" -e P2P_RDV="$P2P_RDV" \
    -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    -e TOKENS_OUT="/artifacts/tp2logs/${TAG}_tokens" "${EXTRA[@]}" \
    exl3-tp2 bash -c "cd /src/exllamav3 && python3 -m qmesh.tp2.run_tp2 > /artifacts/tp2logs/${TAG}_rank${r}.log 2>&1"
done
echo "launched $TAG"
