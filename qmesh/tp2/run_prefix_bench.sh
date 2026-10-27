#!/usr/bin/env bash
set -euo pipefail
tag=${1:?unique tag required}; shift
logs=/home/treadall/recipe-artifacts/2x-rtx-tp2/tp2logs
exec 9>/home/treadall/recipe-artifacts/2x-rtx-tp2/gpu.lock
flock 9
while :; do
  f0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  f1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  if [ "$f0" -gt 40000 ] && [ "$f1" -gt 40000 ]; then break; fi
  sleep 10
done
for r in 0 1; do
  docker exec -d -e RANK="$r" -e WORLD_SIZE=2 -e INIT_METHOD="file:///artifacts/tp2logs/${tag}_rendezvous" \
    -e P2P_RDV="/artifacts/tp2logs/${tag}_p2p" -e COMM=p2p -e TORCH_CUDA_ARCH_LIST=12.0a \
    -e EXL3_DSV4_B1_BATCH=1 -e EXL3_PREFIX_CACHE=1 "$@" exl3-tp2-pc \
    bash -c "cd /src/exllamav3 && python3 -m qmesh.tp2.bench_prefix_cache > /artifacts/tp2logs/${tag}_rank${r}.log 2>&1"
done
while :; do
  if ! docker exec exl3-tp2-pc pgrep -f '[q]mesh.tp2.bench_prefix_cache' >/dev/null; then break; fi
  sleep 5
done
printf 'Logs: %s/%s_rank{0,1}.log\n' "$logs" "$tag"
