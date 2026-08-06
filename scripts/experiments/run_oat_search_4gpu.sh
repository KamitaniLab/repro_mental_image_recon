#!/usr/bin/env bash
# Launch the one-at-a-time (OAT) sampling-parameter sweep split across N GPUs.
#
# OAT = 22 combos (baseline original_all + vary one parameter at a time), default
# over ALL 25 imagery stimuli x 3 subjects = 1,650 reconstructions. Sharded by
# combo index (index % NUM_SHARDS == shard_id); each shard pinned to one GPU via
# CUDA_VISIBLE_DEVICES. Separate output root from the full-grid run.
#
# Usage:
#   bash scripts/experiments/run_oat_search_4gpu.sh              # 4 GPUs (0,1,2,3)
#   bash scripts/experiments/run_oat_search_4gpu.sh 2            # first 2 GPUs
#   OUT=./results/my_oat bash scripts/experiments/run_oat_search_4gpu.sh
#   bash scripts/experiments/run_oat_search_4gpu.sh 4 --targetID 0 7 15 17 23   # pass-through args
set -euo pipefail

NUM_GPUS="${1:-4}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then shift; fi

OUT="${OUT:-./results/oat_sampling_params}"
SCRIPT="scripts/experiments/oat_search_SGD_SGLD_sampling_params.py"

mkdir -p "$OUT"
echo "Launching OAT sweep on ${NUM_GPUS} GPU(s), out=${OUT}"

pids=()
for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
  CUDA_VISIBLE_DEVICES="$gpu" \
    uv run python "$SCRIPT" \
      --num_shards "$NUM_GPUS" --shard_id "$gpu" \
      --out "$OUT" --resume \
      "$@" \
      > "${OUT}/shard_${gpu}.log" 2>&1 &
  pids+=($!)
  echo "  shard ${gpu} -> GPU ${gpu} (pid ${pids[-1]}, log ${OUT}/shard_${gpu}.log)"
done

fail=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "shard pid ${pid} failed" >&2
    fail=1
  fi
done

echo "All shards finished (fail=${fail})."
exit "$fail"
