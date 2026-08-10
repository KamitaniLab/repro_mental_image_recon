#!/usr/bin/env bash
# Launch a sampling-parameter sweep split across N GPUs.
#
# MODE=oat   : 22 combos, vary one parameter at a time around the released setting
# MODE=slice : 25 combos, the 5x5 lr_a x T plane
# Figure A5 uses the union of the two (38 unique settings), so run both. Each mode
# covers ALL 25 imagery stimuli x 3 subjects. Sharded by combo index
# (index % NUM_SHARDS == shard_id); each shard is pinned to one GPU via
# CUDA_VISIBLE_DEVICES, and each mode writes to its own output root.
#
# Usage:
#   bash scripts/experiments/run_oat_search_4gpu.sh              # OAT on 4 GPUs
#   MODE=slice bash scripts/experiments/run_oat_search_4gpu.sh   # the lr_a x T plane
#   bash scripts/experiments/run_oat_search_4gpu.sh 2            # first 2 GPUs
#   OUT=./results/my_oat bash scripts/experiments/run_oat_search_4gpu.sh
#   bash scripts/experiments/run_oat_search_4gpu.sh 4 --targetID 0 7 15 17 23   # pass-through args
set -euo pipefail

NUM_GPUS="${1:-4}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then shift; fi

MODE="${MODE:-oat}"
case "$MODE" in
  oat)   DEFAULT_OUT="./results/oat_sampling_params" ;;
  slice) DEFAULT_OUT="./results/lr_a_T_slice" ;;
  *)     echo "unknown MODE=$MODE (expected oat or slice)" >&2; exit 2 ;;
esac
OUT="${OUT:-$DEFAULT_OUT}"
SCRIPT="scripts/experiments/oat_search_SGD_SGLD_sampling_params.py"

mkdir -p "$OUT"
echo "Launching ${MODE} sweep on ${NUM_GPUS} GPU(s), out=${OUT}"

pids=()
for (( gpu=0; gpu<NUM_GPUS; gpu++ )); do
  CUDA_VISIBLE_DEVICES="$gpu" \
    uv run python "$SCRIPT" \
      --mode "$MODE" \
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
