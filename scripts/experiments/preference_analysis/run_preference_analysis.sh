#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python}"
EXTRA_ARGS=("$@")

runs=(
  "feature cand2"
  "feature cand4"
  "dreamsim cand2"
  "dreamsim cand4"
  "lpips cand2"
  "lpips cand4"
)

for run in "${runs[@]}"; do
  read -r metric comparison <<<"$run"
  echo "[preference_analysis] metric=${metric} comparison=${comparison}"
  "$PYTHON_BIN" "$SCRIPT_DIR/run_preference_analysis.py" \
    --comparison "$comparison" \
    --metric "$metric" \
    "${EXTRA_ARGS[@]}"
done
