#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Run this through `uv run bash`, as the README does, or with the venv activated:
# a bare `python` is whatever is on PATH. Override with PYTHON=... if neither fits.
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
