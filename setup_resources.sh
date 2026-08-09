#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

run_download_brain_features() {
  echo "[1/3] Downloading brain features..."
  python3 "$REPO_ROOT/download_brain_features.py"
}

run_download_vqgan_model() {
  echo "[2/3] Downloading VQGAN model (this may take a while)..."
  bash "$REPO_ROOT/download_vqgan_model.sh"
}

main() {
  run_download_brain_features
  run_download_vqgan_model
  echo "All assets prepared."
}

main "$@"
