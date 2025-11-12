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

extract_imagery_archive() {
  local archive="$REPO_ROOT/data/imageryExpStim.zip"
  if [[ ! -f "$archive" ]]; then
    echo "[3/3] imageryExpStim.zip not found at $archive; skipping extraction." >&2
    return 1
  fi

  if ! command -v unzip >/dev/null 2>&1; then
    echo "[3/3] unzip command not available. Please install unzip and rerun." >&2
    return 1
  fi

  echo "[3/3] Extracting imagery stimuli from $(realpath "$archive")"
  unzip -o "$archive" -d "$REPO_ROOT/data"
}

main() {
  run_download_brain_features
  run_download_vqgan_model
  extract_imagery_archive
  echo "All assets prepared."
}

main "$@"
