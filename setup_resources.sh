#!/usr/bin/env bash
# Fetch the two archives the reconstruction needs: the decoded brain features and
# the VQGAN weights. About 2.7 GB downloaded, 4.5 GB on disk once extracted.
#
# Both steps verify a sha256 and skip work that is already done, so re-running
# this after an interrupted download is safe.
#
# It does NOT fetch the DreamSim weights (~3.8 GB): those are pulled by the
# dreamsim package itself, into ./models/, the first time an evaluation script
# runs. See the README.
set -euo pipefail

# Resolve repository root from this script's location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

run_download_brain_features() {
  echo "[1/2] Downloading brain features..."
  python3 "$REPO_ROOT/download_brain_features.py"
}

run_download_vqgan_model() {
  echo "[2/2] Downloading VQGAN model (this may take a while)..."
  bash "$REPO_ROOT/download_vqgan_model.sh"
}

main() {
  run_download_brain_features
  run_download_vqgan_model
  echo "All assets prepared."
}

main "$@"
