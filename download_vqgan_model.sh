#!/usr/bin/env bash
# Fetch the VQGAN (ImageNet f16, 1024 codebook) checkpoint and its config.
#
# Both files come from the CompVis release that Koide-Majima et al. used, and land
# under the taming-transformers submodule where config_recon.yaml expects them:
#
#   lib/taming-transformers/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt
#   lib/taming-transformers/logs/vqgan_imagenet_f16_1024/configs/model.yaml
#
# Each download is checked against the sha256 of the copy these analyses ran on.
# Re-running is cheap: a file that already verifies is left alone.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/lib/taming-transformers/logs/vqgan_imagenet_f16_1024"

CKPT="$LOG_DIR/checkpoints/last.ckpt"
CKPT_URL='https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1'
CKPT_SHA256='e38a3aed8c86e8d47436f611325b3a23888cab70a0cc56c13cd84e8d4aab6e71'

CONFIG="$LOG_DIR/configs/model.yaml"
CONFIG_URL='https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1'
CONFIG_SHA256='cbd6fa371d07bf5bb960e32186e6e81356accbefdcc722bf94583b6a98bda00c'

verify() {  # verify <path> <expected sha256>; true when the file is already correct
  [ -f "$1" ] && [ "$(sha256sum "$1" | cut -d' ' -f1)" = "$2" ]
}

fetch() {  # fetch <path> <url> <expected sha256>
  local path="$1" url="$2" want="$3"
  if verify "$path" "$want"; then
    echo "  already present and verified: $path"
    return
  fi
  echo "  downloading $(basename "$path") ..."
  mkdir -p "$(dirname "$path")"
  # Download beside the target so a failed transfer cannot masquerade as a
  # complete file on the next run.
  wget --progress=dot:giga "$url" -O "$path.part"
  local got
  got="$(sha256sum "$path.part" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    rm -f "$path.part"
    echo "  sha256 mismatch for $path" >&2
    echo "    expected $want" >&2
    echo "    got      $got" >&2
    exit 1
  fi
  mv "$path.part" "$path"
  echo "  ok: $path"
}

if [ ! -d "$SCRIPT_DIR/lib/taming-transformers" ]; then
  echo "lib/taming-transformers is empty -- fetch the submodules first:" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

fetch "$CKPT" "$CKPT_URL" "$CKPT_SHA256"
fetch "$CONFIG" "$CONFIG_URL" "$CONFIG_SHA256"
