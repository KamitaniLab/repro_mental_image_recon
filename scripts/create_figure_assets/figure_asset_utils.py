"""Common helpers for generating figure assets from reconstruction notebooks.

These utilities centralize path resolution and data loading logic that was
previously duplicated across several notebooks. The goal is to make the
refactored scripts concise and easier to maintain for publication.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

# Canonical ordering of the stimulus images used throughout the figures.
#
# The order is NOT the numeric order of the file names, and that is deliberate: it
# is the order in which the reconstructions come out of `sorted()`. Reconstructions
# are written as `recon_img_normalized-Img{NNNN}.jpg`, where NNNN is the imageryExpStim
# number, while the stimulus that produced Img{NNNN} is `target_labels[targetID]` from
# the mental_img_recon submodule -- and that list orders the natural images by ImageNet
# id, not by stimulus number. The resulting pairing is
#     Img0017 -> Stim18 goldfish   Img0020 -> Stim20 leopard
#     Img0018 -> Stim19 iguana     Img0021 -> Stim17 goat
#     Img0019 -> Stim21 swan       Img0022..0026 -> Stim22..26
# so listing the names in this order makes index i here match index i of the sorted
# reconstruction files, which is what `recon_lookup` relies on. Keep them in sync.
SOURCE_IMAGE_NAMES: tuple[str, ...] = (
    "imageryExpStim01_red_smallring.tiff",
    "imageryExpStim02_red_+.tiff",
    "imageryExpStim03_red_X.tiff",
    "imageryExpStim04_green_smallring.tiff",
    "imageryExpStim05_green_+.tiff",
    "imageryExpStim06_green_X.tiff",
    "imageryExpStim07_blue_smallring.tiff",
    "imageryExpStim08_blue_+.tiff",
    "imageryExpStim09_blue_X.tiff",
    "imageryExpStim10_white_smallring.tiff",
    "imageryExpStim11_white_+.tiff",
    "imageryExpStim12_white_X.tiff",
    "imageryExpStim13_black_smallring.tiff",
    "imageryExpStim14_black_+.tiff",
    "imageryExpStim15_black_X.tiff",
    "imageryExpStim18_anat_goldfish.tiff",
    "imageryExpStim19_anat_iguana.tiff",
    "imageryExpStim21_anat_swan.tiff",
    "imageryExpStim20_anat_leopard.tiff",
    "imageryExpStim17_anat_goat.tiff",
    "imageryExpStim22_inat_airliner.tiff",
    "imageryExpStim23_inat_bowling.tiff",
    "imageryExpStim24_inat_post.tiff",
    "imageryExpStim25_inat_stainedglass.tiff",
    "imageryExpStim26_inat_umbrella.tiff",
)

SUBJECTS: tuple[str, ...] = ("S1", "S2", "S3")

# Lazily evaluated to avoid surprises when the module is imported outside the repo.
_PROJECT_ROOT: Path | None = None


def project_root() -> Path:
    """Return the repository root inferred from this file location."""
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    return _PROJECT_ROOT


def resolve_data_dir(env_var: str = "IMAGERY_SOURCE_DIR") -> Path:
    """Resolve the directory that stores stimulus images.

    An absolute directory can be provided via ``env_var``. When the variable is
    not set, the default is ``project_root() / "data/source"``.
    """
    env_override = os.getenv(env_var)
    if env_override:
        return Path(env_override).expanduser().resolve()
    return project_root() / "data/source"


def ensure_directory(path: Path) -> Path:
    """Create ``path`` if needed and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_image(path: Path) -> Image.Image:
    """Load ``path`` as an RGB ``Image`` while closing the file handle."""
    with Image.open(path) as img:
        return img.convert("RGB")


def load_target_images(
    image_names: Sequence[str] = SOURCE_IMAGE_NAMES,
    base_dir: Path | None = None,
) -> list[Image.Image]:
    """Load the ordered list of target stimuli as ``Image`` objects."""
    base = Path(base_dir) if base_dir else resolve_data_dir()
    images: list[Image.Image] = []
    for name in image_names:
        path = base / name
        if not path.exists():
            raise FileNotFoundError(f"Missing stimulus image: {path}")
        images.append(_load_image(path))
    return images


def collect_image_paths(
    image_names: Sequence[str],
    base_dir: Path | None = None,
) -> list[Path]:
    """Return resolved paths for ``image_names`` relative to ``base_dir``."""
    base = Path(base_dir) if base_dir else resolve_data_dir()
    paths: list[Path] = []
    for name in image_names:
        path = (base / name).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing stimulus image: {path}")
        paths.append(path)
    return paths


def recon_lookup(recon_dir: Path, expected_names: Sequence[str]) -> dict[str, Path]:
    """Map each stimulus name in ``expected_names`` to its reconstruction path.

    Two matching strategies, in order:

    1. by stem -- used when the reconstructions are named after the stimulus (e.g.
       ``imageryExpStim01_red_smallring.jpg``);
    2. by position -- the reconstruction pipeline names its output by stimulus index
       (``recon_img_normalized-Img0017.jpg``) rather than by stem, so this is the
       usual path. ``SOURCE_IMAGE_NAMES`` is ordered to match the sorted
       reconstruction file names (see its definition), making index i of one the
       index i of the other.

    Positional matching is only valid for a complete directory: a partial run would
    shift every stimulus after the gap and silently produce a figure whose targets
    and reconstructions do not correspond. It therefore requires the full set and
    raises otherwise.
    """
    recon_dir = Path(recon_dir)
    candidates = list(recon_dir.glob("*.jpg")) + list(recon_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No reconstructions found under {recon_dir}")

    stem_to_path: dict[str, Path] = {}
    for path in candidates:
        stem = path.stem
        if stem in stem_to_path:
            raise ValueError(
                f"Multiple reconstruction files share stem '{stem}' in {recon_dir}"
            )
        stem_to_path[stem] = path

    lookup: dict[str, Path] = {}
    missing: list[str] = []
    for name in expected_names:
        stem = Path(name).stem
        if stem not in stem_to_path:
            missing.append(name)
            continue
        lookup[name] = stem_to_path[stem]

    if missing:
        sorted_candidates = sorted(candidates, key=lambda path: path.name)
        if len(sorted_candidates) != len(SOURCE_IMAGE_NAMES):
            raise FileNotFoundError(
                f"Cannot match reconstructions in {recon_dir} by name "
                f"(no file named after {', '.join(missing)}), and matching by position "
                f"needs the complete set of {len(SOURCE_IMAGE_NAMES)} reconstructions "
                f"but found {len(sorted_candidates)}. Re-run the reconstruction for "
                f"this condition before drawing the figure."
            )
        canonical_map = dict(zip(SOURCE_IMAGE_NAMES, sorted_candidates))

        unresolved = [name for name in expected_names if name not in canonical_map]
        if unresolved:
            raise FileNotFoundError(
                "Missing reconstructions for stimuli: " + ", ".join(unresolved)
            )
        return {name: canonical_map[name] for name in expected_names}
    return lookup


def load_recon_images(
    recon_dir: Path,
    image_names: Sequence[str],
) -> list[Image.Image]:
    """Load the reconstructions corresponding to ``image_names``."""
    mapping = recon_lookup(recon_dir, image_names)
    return [_load_image(mapping[name]) for name in image_names]


def subset_by_names(
    names: Sequence[str],
    selection: Iterable[str],
) -> list[str]:
    """Return items from ``names`` that appear in ``selection`` preserving order."""
    selected = set(selection)
    return [name for name in names if name in selected]
