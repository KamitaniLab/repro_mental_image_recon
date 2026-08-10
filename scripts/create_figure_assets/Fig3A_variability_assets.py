"""Generate reconstruction variability panel for Figure 3A.

Four of the 10 runs, selected at random (seed-fixed for reproducibility), are displayed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np

from repro_mental_image_recon.figures.drawing import GroupImageDrawer

from repro_mental_image_recon.figures.assets import (
    ensure_directory,
    load_recon_images,
    load_target_images,
    project_root,
)

PROJECT_ROOT = project_root()
RECON_ROOT = (
    PROJECT_ROOT / "results" / "rep_recon_image_koide-majima_recon_variability_no_seed"
)
OUTPUT_DIR = PROJECT_ROOT / "assets" / "fig03"

CONDITION_KEY = "original_all"
SUBJECT_ID = "S2"
STIMULUS_NAME_LIST = (
    "imageryExpStim20_anat_leopard.tiff",
    "imageryExpStim22_inat_airliner.tiff",
)

# Four runs selected at random; seeded for reproducibility
_rng = np.random.RandomState(42)
_selected = sorted(_rng.choice(range(1, 11), size=4, replace=False))
ITERATIONS = tuple(f"iter{idx:02d}" for idx in _selected)


def _short_name(stimulus_name: str) -> str:
    """imageryExpStim20_anat_leopard.tiff -> 'leopard' (file-name friendly)."""
    parts = Path(stimulus_name).stem.split("_")
    return "_".join(p for p in parts[1:] if p not in ("anat", "inat")) or parts[0]


def _check_recon_root(recon_root: Path) -> Path:
    if not recon_root.exists():
        raise FileNotFoundError(
            f"No reconstruction variability directory at {recon_root}. Run "
            "scripts/experiments/recon_image_koide-majima_methods_multi_times_no_seed.py "
            "first, or pass --recon-root."
        )
    return recon_root


def _load_iteration_recons(
    recon_root: Path,
    subject: str,
    stimulus_name: str,
    iterations: Sequence[str],
) -> list:
    recon_images = []
    for iteration in iterations:
        recon_dir = recon_root / CONDITION_KEY / subject / iteration / "VC"
        images = load_recon_images(recon_dir, [stimulus_name])
        recon_images.append(images[0])
    return recon_images


def generate_variability_panel(recon_root: Path, output_dir: Path) -> None:
    recon_root = _check_recon_root(recon_root)
    output_dir = ensure_directory(output_dir)

    # One panel per stimulus: they share a file name until the stimulus is part of
    # it, and the loop would otherwise leave only the last one on disk.
    for stimulus_name in STIMULUS_NAME_LIST:
        target_image = load_target_images([stimulus_name])[0]
        recon_images = _load_iteration_recons(
            recon_root, SUBJECT_ID, stimulus_name, ITERATIONS
        )

        target_row = [target_image] * len(ITERATIONS)
        conditions = [
            {"title": "Target", "images": target_row},
            {"title": "Reconstruction", "images": recon_images},
        ]

        drawer = GroupImageDrawer(
            conditions,
            title_fontcolor="black",
            title_fontsize=12,
            max_column_size=len(ITERATIONS),
            id_fontsize=14,
        )
        panel = drawer.draw()
        output_path = (
            output_dir / f"Fig3A_recon_image_variable_{_short_name(stimulus_name)}.pdf"
        )
        panel.save(output_path)
        print(f"saved {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recon-root",
        type=Path,
        default=RECON_ROOT,
        help="root of the seed-free repeat runs (<root>/<condition>/<subject>/iterNN/VC)",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    generate_variability_panel(args.recon_root, args.output_dir)


if __name__ == "__main__":
    main()
