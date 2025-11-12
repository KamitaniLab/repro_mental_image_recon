"""Generate reconstruction variability panel for Figure 2D."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from fig_utils import GroupImageDrawer

from figure_asset_utils import (
    ensure_directory,
    load_recon_images,
    load_target_images,
    project_root,
)

PROJECT_ROOT = project_root()
RECON_ROOT_CANDIDATES = (
    PROJECT_ROOT / "results" / "rep_recon_image_koide-majima_recon_variability_no_seed_latest",
    PROJECT_ROOT / "results" / "rep_recon_image_koide-majima_recon_variability_no_seed",
)
OUTPUT_PATH = ensure_directory(PROJECT_ROOT / "assets" / "fig02") / "Fig2D_recon_image_variable.pdf"

CONDITION_KEY = "original_all"
SUBJECT_ID = "S2"
STIMULUS_NAME_LIST = ('imageryExpStim20_anat_leopard.tiff',
                'imageryExpStim22_inat_airliner.tiff')

ITERATIONS = tuple(f"iter{idx:02}" for idx in range(4, 8))


def _resolve_recon_root() -> Path:
    for candidate in RECON_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No reconstruction variability directory found. Checked: "
        + ", ".join(str(path) for path in RECON_ROOT_CANDIDATES)
    )


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


def generate_variability_panel() -> None:
    recon_root = _resolve_recon_root()
    
    for STIMULUS_NAME in STIMULUS_NAME_LIST:
        target_image = load_target_images([STIMULUS_NAME])[0]
        recon_images = _load_iteration_recons(recon_root, SUBJECT_ID, STIMULUS_NAME, ITERATIONS)

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
        panel.save(OUTPUT_PATH)


def main() -> None:
    generate_variability_panel()


if __name__ == "__main__":
    main()
