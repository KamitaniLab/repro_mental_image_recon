"""Generate figure assets for Figure 2C (target vs reconstructed images).

This script consolidates the logic that previously lived in the Jupyter
notebook. Running it will regenerate the publication-ready panels stored under
``assets/fig02``.
"""
from __future__ import annotations

from pathlib import Path

from fig_utils import GroupImageDrawer

from figure_asset_utils import (
    SOURCE_IMAGE_NAMES,
    SUBJECTS,
    ensure_directory,
    load_recon_images,
    load_target_images,
    project_root,
)

# Directories
PROJECT_ROOT = project_root()
RECON_ROOT = PROJECT_ROOT / "results" / "rep_recon_image_koide-majima"
OUTPUT_DIR = ensure_directory(PROJECT_ROOT / "assets" / "fig02")

# Dataset metadata
CONDITION_KEY = "original_all"
COMPARISON_LABEL = "Koide-Majima"
RANDOM_SELECTION = (
    "imageryExpStim18_anat_goldfish.tiff",
    "imageryExpStim21_anat_swan.tiff",
    "imageryExpStim24_inat_post.tiff",
    "imageryExpStim25_inat_stainedglass.tiff",
)


def _subject_recon_dir(subject: str) -> Path:
    return RECON_ROOT / CONDITION_KEY / subject / "VC"


def _load_subject_recon(subject: str, image_names: tuple[str, ...] | list[str]) -> list:
    recon_dir = _subject_recon_dir(subject)
    return load_recon_images(recon_dir, image_names)


def generate_full_panel() -> None:
    """Target vs. reconstruction panel across all stimuli and subjects."""
    target_images = load_target_images()
    conditions = [{"title": "Target", "images": target_images}]

    for subject in SUBJECTS:
        recon_images = _load_subject_recon(subject, SOURCE_IMAGE_NAMES)
        conditions.append({"title": subject, "images": recon_images})

    drawer = GroupImageDrawer(
        conditions,
        title_fontcolor="black",
        title_fontsize=12,
        max_column_size=15,
    )
    panel = drawer.draw()
    panel.save(OUTPUT_DIR / "FigS1_recon_image_all.pdf")


def generate_random_subset_panel() -> None:
    """Target vs. reconstruction panel for the copyright-safe subset."""
    target_images = load_target_images(RANDOM_SELECTION)
    conditions = [{"title": "Target", "images": target_images}]

    for subject in SUBJECTS:
        recon_images = _load_subject_recon(subject, RANDOM_SELECTION)
        conditions.append({"title": subject, "images": recon_images})

    drawer = GroupImageDrawer(
        conditions,
        title_fontcolor="black",
        title_fontsize=12,
        max_column_size=len(RANDOM_SELECTION),
    )
    panel = drawer.draw()
    panel.save(OUTPUT_DIR / "Fig2C_recon_image_random.pdf")


def main() -> None:
    generate_full_panel()
    generate_random_subset_panel()


if __name__ == "__main__":
    main()
