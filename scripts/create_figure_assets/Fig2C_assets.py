"""Generate figure assets for Figures 2C and A1 (target vs reconstructed images).

Both panels come from the same reconstructions and differ only in how many stimuli
they show: Figure 2C the four stimuli cleared for publication, Figure A1 all 25.
Running this regenerates both under ``assets/fig02``.
"""

from __future__ import annotations

from pathlib import Path

from repro_mental_image_recon.figures.assets import (
    SOURCE_IMAGE_NAMES,
    SUBJECTS,
    ensure_directory,
    load_recon_images,
    load_target_images,
    project_root,
)
from repro_mental_image_recon.figures.drawing import GroupImageDrawer

# Directories
PROJECT_ROOT = project_root()
RECON_ROOT = PROJECT_ROOT / "results" / "rep_recon_image_koide-majima"
OUTPUT_DIR = ensure_directory(PROJECT_ROOT / "assets" / "fig02")

# Dataset metadata
CONDITION_KEY = "original_all"
# The four stimuli cleared for publication; the full set is shown in Figure A1,
# which is only reproducible locally by whoever holds the stimulus images.
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
    panel.save(OUTPUT_DIR / "FigA1_recon_image_all.pdf")


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
