"""Generate assets for Figure 5B and related supplementary panels."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import ImageChops

from fig_utils import GroupImageDrawer
from figure_asset_utils import (
    ensure_directory,
    load_recon_images,
    load_target_images,
    project_root,
)

PROJECT_ROOT = project_root()
RECON_ROOT = PROJECT_ROOT / "results" / "rep_recon_image_koide-majima_comparing_SGD_updated_sampling_parameters"
OUTPUT_DIR = ensure_directory(PROJECT_ROOT / "assets" / "fig05")

SUBJECT_ID = "S1"
BASE_CONDITION = "original_all"
RANDOM_POOL = (
    # copy right ok natural images
    "imageryExpStim01_red_smallring.tiff",
    #"imageryExpStim02_red_+.tiff",
    #"imageryExpStim04_green_smallring.tiff",
    #"imageryExpStim05_green_+.tiff",
    #"imageryExpStim07_blue_smallring.tiff",
    "imageryExpStim08_blue_+.tiff",
    #"imageryExpStim10_white_smallring.tiff",
    #"imageryExpStim11_white_+.tiff",
    "imageryExpStim18_anat_goldfish.tiff",
    "imageryExpStim21_anat_swan.tiff",
    #"imageryExpStim24_inat_post.tiff",
    "imageryExpStim25_inat_stainedglass.tiff",
    #"imageryExpStim26_inat_umbrella.tiff",
)
RANDOM_COUNT = 5
RANDOM_SEED = 42
IMAGE_SELECTION = tuple(sorted(np.random.default_rng(RANDOM_SEED).choice(RANDOM_POOL, size=RANDOM_COUNT, replace=False)))

SGLD_VARIANTS = (
    ("eps = 0.1", "original_all_fixed_values_SGLD_v3"),
    ("eps = 1.0", "original_all_fixed_values_SGLD_v4"),
    ("eps = 10", "original_all_fixed_values_SGLD_v5"),
)
SGLD_VARIANTS_NORMAL_TEMP = (
    ("eps = 0.1", "original_all_fixed_values_SGLD_v3_nomal_temp"),
    ("eps = 1.0", "original_all_fixed_values_SGLD_v4_normal_temp"),
    ("eps = 10", "original_all_fixed_values_SGLD_v5_normal_temp"),
)


def _recon_dir(condition_key: str, extra: str | None = None) -> Path:
    base = RECON_ROOT / condition_key / SUBJECT_ID / "VC"
    return base / extra if extra else base


def _load_targets() -> list:
    return load_target_images(IMAGE_SELECTION)


def _load_recons(condition_key: str, extra: str | None = None) -> list:
    return load_recon_images(_recon_dir(condition_key, extra), IMAGE_SELECTION)


def export_diff_panel() -> None:
    targets = _load_targets()
    adam_images = _load_recons(BASE_CONDITION, "wo_lang")
    sgld_images = _load_recons(BASE_CONDITION)
    diff_images = [ImageChops.difference(a, b) for a, b in zip(adam_images, sgld_images)]

    conditions = [
        {"title": "Target", "images": targets},
        {"title": "Adam (1000 iter)", "images": adam_images},
        {"title": "SGLD (500 iter)", "images": sgld_images},
        {"title": "Difference", "images": diff_images},
    ]

    drawer = GroupImageDrawer(
        conditions,
        title_fontcolor="black",
        title_fontsize=12,
        max_column_size=len(IMAGE_SELECTION),
    )
    panel = drawer.draw()
    panel.save(OUTPUT_DIR / f"fig05_{SUBJECT_ID}_recon_image_compare_diff_random.pdf")


def _export_sgld_variants(variants: Iterable[tuple[str, str]], output_name: str) -> None:
    targets = _load_targets()
    pre_sgld = _load_recons(BASE_CONDITION, "wo_lang")
    baseline = _load_recons(BASE_CONDITION)

    conditions = [
        {"title": "Target", "images": targets},
        {"title": "Pre-SGLD", "images": pre_sgld},
        {"title": "Koide-Majima", "images": baseline},
    ]

    for label, condition_key in variants:
        images = _load_recons(condition_key)
        conditions.append({"title": label, "images": images})

    drawer = GroupImageDrawer(
        conditions,
        title_fontcolor="black",
        title_fontsize=12,
        max_column_size=len(IMAGE_SELECTION),
    )
    panel = drawer.draw()
    panel.save(OUTPUT_DIR / output_name)


def export_variant_panels() -> None:
    _export_sgld_variants(SGLD_VARIANTS, f"figS05_{SUBJECT_ID}_recon_image_compare.pdf")
    _export_sgld_variants(
        SGLD_VARIANTS_NORMAL_TEMP,
        f"figS05_{SUBJECT_ID}_recon_image_compare_normal_temp.pdf",
    )


def main() -> None:
    export_diff_panel()
    export_variant_panels()


if __name__ == "__main__":
    main()
