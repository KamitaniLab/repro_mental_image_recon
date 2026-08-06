"""Generate the ablation reconstruction panels (all-stimuli / selected-stimuli).

Reproduces the two panel sets that were originally produced by
``original_jupyter/Fig4_assets.ipynb`` (cells 2 and 4) and shipped as
``figures/250410_recon_image_koide-majima/Fig_03/{subject}_recon_image_{compare,selected}.pdf``.

Each panel stacks the target stimuli on top of the reconstructions obtained
under the four ablation conditions, one row per condition, followed by the
DeepRecon (iCNN/VGG19) reconstructions as an external reference row.

Examples
--------
    python Fig4_recon_panels.py                      # both panels, S1/S2/S3
    python Fig4_recon_panels.py --panel selected     # selected stimuli only
    python Fig4_recon_panels.py --subjects S2        # single subject
    python Fig4_recon_panels.py --no-deeprecon       # ablation conditions only
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from PIL import Image

from fig_utils import GroupImageDrawer
from figure_asset_utils import (
    SOURCE_IMAGE_NAMES,
    SUBJECTS,
    ensure_directory,
    load_recon_images,
    load_target_images,
    project_root,
)

PROJECT_ROOT = project_root()
DEFAULT_RECON_ROOT = PROJECT_ROOT / "results" / "rep_recon_image_koide-majima"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "fig04"

COMPARISON_CONDITIONS = {
    "Koide-Majima": "original_all",
    "w/o Baye": "AdamOnly_all",
    "w/o CLIP": "VGGonly_all",
    "w/o Baye and CLIP": "wo_SGLD_CLIP_all",
}

# DeepRecon (iCNN + VGG19 relu7 generator) reconstructions produced outside this
# repository. Subjects there are keyed by initials rather than S1/S2/S3.
DEFAULT_DEEPRECON_ROOT = Path(
    "/home/nu/mtanaka/project/feature-based-reconstruction-test/murakiy/work/data/reconstruction/icnn"
    "/check_loss_and_latents_recon_icnn_image_gd_dist_vgg19_relu7generator_scaling_feature_std_train_mean_center_1000iter"
    "/decoded/Imagery_deeprecon_VGG19"
)
DEEPRECON_LABEL = "DeepRecon (VGG19)"
DEEPRECON_SUBJECT_MAP = {"S1": "TH", "S2": "AM", "S3": "ES"}

# Hand-picked stimuli used for the compact panel (one artificial shape plus
# four natural images), in the order they appear in the published figure.
SELECTED_IMAGE_NAMES = (
    "imageryExpStim15_black_X.tiff",
    "imageryExpStim20_anat_leopard.tiff",
    "imageryExpStim17_anat_goat.tiff",
    "imageryExpStim22_inat_airliner.tiff",
    "imageryExpStim23_inat_bowling.tiff",
)

PANEL_SPECS = {
    "compare": (SOURCE_IMAGE_NAMES, "{subject}_recon_image_compare.pdf"),
    "selected": (SELECTED_IMAGE_NAMES, "{subject}_recon_image_selected.pdf"),
}


def load_deeprecon_images(
    subject: str,
    image_names: Sequence[str],
    deeprecon_root: Path,
) -> list[Image.Image]:
    """Load DeepRecon reconstructions for ``subject`` in ``image_names`` order.

    Files are named ``recon_image-<stimulus stem>.tiff``, so stimuli are matched
    by name rather than by the positional fallback used for the ablation
    reconstructions (that directory also contains ``imageryExpStim16_fixation``,
    which would shift a position-based mapping).
    """
    initials = DEEPRECON_SUBJECT_MAP.get(subject)
    if initials is None:
        raise KeyError(
            f"No DeepRecon subject mapped to '{subject}' (known: {sorted(DEEPRECON_SUBJECT_MAP)})"
        )
    recon_dir = deeprecon_root / initials / "VC"

    images: list[Image.Image] = []
    for name in image_names:
        path = recon_dir / f"recon_image-{Path(name).stem}.tiff"
        if not path.exists():
            raise FileNotFoundError(f"Missing DeepRecon reconstruction: {path}")
        with Image.open(path) as img:
            images.append(img.convert("RGB"))
    return images


def export_panel(
    subject: str,
    image_names: Sequence[str],
    filename: str,
    recon_root: Path,
    output_dir: Path,
    deeprecon_root: Path | None = None,
) -> Path:
    """Draw one condition-comparison panel for ``subject`` and save it as PDF."""
    conditions = [{"title": "Target", "images": load_target_images(image_names)}]
    for label, condition_key in COMPARISON_CONDITIONS.items():
        recon_dir = recon_root / condition_key / subject / "VC"
        conditions.append({"title": label, "images": load_recon_images(recon_dir, image_names)})

    if deeprecon_root is not None:
        conditions.append(
            {
                "title": DEEPRECON_LABEL,
                "images": load_deeprecon_images(subject, image_names, deeprecon_root),
            }
        )

    drawer = GroupImageDrawer(conditions, title_fontcolor="black", title_fontsize=12)
    output_path = output_dir / filename.format(subject=subject)
    drawer.draw().save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--panel",
        choices=(*PANEL_SPECS, "all"),
        default="all",
        help="which panel set to export (default: all)",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=list(SUBJECTS),
        help=f"subject ids to export (default: {' '.join(SUBJECTS)})",
    )
    parser.add_argument(
        "--recon-root",
        type=Path,
        default=DEFAULT_RECON_ROOT,
        help="directory holding <condition>/<subject>/VC reconstructions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="destination directory for the generated PDFs",
    )
    parser.add_argument(
        "--deeprecon-root",
        type=Path,
        default=DEFAULT_DEEPRECON_ROOT,
        help="directory holding <initials>/VC DeepRecon tiff reconstructions",
    )
    parser.add_argument(
        "--no-deeprecon",
        action="store_true",
        help="omit the DeepRecon reference row",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)
    panels = list(PANEL_SPECS) if args.panel == "all" else [args.panel]
    deeprecon_root = None if args.no_deeprecon else args.deeprecon_root

    for panel in panels:
        image_names, filename = PANEL_SPECS[panel]
        for subject in args.subjects:
            saved = export_panel(
                subject, image_names, filename, args.recon_root, output_dir, deeprecon_root
            )
            print(f"saved {saved}")


if __name__ == "__main__":
    main()
