"""Generate the ablation reconstruction panels for Figures A2-A4.

Each panel stacks the target stimuli on top of the reconstructions obtained under the
four ablation conditions, one row per condition: one panel per subject over all 25
stimuli (A2 = S1, A3 = S2, A4 = S3). The five-stimulus panel of Figure 5B comes from
Fig5_ablation_assets.py, which draws its stimuli with a seed.

The manuscript's A2-A4 also carry a reference row from a separate iCNN implementation.
Those reconstructions are produced outside this repository, so the row is drawn only
when its directory is passed explicitly; by default the panels show the ablation
conditions alone.

Examples
--------
    python Fig5_ablation_recon_panels.py                        # S1/S2/S3
    python Fig5_ablation_recon_panels.py --subjects S2          # single subject
    python Fig5_ablation_recon_panels.py --deeprecon-root DIR   # add the reference row
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
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "fig05"

COMPARISON_CONDITIONS = {
    "Koide-Majima": "original_all",
    "w/o Baye": "AdamOnly_all",
    "w/o CLIP": "VGGonly_all",
    "w/o Baye and CLIP": "wo_SGLD_CLIP_all",
}

# Reference-row reconstructions from a separate iCNN implementation, produced outside
# this repository (see the Figure A2-A4 note in README). Subjects are keyed there by
# initials rather than S1/S2/S3.
DEEPRECON_LABEL = "iCNN (reference)"
DEEPRECON_SUBJECT_MAP = {"S1": "TH", "S2": "AM", "S3": "ES"}

# One appendix figure per subject.
APPENDIX_FIGURE = {"S1": "A2", "S2": "A3", "S3": "A4"}

PANEL_FILENAME = "Fig{appendix}_{subject}_recon_image_compare.pdf"


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
    output_path = output_dir / filename.format(
        subject=subject, appendix=APPENDIX_FIGURE.get(subject, subject)
    )
    drawer.draw().save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
        default=None,
        help="directory holding <initials>/VC reference tiff reconstructions. Omitted "
             "by default: those reconstructions come from a separate repository, so "
             "the reference row is drawn only when this is given.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_directory(args.output_dir)

    for subject in args.subjects:
        saved = export_panel(
            subject,
            SOURCE_IMAGE_NAMES,
            PANEL_FILENAME,
            args.recon_root,
            output_dir,
            args.deeprecon_root,
        )
        print(f"saved {saved}")


if __name__ == "__main__":
    main()
