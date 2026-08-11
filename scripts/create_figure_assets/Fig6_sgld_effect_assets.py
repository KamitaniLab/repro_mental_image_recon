"""Generate the Figure 6B panel: reconstructions before and after SGLD.

Rows are the target, the reconstruction after the 1000 Adam steps (pre-SGLD, saved by
the reconstruction script under ``wo_lang/``), the reconstruction after the 500 SGLD
steps, and their pixel-wise absolute difference.

The manuscript shows this for the same stimuli as Figure 5B, so the selection comes
from ``repro_mental_image_recon.figures.stimuli``, which Fig 5B also draws from,
rather than being restated here where the two could drift.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import ImageChops

from repro_mental_image_recon.figures.assets import (
    ensure_directory,
    load_recon_images,
    load_target_images,
    project_root,
)
from repro_mental_image_recon.figures.drawing import GroupImageDrawer
from repro_mental_image_recon.figures.stimuli import SUBJECT_ID, select_random_stimuli

PROJECT_ROOT = project_root()
RECON_ROOT = PROJECT_ROOT / "results" / "rep_recon_image_koide-majima"
OUTPUT_DIR = PROJECT_ROOT / "assets" / "fig06"

BASE_CONDITION = "original_all"
# "using the same target samples shown in Figure 5B" -- see module docstring.
IMAGE_SELECTION = select_random_stimuli()


def _recon_dir(recon_root: Path, extra: str | None = None) -> Path:
    base = recon_root / BASE_CONDITION / SUBJECT_ID / "VC"
    return base / extra if extra else base


def export_diff_panel(recon_root: Path, output_dir: Path) -> Path:
    targets = load_target_images(IMAGE_SELECTION)
    adam_images = load_recon_images(_recon_dir(recon_root, "wo_lang"), IMAGE_SELECTION)
    sgld_images = load_recon_images(_recon_dir(recon_root), IMAGE_SELECTION)
    diff_images = [
        ImageChops.difference(a, b) for a, b in zip(adam_images, sgld_images)
    ]

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
    output_path = output_dir / f"Fig6B_{SUBJECT_ID}_recon_image_diff.pdf"
    drawer.draw().save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recon-root",
        type=Path,
        default=RECON_ROOT,
        help="directory holding <condition>/<subject>/VC{,/wo_lang}",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    saved = export_diff_panel(args.recon_root, ensure_directory(args.output_dir))
    print(f"saved {saved}")


if __name__ == "__main__":
    main()
