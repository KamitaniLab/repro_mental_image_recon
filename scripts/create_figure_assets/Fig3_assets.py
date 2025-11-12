"""Generate assets for Figure 3 (CLIP-only reconstructions and evaluation)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from fig_utils import GroupImageDrawer
from figure_asset_utils import (
    SUBJECTS,
    ensure_directory,
    load_recon_images,
    load_target_images,
    project_root,
)

PROJECT_ROOT = project_root()
RECON_IMAGE_ROOT = PROJECT_ROOT / "results" / "rep_recon_image_koide-majima"
OUTPUT_DIR = ensure_directory(PROJECT_ROOT / "assets" / "fig03")

CONDITION_LABEL = "CLIP-only"
CONDITION_KEY = "CLIPonly_all"
RANDOM_POOL = (
    "imageryExpStim19_anat_iguana.tiff",
    "imageryExpStim21_anat_swan.tiff",
    #"imageryExpStim20_anat_leopard.tiff",
    #"imageryExpStim17_anat_goat.tiff",
    #"imageryExpStim22_inat_airliner.tiff",
    "imageryExpStim23_inat_bowling.tiff",
    #"imageryExpStim24_inat_post.tiff",
    "imageryExpStim25_inat_stainedglass.tiff",
    #"imageryExpStim26_inat_umbrella.tiff",
)
RANDOM_COUNT = 4
RANDOM_SEED = 42

PAIRWISE_EVAL_MODEL = "pixel"
PAIRWISE_EVAL_METRIC = "correlation"


def _recon_dir(subject: str) -> Path:
    return RECON_IMAGE_ROOT / CONDITION_KEY / subject / "VC"


def _select_random_stimuli() -> tuple[str, ...]:
    rng = np.random.default_rng(RANDOM_SEED)
    selection = rng.choice(RANDOM_POOL, size=RANDOM_COUNT, replace=False)
    return tuple(sorted(selection))


def _load_subject_recons(subject: str, image_names: Sequence[str]) -> list:
    return load_recon_images(_recon_dir(subject), image_names)


def export_random_reconstruction_panel() -> None:
    image_names = _select_random_stimuli()
    target_images = load_target_images(image_names)
    conditions = [{"title": "Target", "images": target_images}]

    for subject in SUBJECTS:
        recon_images = _load_subject_recons(subject, image_names)
        conditions.append({"title": subject, "images": recon_images})

    drawer = GroupImageDrawer(
        conditions,
        title_fontcolor="black",
        title_fontsize=12,
        max_column_size=len(image_names),
    )
    panel = drawer.draw()
    panel.save(OUTPUT_DIR / "Fig3A_recon_image_random.pdf")


def _load_same_feature_scores(condition_key: str) -> list[float]:
    eval_path = RECON_IMAGE_ROOT / condition_key / "pairwise_identification_results_koide_majima_optsame_metric.pkl"
    with eval_path.open("rb") as handle:
        data = pickle.load(handle)
    # The pickle structure is assumed to be {subject_id: score_list}
    return [float(np.mean(values)) for values in data.values()]


def _load_independent_scores(
    condition_key: str,
    eval_model: str,
    metric: str,
) -> list[float]:
    eval_path = RECON_IMAGE_ROOT / condition_key / f"pairwise_identification_results_{eval_model}_{metric}_sim_matrix.pkl"
    with eval_path.open("rb") as handle:
        data = pickle.load(handle)

    subject_scores: list[float] = []
    for subject in SUBJECTS:
        layers = data[subject]
        layer_names = list(layers.keys())
        sim_matrix = sum(layers[name] for name in layer_names) / len(layer_names)
        diagonal = np.diag(sim_matrix)
        comparisons = sim_matrix - diagonal[:, None]
        correct_rate = np.sum(comparisons < 0, axis=1) / (sim_matrix.shape[1] - 1)
        subject_scores.append(float(np.mean(correct_rate)))
    return subject_scores


def export_pairwise_identification_plot() -> None:
    same_feature_scores = _load_same_feature_scores(CONDITION_KEY)
    independent_scores = _load_independent_scores(CONDITION_KEY, PAIRWISE_EVAL_MODEL, PAIRWISE_EVAL_METRIC)

    df = pd.DataFrame(
        [
            {
                "condition": "Same features",
                "points": same_feature_scores,
                "ave_acc": float(np.mean(same_feature_scores)),
            },
            {
                "condition": "Pixel (independent)",
                "points": independent_scores,
                "ave_acc": float(np.mean(independent_scores)),
            },
        ]
    )

    fig, ax = plt.subplots(figsize=(6, 7))
    x = np.arange(len(df))
    bars = ax.bar(x, df["ave_acc"], color="0.75", width=0.5)

    for idx, scores in enumerate(df["points"]):
        ax.scatter(
            np.full(len(scores), x[idx]),
            scores,
            color="0.1",
            edgecolor="white",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"], rotation=15, ha="right")
    ax.set_ylabel("Pairwise identification accuracy")
    ax.set_ylim(0.3, 1.0)
    ax.axhline(0.5, color="0.5", linestyle="--", linewidth=1)
    ax.set_title(f"Pairwise identification ({CONDITION_LABEL})")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "Fig3B_pairwise_identification_concat_pixel_correlation.pdf"
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    export_random_reconstruction_panel()
    export_pairwise_identification_plot()


if __name__ == "__main__":
    main()
