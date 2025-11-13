"""Generate assets for Figure 4 (ablation reconstructions and preference analysis)."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pickle

from fig_utils import GroupImageDrawer
from figure_asset_utils import (
    ensure_directory,
    load_recon_images,
    load_target_images,
    project_root,
)

PROJECT_ROOT = project_root()
RECON_ROOT = PROJECT_ROOT / "results" / "rep_recon_image_koide-majima"
OUTPUT_DIR = ensure_directory(PROJECT_ROOT / "assets" / "fig04")

SUBJECT_ID = "S2"
COMPARISON_CONDITIONS = {
    "Koide-Majima": "original_all",
    "w/o Baye": "AdamOnly_all",
    "w/o CLIP": "VGGonly_all",
    "w/o Baye and CLIP": "wo_SGLD_CLIP_all",
}

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
    #"imageryExpStim25_inat_stainedglass.tiff",
    #"imageryExpStim26_inat_umbrella.tiff",
)

RANDOM_COUNT = 4
RANDOM_SEED = 42

PREFERENCE_MODELS = ("dreamsim", "alexnet", "RN50", "lpips")
PREFERENCE_PREFIX_FOUR = "ref_compare_4"
PREFERENCE_PREFIX_TWO = "ref_compare_2"

FOUR_METHOD_LABELS = (
    "Koide-Majima",
    "w/o SGLD",
    "w/o CLIP",
    "w/o Baye and CLIP",
)
FOUR_GROUP_KEYS = {
    "LPIPS": ("lpips", "pool"),
    "DreamSim": ("dreamsim", "output"),
    "AlexNet conv2": ("alexnet", "features.3"),
    "AlexNet conv5": ("alexnet", "features.10"),
    "CLIP RN50": ("RN50", "pool"),
}
TWO_METHOD_LABELS = ("Koide-Majima", "w/o Baye and CLIP")
TWO_GROUP_KEYS = {
    "LPIPS": ("lpips", "pool"),
    "DreamSim": ("dreamsim", "output"),
    "AlexNet conv2": ("alexnet", "features.3"),
    "AlexNet conv5": ("alexnet", "features.10"),
    "CLIP RN50": ("RN50", "pool"),
}

STACK_COLORS_FOUR = (
    (30 / 255, 118 / 255, 180 / 255),
    (174 / 255, 198 / 255, 232 / 255),
    (254 / 255, 127 / 255, 11 / 255),
    (254 / 255, 187 / 255, 119 / 255),
)
STACK_COLORS_TWO = (
    (30 / 255, 118 / 255, 180 / 255),
    (254 / 255, 187 / 255, 119 / 255),
)


def _recon_dir(condition_key: str) -> Path:
    return RECON_ROOT / condition_key / SUBJECT_ID / "VC"


def _select_random_stimuli() -> tuple[str, ...]:
    rng = np.random.default_rng(RANDOM_SEED)
    selection = rng.choice(RANDOM_POOL, size=RANDOM_COUNT, replace=False)
    return tuple(sorted(selection))


def export_random_comparison_panel() -> None:
    image_names = _select_random_stimuli()
    target_images = load_target_images(image_names)
    conditions = [{"title": "Target", "images": target_images}]

    for label, condition_key in COMPARISON_CONDITIONS.items():
        recon_images = load_recon_images(_recon_dir(condition_key), image_names)
        conditions.append({"title": label, "images": recon_images})

    drawer = GroupImageDrawer(
        conditions,
        title_fontcolor="black",
        title_fontsize=12,
        max_column_size=len(image_names),
    )
    panel = drawer.draw()
    panel.save(OUTPUT_DIR / "Fig4b_S2_recon_image_random.pdf")


def _load_preference_summary(prefix: str) -> dict[str, dict[str, np.ndarray]]:
    summary: dict[str, dict[str, np.ndarray]] = {}
    for model in PREFERENCE_MODELS:
        result_path = RECON_ROOT / f"{prefix}_preference_analysis_results_{model}_correlation.pkl"
        sim_path = RECON_ROOT / f"{prefix}_preference_analysis_results_{model}_correlation_sim_matrix.pkl"
        with result_path.open("rb") as handle:
            eval_data: dict[str, np.ndarray] = pickle.load(handle)
        with sim_path.open("rb") as handle:
            sim_data: dict[str, np.ndarray] = pickle.load(handle)

        ordered = [sim_data[key] for key in sorted(sim_data.keys())]
        mean_matrix = np.mean(np.stack(ordered), axis=0)
        pooled = np.argmax(mean_matrix, axis=1)

        eval_data = dict(eval_data)
        eval_data["pool"] = pooled
        summary[model] = eval_data
    return summary


def _prepare_group_data(
    summary: dict[str, dict[str, np.ndarray]],
    group_keys: dict[str, tuple[str, str]],
) -> dict[str, np.ndarray]:
    group_data: dict[str, np.ndarray] = {}
    for group_label, (model, key) in group_keys.items():
        group_data[group_label] = np.asarray(summary[model][key])
    return group_data


def _plot_stacked_preferences(
    data: dict[str, np.ndarray],
    method_labels: Sequence[str],
    colors: Sequence[tuple[float, float, float]],
    output_path: Path,
) -> None:
    num_methods = len(method_labels)
    group_names = list(data.keys())
    proportions = []
    for values in data.values():
        counts = np.bincount(values, minlength=num_methods)
        proportions.append(counts / counts.sum())
    proportions = np.array(proportions)

    x = np.arange(len(group_names))
    bottom = np.zeros(len(group_names))

    fig, ax = plt.subplots(figsize=(6, 8))
    for idx, (label, color) in enumerate(zip(method_labels, colors[::-1])):
        height = proportions[:, -1 - idx]
        ax.bar(x, height, bottom=bottom, color=color, label=label)
        for xpos, base, h in zip(x, bottom, height):
            if h > 0:
                ax.text(xpos, base + h / 2, f"{h:.2f}", ha="center", va="center", color="white", fontsize=8)
        bottom += height

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Preference proportion")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def export_preference_analyses() -> None:
    summary_four = _load_preference_summary(PREFERENCE_PREFIX_FOUR)
    data_four = _prepare_group_data(summary_four, FOUR_GROUP_KEYS)
    _plot_stacked_preferences(
        data_four,
        FOUR_METHOD_LABELS,
        STACK_COLORS_FOUR,
        OUTPUT_DIR / "Fig4d_preference_analysis_four.pdf",
    )

    summary_two = _load_preference_summary(PREFERENCE_PREFIX_TWO)
    data_two = _prepare_group_data(summary_two, TWO_GROUP_KEYS)
    colors = STACK_COLORS_TWO + ((0.8, 0.8, 0.8),) * (len(TWO_METHOD_LABELS) - len(STACK_COLORS_TWO))
    _plot_stacked_preferences(
        data_two,
        TWO_METHOD_LABELS,
        colors[: len(TWO_METHOD_LABELS)],
        OUTPUT_DIR / "Fig4e_preference_analysis_two.pdf",
    )


def main() -> None:
    export_random_comparison_panel()
    export_preference_analyses()


if __name__ == "__main__":
    main()
