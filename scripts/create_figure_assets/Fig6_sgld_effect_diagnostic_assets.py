"""Generate the sampling-analysis figure for Figure 6C-E.

One SGLD chain from the seed-free repeat runs: latent and pixel traces (C),
histograms of their within-chain standard deviation (D), and the
coordinate-averaged autocorrelation (E).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle

from figure_asset_utils import (
    ensure_directory,
    project_root,
)

PROJECT_ROOT = project_root()
RECON_ROOT = (
    PROJECT_ROOT / "results" / "rep_recon_image_koide-majima_recon_variability_no_seed"
)
OUTPUT_DIR = PROJECT_ROOT / "assets" / "fig06"
OUTPUT_NAME = "Fig6CDE_sampling_analysis.pdf"

CONDITION_KEY = "original_all"
SUBJECT_ID = "S2"
ITERATION = "iter01"
STIMULUS_INDEX = 18  # zero-based index into SOURCE_IMAGE_NAMES
MAX_LAG = 30
TRACE_SAMPLE_COUNT = 3
TRACE_SEED = 10


def _trace_directory(recon_root: Path, iteration: str) -> Path:
    return recon_root / CONDITION_KEY / SUBJECT_ID / iteration / "VC"


def _resolve_trace_path(recon_root: Path, iteration: str, stimulus_index: int) -> Path:
    trace_dir = _trace_directory(recon_root, iteration)
    pkl_files = sorted(trace_dir.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No trace files found under {trace_dir}")
    if stimulus_index >= len(pkl_files):
        raise IndexError("Stimulus index out of range for available trace files")
    return pkl_files[stimulus_index]


def _load_trace_arrays(recon_root: Path) -> tuple[np.ndarray, np.ndarray]:
    trace_path = _resolve_trace_path(recon_root, ITERATION, STIMULUS_INDEX)
    with trace_path.open("rb") as handle:
        data = pickle.load(handle)

    latent = np.asarray(data["current_LatentVec_withLangevin_list"], dtype=np.float32)
    latent = latent.reshape(latent.shape[0], -1)

    pixels = np.asarray(data["currentImg_withLangevin_list"], dtype=np.float32)
    pixels = pixels.reshape(pixels.shape[0], -1)
    return latent, pixels


def autocorr_matrix(traces: np.ndarray, max_lag: int) -> np.ndarray:
    centered = traces - traces.mean(axis=0, keepdims=True)
    var = np.sum(centered * centered, axis=0)
    result = np.empty((max_lag + 1, traces.shape[1]), dtype=np.float32)
    result[0] = 1.0

    for lag in range(1, max_lag + 1):
        numerator = np.einsum("ij,ij->j", centered[:-lag], centered[lag:])
        with np.errstate(divide="ignore", invalid="ignore"):
            result[lag] = numerator / var
        result[lag, var == 0] = 0.0
    return result


def _plot_autocorr(ax, mean: np.ndarray, std: np.ndarray, title: str) -> None:
    lags = np.arange(len(mean))
    ax.plot(lags, mean, color="0.3", linewidth=1.2)
    ax.fill_between(lags, mean - std, mean + std, color="0.3", alpha=0.2)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_ylim(0, 1)
    ax.set_title(title)


def export_sampling_analysis(recon_root: Path, output_dir: Path) -> Path:
    latent_traces, pixel_traces = _load_trace_arrays(recon_root)
    latent_std = latent_traces.std(axis=0)
    pixel_std = pixel_traces.std(axis=0)

    rng = np.random.default_rng(TRACE_SEED)
    latent_indices = np.sort(
        rng.choice(latent_traces.shape[1], TRACE_SAMPLE_COUNT, replace=False)
    )
    pixel_indices = np.sort(
        rng.choice(pixel_traces.shape[1], TRACE_SAMPLE_COUNT, replace=False)
    )
    latent_max_idx = int(np.argmax(latent_std))
    pixel_max_idx = int(np.argmax(pixel_std))

    latent_autocorr = autocorr_matrix(latent_traces, MAX_LAG)
    pixel_autocorr = autocorr_matrix(pixel_traces, MAX_LAG)

    latent_mean = latent_autocorr.mean(axis=1)
    latent_sd = latent_autocorr.std(axis=1)
    pixel_mean = np.nanmean(pixel_autocorr, axis=1)
    pixel_sd = np.nanstd(pixel_autocorr, axis=1)

    fig, axes = plt.subplots(3, 2, figsize=(10, 6))

    latent_trace_indices = list(latent_indices) + [latent_max_idx]
    colors_latent = [
        plt.get_cmap("tab20")(2 * i + 1) for i in range(TRACE_SAMPLE_COUNT)
    ] + ["0.5"]
    for idx, color in zip(latent_trace_indices, colors_latent):
        axes[0, 0].plot(latent_traces[:, idx], color=color, linewidth=1.0)
    axes[0, 0].set_ylabel("Latent value")
    axes[0, 0].set_title("Latent traces")

    axes[1, 0].hist(latent_std, bins=100, color="0.6", edgecolor="white")
    for idx, color in zip(latent_trace_indices, colors_latent):
        axes[1, 0].axvline(latent_std[idx], color=color, linestyle="--")
    axes[1, 0].set_xlabel("Standard deviation")
    axes[1, 0].set_ylabel("Frequency")

    pixel_trace_indices = list(pixel_indices) + [pixel_max_idx]
    colors_pixel = [plt.get_cmap("tab20")(2 * i) for i in range(TRACE_SAMPLE_COUNT)] + [
        "0.4"
    ]
    for idx, color in zip(pixel_trace_indices, colors_pixel):
        axes[0, 1].plot(pixel_traces[:, idx], color=color, linewidth=1.0)
    axes[0, 1].set_ylabel("Pixel value")
    axes[0, 1].set_ylim(0, 255)
    axes[0, 1].set_title("Pixel traces")

    axes[1, 1].hist(pixel_std, bins=100, color="0.75", edgecolor="white")
    for idx, color in zip(pixel_trace_indices, colors_pixel):
        axes[1, 1].axvline(pixel_std[idx], color=color, linestyle="--")
    axes[1, 1].set_xlabel("Standard deviation")
    axes[1, 1].set_ylabel("Frequency")

    _plot_autocorr(axes[2, 0], latent_mean, latent_sd, "Latent autocorrelation")
    _plot_autocorr(axes[2, 1], pixel_mean, pixel_sd, "Pixel autocorrelation")

    fig.tight_layout()
    output_path = output_dir / OUTPUT_NAME
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recon-root", type=Path, default=RECON_ROOT,
                        help="root of the seed-free repeat runs")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    saved = export_sampling_analysis(args.recon_root, ensure_directory(args.output_dir))
    print(f"saved {saved}")


if __name__ == "__main__":
    main()
