"""Systematic summary of the SGLD sampling effect across all subjects/images.

This is Figure A6. It extends Figure 6C-E -- which illustrated the minimal SGLD effect on a single
trace -- with a dataset-wide quantification. It aggregates the per-reconstruction
summaries produced by ``scripts/experiments/sgld_effect_summary.py`` (one ``.npz``
per subject) into:

  * per-subject distributions of the per-coordinate std of the SGLD chain
    (pixel and latent space), i.e. how much the chain actually moves; and
  * the autocorrelation of the SGLD chain aggregated across all reconstructions
    (per-subject mean curves + overall mean +/- SD), i.e. how poorly it mixes.

It also writes a CSV of summary statistics. Run the experiment script for each
subject first, then::

    python scripts/create_figure_assets/FigA6_sgld_systematic_assets.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from repro_mental_image_recon.figures.assets import SUBJECTS, ensure_directory, project_root

PROJECT_ROOT = project_root()
DEFAULT_SUMMARY_DIR = PROJECT_ROOT / "results" / "sgld_effect_summary"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "fig06"

SUBJECT_COLORS = {"S1": "#1b9e77", "S2": "#d95f02", "S3": "#7570b3"}


def load_subject(summary_dir: Path, subject: str) -> dict[str, np.ndarray]:
    path = summary_dir / f"{subject}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing summary for {subject}: {path}\n"
            f"Run: python scripts/experiments/sgld_effect_summary.py {subject}"
        )
    with np.load(path, allow_pickle=True) as handle:
        return {key: handle[key] for key in handle.files}


def _finite(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).ravel()
    return flat[np.isfinite(flat)]


# Number of coordinates subsampled per subject purely for the violin KDE
# (the full pooled arrays have millions of points; the bound/stats use them all).
_KDE_SAMPLE = 40000


def _plot_violin(ax, per_subject: dict[str, np.ndarray], ylabel: str,
                 log: bool) -> None:
    """Violin of the full per-coordinate SD distribution, one violin per subject.

    No mean aggregation: every coordinate's SD over the SGLD chain is a sample,
    so the violins span up to the largest variability (std_max) observed.
    """
    subjects = list(per_subject.keys())
    positions = np.arange(1, len(subjects) + 1)
    rng = np.random.default_rng(0)

    vmax = max(float(np.max(per_subject[s])) for s in subjects)
    plot_data = []
    dropped = 0
    for s in subjects:
        values = per_subject[s]
        if log:
            positive = values[values > 0]
            dropped += values.size - positive.size
            values = positive
        if values.size > _KDE_SAMPLE:
            values = rng.choice(values, _KDE_SAMPLE, replace=False)
        plot_data.append(values)

    parts = ax.violinplot(plot_data, positions=positions, widths=0.8,
                          showmedians=True, showextrema=False)
    for body, subject in zip(parts["bodies"], subjects):
        body.set_facecolor(SUBJECT_COLORS.get(subject, "0.6"))
        body.set_edgecolor("0.3")
        body.set_alpha(0.4)
    if "cmedians" in parts:
        parts["cmedians"].set_color("0.3")
        parts["cmedians"].set_linewidth(1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels(subjects)
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
        if dropped:
            ax.annotate(f"{dropped} zero-SD coords omitted (log)",
                        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7,
                        color="0.4")
    else:
        # Bound the axis by the largest variability value observed in the data.
        ax.set_ylim(0, vmax)


def _draw_autocorr_curves(ax, lags, per_subject_curves):
    """Plot per-subject mean curves + pooled mean +/- SD band.

    Returns (subject_means, pooled_mean, pooled_sd).
    """
    subject_means = []
    all_curves = []
    for subject, curves in per_subject_curves.items():
        mean_curve = np.nanmean(curves, axis=0)  # curves: (n_recon, n_lag)
        ax.plot(lags, mean_curve, color=SUBJECT_COLORS.get(subject, "0.3"),
                linewidth=1.3, label=subject)
        subject_means.append(mean_curve)
        all_curves.append(curves)
    pooled = np.concatenate(all_curves, axis=0)
    pooled_mean = np.nanmean(pooled, axis=0)
    pooled_sd = np.nanstd(pooled, axis=0)
    ax.fill_between(lags, pooled_mean - pooled_sd, pooled_mean + pooled_sd,
                    color="0.5", alpha=0.25, label="All +/- SD")
    return np.array(subject_means), pooled_mean, pooled_sd


def _plot_autocorr(ax, lags: np.ndarray, per_subject_curves: dict[str, np.ndarray],
                   title: str, zoom_window: tuple[int, int] | None = None) -> None:
    """Per-subject mean autocorrelation curves + overall mean +/- SD band.

    When ``zoom_window`` (lag_lo, lag_hi) is given, a boxed zoom of that lag range
    is drawn as an inset with connector lines (``indicate_inset_zoom``), with the
    y-axis tightly fit to the curves in the window.
    """
    subject_means, pooled_mean, pooled_sd = _draw_autocorr_curves(
        ax, lags, per_subject_curves)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.annotate(f"max pooled SD = {pooled_sd.max():.4f}",
                xy=(0.03, 0.06), xycoords="axes fraction", fontsize=7, color="0.4")
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    if zoom_window is not None:
        lo, hi = zoom_window
        window = (lags >= lo) & (lags <= hi)
        axins = ax.inset_axes([0.40, 0.14, 0.45, 0.40])
        _draw_autocorr_curves(axins, lags, per_subject_curves)
        lo_y = float(min(subject_means[:, window].min(),
                         (pooled_mean - pooled_sd)[window].min()))
        hi_y = float(max(subject_means[:, window].max(),
                         (pooled_mean + pooled_sd)[window].max()))
        margin = (hi_y - lo_y) * 0.05
        axins.set_xlim(lo, hi)
        axins.set_ylim(lo_y - margin, hi_y + margin)
        axins.set_xticks(np.arange(lo, hi + 1))  # integer lags only
        axins.tick_params(labelsize=6)
        ax.indicate_inset_zoom(axins, edgecolor="0.4")


def _summary_rows(metric: str, per_subject: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    pooled = []
    for subject, values in per_subject.items():
        pooled.append(values)
        rows.append({
            "metric": metric,
            "group": subject,
            "n": len(values),
            "median": float(np.median(values)),
            "q1": float(np.percentile(values, 25)),
            "q3": float(np.percentile(values, 75)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        })
    flat = np.concatenate(pooled)
    rows.append({
        "metric": metric,
        "group": "all",
        "n": len(flat),
        "median": float(np.median(flat)),
        "q1": float(np.percentile(flat, 25)),
        "q3": float(np.percentile(flat, 75)),
        "mean": float(np.mean(flat)),
        "max": float(np.max(flat)),
    })
    return rows


def _make_figure(pixel_std_all, latent_std_all, pixel_ac, latent_ac, lags,
                 n_total, log: bool):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # latent on the left column, pixel on the right column
    _plot_violin(axes[0, 0], latent_std_all,
                 "Per-latent SD over SGLD chain", log=log)
    axes[0, 0].set_title("Latent-space SGLD variability")
    _plot_violin(axes[0, 1], pixel_std_all,
                 "Per-pixel SD over SGLD chain\n(intensity, 0-255)", log=log)
    axes[0, 1].set_title("Pixel-space SGLD variability")
    # Latent curves are near-identical across subjects/recons; the per-subject
    # separation is largest at high lags, so box-zoom the lag 28-30 tail.
    _plot_autocorr(axes[1, 0], lags, latent_ac, "Latent autocorrelation",
                   zoom_window=(28, 30))
    _plot_autocorr(axes[1, 1], lags, pixel_ac, "Pixel autocorrelation")

    scale = "log y" if log else "linear y"
    fig.suptitle(f"SGLD sampling effect across all subjects x stimuli "
                 f"(n = {n_total} reconstructions; per-coordinate SD, {scale})")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def export(summary_dir: Path, output_dir: Path, table_dir: Path,
           subjects: tuple[str, ...]) -> None:
    summaries = {s: load_subject(summary_dir, s) for s in subjects}
    lags = next(iter(summaries.values()))["lags"]

    # Full per-coordinate SD distributions (no mean aggregation).
    pixel_std_all = {s: _finite(d["pixel_std_all"]) for s, d in summaries.items()}
    latent_std_all = {s: _finite(d["latent_std_all"]) for s, d in summaries.items()}
    pixel_ac = {
        s: d["pixel_autocorr"].reshape(-1, d["pixel_autocorr"].shape[-1])
        for s, d in summaries.items()
    }
    latent_ac = {
        s: d["latent_autocorr"].reshape(-1, d["latent_autocorr"].shape[-1])
        for s, d in summaries.items()
    }
    n_total = sum(int(np.isfinite(d["pixel_std_mean"]).sum())
                  for d in summaries.values())

    ensure_directory(output_dir)
    for log, suffix in ((False, ""), (True, "_log")):
        fig = _make_figure(pixel_std_all, latent_std_all, pixel_ac, latent_ac,
                           lags, n_total, log=log)
        pdf_path = output_dir / f"FigA6_sgld_systematic{suffix}.pdf"
        fig.savefig(pdf_path)
        fig.savefig(pdf_path.with_suffix(".png"), dpi=150)
        plt.close(fig)
        print(f"Saved figure to {pdf_path}")

    # The table is a number, not a figure, so it stays under results/.
    ensure_directory(table_dir)
    csv_path = table_dir / "FigA6_sgld_systematic_summary.csv"
    rows = (_summary_rows("pixel_per_coord_SD", pixel_std_all)
            + _summary_rows("latent_per_coord_SD", latent_std_all))
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved summary table to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", type=str, default="original_all")
    parser.add_argument("--summary-dir", type=Path, default=DEFAULT_SUMMARY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--table-dir", type=Path, default=None,
                        help="where the summary table goes (default: alongside the "
                             "per-subject .npz under results/)")
    args = parser.parse_args()
    summary_dir = args.summary_dir / args.condition
    export(summary_dir, args.output_dir, args.table_dir or summary_dir, SUBJECTS)


if __name__ == "__main__":
    main()
