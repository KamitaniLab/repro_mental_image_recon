"""Summarize the SGLD (Langevin) sampling effect across a whole subject.

The reviewer noted that the "minimal SGLD effect" observation in Figure 5C-E was
shown only for a single representative trace (Subject 1 / one stimulus). This
script computes the *same* per-chain quantities used in that figure -- the
per-coordinate standard deviation of the SGLD chain and its autocorrelation --
for every reconstruction of one subject (10 iterations x 25 stimuli), so the
distribution across the full dataset can be reported.

The reconstruction trace files are large (~672 MB each, ~164 GB per subject), so
the work is parameterized by subject and run once per subject:

    python scripts/experiments/sgld_effect_summary.py S1
    python scripts/experiments/sgld_effect_summary.py S2
    python scripts/experiments/sgld_effect_summary.py S3

By default only ``iter01`` is processed -- one independent SGLD chain per
(subject, stimulus), matching the trace used in Fig5c-e and covering every
subject x every target image (75 reconstructions). The ``iter*`` directories are
repeated seed-free re-runs of the same conditions; pass ``--iterations all`` (or
a comma-separated subset) to pool every repeat for an even larger summary.

Each run writes a compact ``.npz`` of per-reconstruction summaries that the
figure script aggregates.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "rep_recon_image_koide-majima_recon_variability_no_seed"
)
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "sgld_effect_summary"

# Trace keys holding the SGLD (Langevin) chain, matching
# Fig6_sgld_effect_diagnostic_assets.py.
LATENT_KEY = "current_LatentVec_withLangevin_list"
PIXEL_KEY = "currentImg_withLangevin_list"


def autocorr_mean_curve(traces: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean (over coordinates) autocorrelation curve of a ``(T, D)`` chain.

    Identical definition to ``Fig5c-e_assets.autocorr_matrix`` followed by a mean
    over the coordinate axis. Coordinates with zero variance contribute 0.
    """
    centered = traces - traces.mean(axis=0, keepdims=True)
    var = np.sum(centered * centered, axis=0)
    curve = np.empty(max_lag + 1, dtype=np.float64)
    curve[0] = 1.0
    nonzero = var > 0
    for lag in range(1, max_lag + 1):
        numerator = np.einsum("ij,ij->j", centered[:-lag], centered[lag:])
        per_coord = np.zeros_like(var)
        per_coord[nonzero] = numerator[nonzero] / var[nonzero]
        curve[lag] = per_coord.mean()
    return curve


def _std_summary(per_coord_std: np.ndarray) -> dict[str, float]:
    """Summarize the per-coordinate std distribution over coordinates."""
    return {
        "std_mean": float(per_coord_std.mean()),
        "std_median": float(np.median(per_coord_std)),
        "std_p95": float(np.percentile(per_coord_std, 95)),
        "std_max": float(per_coord_std.max()),
    }


def summarize_trace(path: Path, max_lag: int) -> dict[str, object]:
    """Compute std and autocorrelation summaries for one reconstruction file."""
    with path.open("rb") as handle:
        data = pickle.load(handle)

    latent = np.asarray(data[LATENT_KEY], dtype=np.float32)
    latent = latent.reshape(latent.shape[0], -1)
    pixel = np.asarray(data[PIXEL_KEY], dtype=np.float32)
    pixel = pixel.reshape(pixel.shape[0], -1)

    result: dict[str, object] = {"n_steps": int(latent.shape[0])}
    for prefix, traces in (("latent", latent), ("pixel", pixel)):
        per_coord_std = traces.std(axis=0).astype(np.float32)
        for key, value in _std_summary(per_coord_std).items():
            result[f"{prefix}_{key}"] = value
        # Full per-coordinate SD distribution (no mean aggregation) so the
        # figure can show the true spread, with the tail up to std_max.
        result[f"{prefix}_std_all"] = per_coord_std
        result[f"{prefix}_autocorr"] = autocorr_mean_curve(traces, max_lag)
    return result


def iter_labels(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob("iter*") if p.is_dir())


def resolve_iterations(available: list[str], requested: str) -> list[str]:
    """Resolve the ``--iterations`` selection against the available iter dirs."""
    if requested == "all":
        return available
    selected = [tok.strip() for tok in requested.split(",") if tok.strip()]
    missing = [it for it in selected if it not in available]
    if missing:
        raise ValueError(
            f"Requested iterations {missing} not found (available: {available})"
        )
    return selected


def process_subject(
    subject: str,
    condition: str,
    root: Path,
    max_lag: int,
    iterations_request: str,
) -> dict[str, np.ndarray]:
    subject_root = root / condition / subject
    if not subject_root.is_dir():
        raise FileNotFoundError(f"Subject directory not found: {subject_root}")

    available = iter_labels(subject_root)
    if not available:
        raise FileNotFoundError(f"No iter* directories under {subject_root}")
    iterations = resolve_iterations(available, iterations_request)

    # Discover the canonical stimulus ordering from the first iteration.
    first_vc = subject_root / iterations[0] / "VC"
    stim_paths = sorted(first_vc.glob("*.pkl"))
    if not stim_paths:
        raise FileNotFoundError(f"No trace files under {first_vc}")
    stim_names = [p.stem for p in stim_paths]
    n_iter, n_stim = len(iterations), len(stim_names)

    scalar_keys = [
        f"{prefix}_{stat}"
        for prefix in ("latent", "pixel")
        for stat in ("std_mean", "std_median", "std_p95", "std_max")
    ]
    scalars = {key: np.full((n_iter, n_stim), np.nan) for key in scalar_keys}
    latent_ac = np.full((n_iter, n_stim, max_lag + 1), np.nan)
    pixel_ac = np.full((n_iter, n_stim, max_lag + 1), np.nan)
    # Full per-coordinate SD values pooled over all reconstructions (no mean
    # aggregation), used for the distribution / violin plots.
    latent_std_all: list[np.ndarray] = []
    pixel_std_all: list[np.ndarray] = []

    total = n_iter * n_stim
    done = 0
    for ii, iteration in enumerate(iterations):
        vc_dir = subject_root / iteration / "VC"
        for si, stim in enumerate(stim_names):
            path = vc_dir / f"{stim}.pkl"
            done += 1
            if not path.exists():
                print(f"[{done}/{total}] MISSING {path}", flush=True)
                continue
            summary = summarize_trace(path, max_lag)
            for key in scalar_keys:
                scalars[key][ii, si] = summary[key]
            latent_ac[ii, si] = summary["latent_autocorr"]
            pixel_ac[ii, si] = summary["pixel_autocorr"]
            latent_std_all.append(summary["latent_std_all"])
            pixel_std_all.append(summary["pixel_std_all"])
            print(
                f"[{done}/{total}] {subject} {iteration} {stim} "
                f"pixel_std_mean={summary['pixel_std_mean']:.3f} "
                f"pixel_std_max={summary['pixel_std_max']:.2f} "
                f"latent_std_mean={summary['latent_std_mean']:.4f}",
                flush=True,
            )

    out: dict[str, np.ndarray] = dict(scalars)
    out["latent_autocorr"] = latent_ac
    out["pixel_autocorr"] = pixel_ac
    out["latent_std_all"] = (
        np.concatenate(latent_std_all) if latent_std_all else np.empty(0, np.float32)
    )
    out["pixel_std_all"] = (
        np.concatenate(pixel_std_all) if pixel_std_all else np.empty(0, np.float32)
    )
    out["iterations"] = np.asarray(iterations)
    out["stimulus_names"] = np.asarray(stim_names)
    out["lags"] = np.arange(max_lag + 1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject", type=str, help="Subject id, e.g. S1, S2, S3")
    parser.add_argument("--condition", type=str, default="original_all")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--max-lag", type=int, default=30)
    parser.add_argument(
        "--iterations",
        type=str,
        default="iter01",
        help="'iter01' (default), 'all', or a comma-separated list e.g. 'iter01,iter02'",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    out = process_subject(
        args.subject, args.condition, args.root, args.max_lag, args.iterations
    )

    out_dir = args.out_dir / args.condition
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.subject}.npz"
    np.savez_compressed(out_path, **out)
    print(f"Saved summary to {out_path}", flush=True)


if __name__ == "__main__":
    main()
