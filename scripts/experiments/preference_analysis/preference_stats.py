"""Inferential statistics for the Fig. 5D/5E preference analyses.

Recomputes exactly the win counts that Fig5_ablation_assets.py turns into stacked bars, and
adds the numbers a reader needs to judge them:

  two-way (Fig 5E) : wins for Koide-Majima out of n, proportion with Wilson and
                     Clopper-Pearson 95% CIs, two-sided exact binomial (sign) test
                     against p0 = 0.5
  four-way (Fig 5D): per-condition "best" proportion with Clopper-Pearson 95% CI and
                     a two-sided exact binomial test against chance p0 = 1/4

Group definitions (metric -> (model file, layer key)) mirror Fig5_ablation_assets.py, including
the "pool" key, which is argmax over the layer-averaged similarity matrix.

The table is printed and also written next to the preference pickles under results/,
as CSV (default) or Markdown.

Usage:
  uv run python scripts/experiments/preference_analysis/preference_stats.py
  uv run python scripts/experiments/preference_analysis/preference_stats.py --format md
  uv run python scripts/experiments/preference_analysis/preference_stats.py --out-dir DIR
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RECON_ROOT = PROJECT_ROOT / "results" / "rep_recon_image_koide-majima"

PREFERENCE_MODELS = ("dreamsim", "alexnet", "RN50", "lpips")

# (metric label) -> (model, layer key); identical to FOUR_GROUP_KEYS / TWO_GROUP_KEYS
GROUP_KEYS = {
    "LPIPS": ("lpips", "pool"),
    "DreamSim": ("dreamsim", "output"),
    "AlexNet conv2": ("alexnet", "features.3"),
    "AlexNet conv5": ("alexnet", "features.10"),
    "CLIP RN50": ("RN50", "pool"),
}

TWO_METHOD_LABELS = ("Koide-Majima", "w/o Baye and CLIP")
FOUR_METHOD_LABELS = ("Koide-Majima", "w/o SGLD", "w/o CLIP", "w/o Baye and CLIP")


def load_preference_summary(
    prefix: str, recon_root: Path
) -> dict[str, dict[str, np.ndarray]]:
    """Same loader as Fig5_ablation_assets._load_preference_summary (adds the pooled key)."""
    summary: dict[str, dict[str, np.ndarray]] = {}
    for model in PREFERENCE_MODELS:
        result_path = (
            recon_root / f"{prefix}_preference_analysis_results_{model}_correlation.pkl"
        )
        sim_path = (
            recon_root
            / f"{prefix}_preference_analysis_results_{model}_correlation_sim_matrix.pkl"
        )
        with result_path.open("rb") as handle:
            eval_data = dict(pickle.load(handle))
        with sim_path.open("rb") as handle:
            sim_data = pickle.load(handle)
        ordered = [sim_data[key] for key in sorted(sim_data.keys())]
        eval_data["pool"] = np.argmax(np.mean(np.stack(ordered), axis=0), axis=1)
        summary[model] = eval_data
    return summary


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval (no continuity correction)."""
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval, via the binomtest inversion."""
    return tuple(binomtest(k, n).proportion_ci(1 - alpha, method="exact"))


def counts_for(values: np.ndarray, n_methods: int) -> np.ndarray:
    return np.bincount(np.asarray(values), minlength=n_methods)


def write_table(rows: list[dict], path: Path) -> None:
    """Write the tidy rows as CSV or Markdown, chosen by the file suffix."""
    fields = list(rows[0].keys())
    if path.suffix == ".md":

        def fmt(value):
            return f"{value:.3f}" if isinstance(value, float) else str(value)

        lines = [
            "| " + " | ".join(fields) + " |",
            "|" + "|".join(["---"] * len(fields)) + "|",
        ]
        lines += ["| " + " | ".join(fmt(r[f]) for f in fields) + " |" for r in rows]
        path.write_text("\n".join(lines) + "\n")
    else:
        import csv

        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recon-root",
        type=Path,
        default=RECON_ROOT,
        help=f"directory holding the preference pickles (default: {RECON_ROOT})",
    )
    ap.add_argument(
        "--format",
        choices=("csv", "md"),
        default="csv",
        help="output table format (default: csv)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="where to write the table; numbers stay under results/ "
        "(default: alongside the pickles)",
    )
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = args.recon_root

    rows = []

    # ---- Fig 5E: two-way, Koide-Majima vs w/o Baye and CLIP ------------------
    summary2 = load_preference_summary("ref_compare_2", args.recon_root)
    print("=" * 78)
    print("Fig5E  two-way preference: wins for Koide-Majima (vs w/o Baye and CLIP)")
    print("       CI = 95%; p = two-sided exact binomial (sign) test against 0.5")
    print("=" * 78)
    for label, (model, key) in GROUP_KEYS.items():
        values = np.asarray(summary2[model][key])
        counts = counts_for(values, len(TWO_METHOD_LABELS))
        n = int(counts.sum())
        k = int(counts[0])
        p_hat = k / n
        wl, wu = wilson_ci(k, n)
        cl, cu = clopper_pearson_ci(k, n)
        p = binomtest(k, n, 0.5, alternative="two-sided").pvalue
        print(
            f"  {label:<15s} {k:>2d}/{n} wins, {p_hat:.3f} "
            f"[{wl:.3f}, {wu:.3f}] Wilson  [{cl:.3f}, {cu:.3f}] CP,  p = {p:.3f}"
        )
        rows.append(
            dict(
                figure="Fig5E",
                metric=label,
                condition=TWO_METHOD_LABELS[0],
                wins=k,
                n=n,
                proportion=p_hat,
                wilson_lo=wl,
                wilson_hi=wu,
                cp_lo=cl,
                cp_hi=cu,
                p_value=p,
                null_p=0.5,
            )
        )

    # ---- Fig 5D: four-way, proportion best per condition ---------------------
    summary4 = load_preference_summary("ref_compare_4", args.recon_root)
    print()
    print("=" * 78)
    print("Fig5D  four-way preference: proportion of trials each condition is best")
    print(
        "       CI = 95% Clopper-Pearson; p = two-sided exact binomial vs chance 0.25"
    )
    print("=" * 78)
    for label, (model, key) in GROUP_KEYS.items():
        values = np.asarray(summary4[model][key])
        counts = counts_for(values, len(FOUR_METHOD_LABELS))
        n = int(counts.sum())
        print(f"  {label}  (n = {n})")
        for m, name in enumerate(FOUR_METHOD_LABELS):
            k = int(counts[m])
            p_hat = k / n
            cl, cu = clopper_pearson_ci(k, n)
            wl, wu = wilson_ci(k, n)
            p = binomtest(k, n, 0.25, alternative="two-sided").pvalue
            print(
                f"      {name:<20s} {k:>2d}/{n} best, {p_hat:.3f} "
                f"[{cl:.3f}, {cu:.3f}],  p = {p:.3f}"
            )
            rows.append(
                dict(
                    figure="Fig5D",
                    metric=label,
                    condition=name,
                    wins=k,
                    n=n,
                    proportion=p_hat,
                    wilson_lo=wl,
                    wilson_hi=wu,
                    cp_lo=cl,
                    cp_hi=cu,
                    p_value=p,
                    null_p=0.25,
                )
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"preference_stats.{args.format}"
    write_table(rows, out_path)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
