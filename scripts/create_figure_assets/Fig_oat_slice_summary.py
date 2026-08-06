"""Combined summary of the OAT sweep and the lr_a x T interaction slice.

One figure that tells the whole sampling-parameter story:

  A  the lr_a x T plane (5x5) coloured by the null - matched DreamSim gap.
     OAT only tested the cross through this plane (row lr_a=ref, column T=ref);
     the slice filled the interior, including the diagonal where lr_a and T rise
     together -- the only direction in which SGLD could genuinely sample. The
     reference (Koide-Majima) cell is ringed; OAT-covered cells are hatched.
  B  every *other* OAT knob (lr_b, lr_gamma, numReps_withLangevin,
     numReps_withoutLangevin) as a gap dot plot against the reference line,
     showing each leaves reconstruction quality unchanged (except woL=0, which
     removes the reconstruction phase and collapses to chance).

Reads the per-condition DreamSim matrices produced by
``scripts/experiments/oat_dreamsim_matrices.py`` for both runs.

    python scripts/create_figure_assets/Fig_oat_slice_summary.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from figure_asset_utils import ensure_directory, project_root

PROJECT_ROOT = project_root()
DEFAULT_OAT_DIR = PROJECT_ROOT / "results" / "oat_sampling_params" / "dreamsim_matrices"
DEFAULT_SLICE_DIR = PROJECT_ROOT / "results" / "lr_a_T_slice" / "dreamsim_matrices"
DEFAULT_OUT = PROJECT_ROOT / "assets" / "oat_sampling_params" / "oat_slice_summary"

N = 25
REFERENCE_TAG = "lr_a0.00015_lr_b0.15_g0.055_T1e-06_woL1000_wL500_nR1000"
REF_LR_A, REF_T = 0.00015, 1e-06

FIELD_RE = re.compile(
    r"lr_a(?P<lr_a>[\d.e+-]+)_lr_b(?P<lr_b>[\d.e+-]+)_g(?P<g>[\d.e+-]+)"
    r"_T(?P<T>[\d.e+-]+)_woL(?P<woL>\d+)_wL(?P<wL>\d+)_")

# Human-readable name + reference value for the non-(lr_a, T) OAT knobs.
OTHER_PARAMS = {
    "lr_b": ("lr_b", 0.15),
    "g": ("lr_gamma", 0.055),
    "wL": ("numReps_withLangevin", 500),
    "woL": ("numReps_withoutLangevin", 1000),
}


def parse_fields(tag: str) -> dict[str, float] | None:
    m = FIELD_RE.match(tag)
    if not m:
        return None
    return {k: float(v) for k, v in m.groupdict().items()}


def gap_of(path: Path) -> tuple[float, float, float]:
    """Return (gap_mean, gap_sem, pixel_sd_mean) for one condition .npz."""
    off_mask = ~np.eye(N, dtype=bool)
    with np.load(path) as h:
        subs = [k[2:] for k in h.files if k.startswith("M_")]
        matched = np.concatenate([np.diag(h[f"M_{s}"]) for s in subs])
        null = np.concatenate([h[f"M_{s}"][off_mask].reshape(N, N - 1).mean(axis=1)
                               for s in subs])
        sd = np.concatenate([h[f"pixel_sd_{s}"] for s in subs])
    gap = null - matched
    return gap.mean(), gap.std(ddof=1) / np.sqrt(gap.size), sd.mean()


def load_dir(matrix_dir: Path) -> dict[str, tuple[float, float, float]]:
    return {p.stem: gap_of(p) for p in sorted(matrix_dir.glob("*.npz"))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oat-dir", type=Path, default=DEFAULT_OAT_DIR)
    parser.add_argument("--slice-dir", type=Path, default=DEFAULT_SLICE_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    oat = load_dir(args.oat_dir)
    slc = load_dir(args.slice_dir)
    if not oat or not slc:
        raise FileNotFoundError("missing matrices; run oat_dreamsim_matrices.py for both runs")

    # --- assemble the lr_a x T plane from the slice (falling back to OAT) ---
    lr_a_vals = sorted({parse_fields(t)["lr_a"] for t in slc})
    t_vals = sorted({parse_fields(t)["T"] for t in slc})
    gap_grid = np.full((len(lr_a_vals), len(t_vals)), np.nan)
    sd_grid = np.full_like(gap_grid, np.nan)
    oat_covered = np.zeros_like(gap_grid, dtype=bool)
    for tag, (g, _, sd) in {**oat, **slc}.items():
        f = parse_fields(tag)
        # only the pure lr_a/T variations belong on the plane
        if f is None or f["lr_b"] != 0.15 or f["g"] != 0.055 or f["woL"] != 1000 or f["wL"] != 500:
            continue
        if f["lr_a"] not in lr_a_vals or f["T"] not in t_vals:
            continue
        i, j = lr_a_vals.index(f["lr_a"]), t_vals.index(f["T"])
        gap_grid[i, j] = g
        sd_grid[i, j] = sd
        if f["lr_a"] == REF_LR_A or f["T"] == REF_T:  # OAT tested only the cross
            oat_covered[i, j] = True

    # The reference condition was run in BOTH sweeps (no fixed seed), so its two
    # gap values bracket the run-to-run noise floor. Show that band, not a line:
    # nothing beats it means nothing beats noise.
    ref_oat = oat[REFERENCE_TAG][0]
    ref_slice = slc[REFERENCE_TAG][0]
    ref_lo, ref_hi = sorted((ref_oat, ref_slice))
    ref_gap = ref_slice  # panel A plane is the slice run

    # --- collect the other OAT knobs for panel B ---
    rows = []
    for tag, (g, sem, sd) in oat.items():
        f = parse_fields(tag)
        if f is None:
            continue
        diffs = {k for k in ("lr_b", "g", "wL", "woL")
                 if f[k] != {"lr_b": 0.15, "g": 0.055, "wL": 500, "woL": 1000}[k]}
        # skip reference and lr_a/T variations (those are the plane)
        if len(diffs) != 1 or f["lr_a"] != REF_LR_A or f["T"] != REF_T:
            continue
        param = diffs.pop()
        name, _ = OTHER_PARAMS[param]
        rows.append({"param": name, "value": f[param], "gap": g, "sem": sem})
    other = pd.DataFrame(rows).sort_values(["param", "value"])

    # ------------------------------------------------------------------ plot
    fig = plt.figure(figsize=(15.5, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.42,
                          left=0.07, right=0.975, top=0.83, bottom=0.14)

    # --- A: lr_a x T plane ---
    ax = fig.add_subplot(gs[0, 0])
    # rows top->bottom = large lr_a -> small; flip for display
    disp = gap_grid[::-1]
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=np.nanmax(gap_grid),
                   interpolation="nearest")
    ax.set_xticks(range(len(t_vals)))
    ax.set_xticklabels([f"{v:g}" for v in t_vals], fontsize=9)
    ax.set_yticks(range(len(lr_a_vals)))
    ax.set_yticklabels([f"{v:g}" for v in lr_a_vals[::-1]], fontsize=9)
    ax.set_xlabel("T (temperature)", fontsize=10)
    ax.set_ylabel("lr_a (step-size scale)", fontsize=10)
    ax.set_title("A  lr_a x T plane: DreamSim gap (null - matched)\n"
                 f"hatched = OAT (the cross); interior = slice; reference={ref_slice:.3f}",
                 loc="left", fontsize=10.5)
    thr = np.nanmax(gap_grid) * 0.6
    for i in range(len(lr_a_vals)):
        di = len(lr_a_vals) - 1 - i  # display row
        for j in range(len(t_vals)):
            v = gap_grid[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, di, f"{v:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if v > thr else "#22252a")
            if oat_covered[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, di - 0.5), 1, 1, fill=False,
                                           hatch="////", edgecolor="#4a4f57", lw=0.0))
            if lr_a_vals[i] == REF_LR_A and t_vals[j] == REF_T:
                ax.add_patch(plt.Rectangle((j - 0.5, di - 0.5), 1, 1, fill=False,
                                           edgecolor="#c2405a", lw=2.6))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="distance gap")
    ax.legend(handles=[
        Patch(facecolor="white", edgecolor="#4a4f57", hatch="////", label="OAT-tested"),
        Patch(facecolor="none", edgecolor="#c2405a", lw=2.6, label="reference"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)

    # --- B: the other OAT knobs ---
    ax = fig.add_subplot(gs[0, 1])
    ax.axvspan(ref_lo, ref_hi, color="#c2405a", alpha=0.15, zorder=0,
               label=f"reference band ({ref_lo:.3f}-{ref_hi:.3f}, both runs)")
    param_colors = {"lr_b": "#3b6fd4", "lr_gamma": "#e07b39",
                    "numReps_withLangevin": "#2f9e6f", "numReps_withoutLangevin": "#a154c4"}
    ypos, ylabels = [], []
    y = 0
    for name in ["lr_b", "lr_gamma", "numReps_withLangevin", "numReps_withoutLangevin"]:
        sub = other[other.param == name]
        for _, r in sub.iterrows():
            ax.errorbar(r["gap"], y, xerr=r["sem"], fmt="o", ms=7, color=param_colors[name],
                        ecolor="#4a4f57", lw=1.1, capsize=2.5, zorder=2)
            ylabels.append(f"{name}={r['value']:g}")
            ypos.append(y)
            y += 1
        y += 0.6  # gap between parameter groups
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=8.5)
    ax.set_ylim(y - 0.4, -0.8)
    ax.set_xlabel("DreamSim gap (null - matched)", fontsize=10)
    ax.set_title("B  Every other OAT knob\n(all sit at the reference; woL=0 = chance)",
                 loc="left", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e2e5e9", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    best_i, best_j = np.unravel_index(np.nanargmax(gap_grid), gap_grid.shape)
    fig.suptitle(
        "Sampling parameters do not improve reconstruction "
        f"(best cell gap {gap_grid[best_i, best_j]:.3f} at lr_a={lr_a_vals[best_i]:g}, "
        f"T={t_vals[best_j]:g}; reference {ref_lo:.3f}-{ref_hi:.3f} across two runs = noise floor)",
        fontsize=11.5, x=0.07, ha="left")

    ensure_directory(args.out.parent)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    print(f"wrote {args.out}.png / .pdf")

    # tidy CSV of the plane
    plane = pd.DataFrame(
        [{"lr_a": lr_a_vals[i], "T": t_vals[j], "gap": gap_grid[i, j],
          "pixel_sd": sd_grid[i, j], "oat_tested": bool(oat_covered[i, j])}
         for i in range(len(lr_a_vals)) for j in range(len(t_vals))
         if np.isfinite(gap_grid[i, j])])
    plane.to_csv(f"{args.out}_plane.csv", index=False)
    other.to_csv(f"{args.out}_other_params.csv", index=False)
    print(plane.to_string(index=False))


if __name__ == "__main__":
    main()
