"""Condition summary of the OAT sampling-parameter sweep, in DreamSim distance.

Consumes the per-condition ``.npz`` matrices from
``scripts/experiments/oat_dreamsim_matrices.py`` and shows both readings of the
same DreamSim distances side by side:

  A  raw matched distance, condition x reconstruction (75 columns);
  B  correlation between condition profiles -> the sweep's distinct behaviours;
  C  matched vs null distance per condition (raw, both on one axis);
  D  the null - matched gap, i.e. the same distance baselined against chance.

C and D are the "both" views: a condition can score a low raw distance simply by
producing an output that is close to *every* target (a near-flat image), which
only the gap in D exposes.

    python scripts/create_figure_assets/Fig_oat_dreamsim_matrix.py
    python scripts/create_figure_assets/Fig_oat_dreamsim_matrix.py --n-clusters 3
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist

from figure_asset_utils import ensure_directory, project_root

PROJECT_ROOT = project_root()
DEFAULT_MATRIX_DIR = PROJECT_ROOT / "results" / "oat_sampling_params" / "dreamsim_matrices"
DEFAULT_OUT = PROJECT_ROOT / "assets" / "figA5" / "dreamsim_condition_matrix"

REFERENCE_TAG = "lr_a0.00015_lr_b0.15_g0.055_T1e-06_woL1000_wL500_nR1000"
SUBJECTS = ("S1", "S2", "S3")
N = 25

# Categorical hues, fixed order -- assigned to clusters, never cycled. Chosen to
# stay clear of the red/blue RdBu correlation matrix they sit beside (no pure red
# or blue), and readable as label text on white.
CLUSTER_COLORS = ["#5e35b1", "#e07b39", "#2e9e6f", "#8d6e63", "#d81b60", "#546e7a"]
MATCHED_COLOR = "#3b6fd4"
NULL_COLOR = "#9aa0a8"

# --- parameter-type ordering (for the OAT sweep, which varies one knob at a time) ---
_FIELD_RE = re.compile(
    r"lr_a(?P<lr_a>[\d.e+-]+)_lr_b(?P<lr_b>[\d.e+-]+)_g(?P<g>[\d.e+-]+)"
    r"_T(?P<T>[\d.e+-]+)_woL(?P<woL>\d+)_wL(?P<wL>\d+)_nR")
REF_FIELDS = {"lr_a": 0.00015, "lr_b": 0.15, "g": 0.055, "T": 1e-06, "woL": 1000, "wL": 500}
# Row-group order requested: reference, then wL, lr_b, lr_gamma, T, lr_a
# (woL / multi-parameter conditions, if any, trail after).
PARAM_SEQUENCE = ["reference", "wL", "lr_b", "g", "T", "lr_a", "woL", "multi"]
PARAM_COLORS = {
    "reference": "#e07b39",  # orange, pinned top
    "wL": "#3b6fd4", "lr_b": "#2f9e6f", "g": "#a154c4",
    "T": "#c2405a", "lr_a": "#7a7f8a", "woL": "#8a6d3b", "multi": "#bbbbbb",
}
PARAM_DISPLAY = {
    "reference": "reference", "wL": "numReps_withLangevin", "lr_b": "lr_b",
    "g": "lr_gamma", "T": "T", "lr_a": "lr_a", "woL": "numReps_withoutLangevin",
    "multi": "multi-parameter",
}


def parse_fields(tag: str) -> dict[str, float] | None:
    m = _FIELD_RE.search(tag)
    return {k: float(v) for k, v in m.groupdict().items()} if m else None


def classify_condition(tag: str) -> tuple[str, float]:
    """Return (varied-parameter key, its value) relative to the reference setting."""
    f = parse_fields(tag)
    if f is None:
        return "multi", 0.0
    diffs = [k for k, ref in REF_FIELDS.items()
             if abs(f[k] - ref) > 1e-12 * max(1.0, abs(ref))]
    if not diffs:
        return "reference", 0.0
    if len(diffs) > 1:
        return "multi", 0.0
    return diffs[0], f[diffs[0]]


def short_label(tag: str, reference: str) -> str:
    """Label a condition by the single field in which it differs from the reference."""
    parts, ref_parts = tag.split("_"), reference.split("_")
    if len(parts) != len(ref_parts):
        return tag
    diffs = [p for p, r in zip(parts, ref_parts) if p != r]
    return ", ".join(diffs) if diffs else "reference"


# Display form of the tag fields: the symbols the paper uses. wL / woL are the
# Langevin and the Adam phase respectively (recon_func_mod_KS.withoutLangevin
# optimises with Adam), so they read as step counts, not weights.
TOKEN_DISPLAY = {
    "a": r"$\alpha$", "b": "b", "g": r"$\gamma$", "T": "T",
    "wL": r"$N_\mathrm{SGLD}$", "woL": r"$N_\mathrm{Adam}$",
}
_TOKEN_RE = re.compile(r"^(woL|wL|a|b|g|T)([\d.e+-]+)$")
REFERENCE_DISPLAY = "KM"


def _format_value(text: str) -> str:
    """3 significant digits (0.115533 -> 0.116), but whole numbers stay plain so
    the step counts read as 1500 rather than 1.5e+03."""
    value = float(text)
    return f"{value:.0f}" if value == int(value) else f"{value:.3g}"


def pretty_label(tag: str, reference: str) -> str:
    """short_label's diff tokens rendered with the paper's symbols.

    ``a0.01`` -> ``alpha = 0.01``, ``b0.115533`` -> ``b = 0.116``,
    ``woL0`` -> ``N_Adam = 0``; the reference condition becomes ``KM``.
    """
    raw = short_label(tag, reference)
    if raw == "reference":
        return REFERENCE_DISPLAY
    out = []
    for token in raw.split(", "):
        m = _TOKEN_RE.match(token)
        out.append(f"{TOKEN_DISPLAY[m[1]]} = {_format_value(m[2])}" if m else token)
    return ", ".join(out)


def load_conditions(matrix_dir: Path) -> dict[str, dict]:
    """Load every condition .npz into matched / null / pixel-SD summaries."""
    off_mask = ~np.eye(N, dtype=bool)
    out = {}
    for path in sorted(matrix_dir.glob("*.npz")):
        with np.load(path) as handle:
            mats = {s: handle[f"M_{s}"] for s in SUBJECTS}
            sds = np.concatenate([handle[f"pixel_sd_{s}"] for s in SUBJECTS])
        out[path.stem] = {
            # 75-value profile: the matched distance of every reconstruction.
            "matched": np.concatenate([np.diag(mats[s]) for s in SUBJECTS]),
            "null": np.concatenate([mats[s][off_mask].reshape(N, N - 1).mean(axis=1)
                                    for s in SUBJECTS]),
            "pixel_sd": sds,
        }
    if not out:
        raise FileNotFoundError(
            f"no .npz in {matrix_dir}\n"
            "Run: python scripts/experiments/oat_dreamsim_matrices.py")
    return out


def cluster_order(profiles: np.ndarray, n_clusters: int):
    """Cluster conditions by the *pattern* of their profile, not its offset."""
    link = linkage(pdist(profiles, metric="correlation"), method="average")
    order = dendrogram(link, no_plot=True)["leaves"]
    return order, fcluster(link, t=n_clusters, criterion="maxclust")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--title", default="OAT sampling-parameter sweep",
                        help="leading phrase of the figure suptitle")
    parser.add_argument("--order", choices=["param", "gap"], default="param",
                        help="'param': group rows by varied parameter "
                             "(reference, wL, lr_b, lr_gamma, T, lr_a); "
                             "'gap': reference on top then descending gap")
    args = parser.parse_args()

    conds = load_conditions(args.matrix_dir)
    tags = list(conds)
    matched = np.array([conds[t]["matched"] for t in tags])
    null = np.array([conds[t]["null"] for t in tags])

    gap_all = (null - matched).mean(axis=1)
    ref_idx = tags.index(REFERENCE_TAG) if REFERENCE_TAG in tags else int(np.argmax(gap_all))

    # Row ORDER depends on the mode; row COLOUR is always the behaviour cluster
    # (conditions that reconstruct similarly share a colour), with the reference
    # pinned to the top row and highlighted.
    if args.order == "param":
        # Group rows by which parameter is varied, in the requested sequence, and
        # order within each group by that parameter's value (ascending).
        param_of = {i: classify_condition(tags[i]) for i in range(len(tags))}
        order = sorted(
            range(len(tags)),
            key=lambda i: (PARAM_SEQUENCE.index(param_of[i][0]), param_of[i][1]))
    else:
        # reference pinned to the top, then every other condition by descending gap.
        order = [ref_idx] + sorted((i for i in range(len(tags)) if i != ref_idx),
                                   key=lambda i: gap_all[i], reverse=True)

    _, clusters = cluster_order(matched, args.n_clusters)
    clusters_ord = clusters[order]
    colors = [CLUSTER_COLORS[(c - 1) % len(CLUSTER_COLORS)] for c in clusters_ord]
    REF_COLOR = "#e07b39"
    colors[order.index(ref_idx)] = REF_COLOR  # reference highlighted

    tags_ord = [tags[i] for i in order]
    matched_ord, null_ord = matched[order], null[order]
    labels = [short_label(t, REFERENCE_TAG) for t in tags_ord]
    ypos = np.arange(len(tags_ord))

    ref_row = order.index(ref_idx)

    def label_axis(axis):
        """Row labels on the left, each tinted by its behaviour-cluster colour so
        the cluster is read from the label itself (no swatch beside the matrix);
        the reference row is additionally bold."""
        axis.set_yticks(ypos)
        axis.set_yticklabels(labels, fontsize=8)
        axis.tick_params(axis="y", length=0)
        for i, lbl in enumerate(axis.get_yticklabels()):
            lbl.set_color(colors[i])
        axis.get_yticklabels()[ref_row].set_fontweight("bold")

    fig = plt.figure(figsize=(15.5, 7.4))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.2, 0.8], wspace=0.72,
                            left=0.11, right=0.965, top=0.85, bottom=0.16)

    # --- B: condition x condition correlation ---
    # Kept square. Row labels on the LEFT, tinted by behaviour cluster (that is
    # how the cluster is shown -- no coloured stripe competing with the red/blue
    # matrix). C carries the same coloured labels so rows match between panels.
    ax = fig.add_subplot(grid[0, 0])
    corr = np.corrcoef(matched_ord)
    im_b = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title("B  Correlation between\ncondition profiles", loc="left", fontsize=11)
    ax.set_xticks([])
    label_axis(ax)
    fig.colorbar(im_b, ax=ax, orientation="horizontal", location="bottom",
                 fraction=0.05, pad=0.04, label="Pearson r")

    # --- C: raw matched vs null distance ---
    # Whiskers are +/-1 SD ACROSS the 75 reconstructions (the spread of the data),
    # NOT the SEM of the mean. This is deliberate: the matched-vs-null mean gap is
    # tiny next to how much individual reconstructions vary, so the two heavily
    # overlap -- reconstructions sit barely above chance. Using SEM would shrink
    # the whiskers ~9x and manufacture a separation that is not there per-image.
    # matched (filled, coloured) and null (open, gray) are dodged +/- a little in
    # y so their overlapping SD whiskers stay separately readable.
    ax = fig.add_subplot(grid[0, 1])
    m_mean = matched_ord.mean(axis=1)
    m_sd = matched_ord.std(axis=1, ddof=1)
    n_mean = null_ord.mean(axis=1)
    n_sd = null_ord.std(axis=1, ddof=1)
    dodge = 0.2
    ax.errorbar(n_mean, ypos + dodge, xerr=n_sd, fmt="none", ecolor=NULL_COLOR, lw=1.0,
                alpha=0.8, zorder=1)
    ax.errorbar(m_mean, ypos - dodge, xerr=m_sd, fmt="none", ecolor="#4a4f57", lw=1.0,
                alpha=0.8, zorder=2)
    ax.scatter(n_mean, ypos + dodge, facecolors="white", edgecolors=NULL_COLOR, s=40,
               linewidths=1.6, zorder=3)
    ax.scatter(m_mean, ypos - dodge, c=colors, s=42, edgecolors="white",
               linewidths=1.2, zorder=4)
    ax.set_title("C  Raw distance: matched vs null\n"
                 "filled=matched, open=null (+/-1 SD)",
                 loc="left", fontsize=10.5)
    ax.set_xlabel("DreamSim distance", fontsize=9)

    # --- D: null - matched gap ---
    ax_d = fig.add_subplot(grid[0, 2])
    gap = null_ord - matched_ord
    g_mean = gap.mean(axis=1)
    g_sem = gap.std(axis=1, ddof=1) / np.sqrt(gap.shape[1])
    ax_d.axvline(0, color="#4a4f57", lw=1.0, zorder=0)
    ax_d.errorbar(g_mean, ypos, xerr=g_sem, fmt="none", ecolor="#4a4f57", lw=1.1, zorder=1)
    ax_d.scatter(g_mean, ypos, c=colors, s=46, edgecolors="white", linewidths=1.2, zorder=2)
    ax_d.set_title("D  Baselined:\nnull - matched (+/-1 SEM)", loc="left", fontsize=10.5)
    ax_d.set_xlabel("distance gap", fontsize=9)

    for axis in (ax, ax_d):
        axis.set_ylim(len(tags_ord) - 0.5, -0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color="#e2e5e9", lw=0.6)
        axis.set_axisbelow(True)
    label_axis(ax)       # C keeps row labels (keys it to B); D stays unlabelled
    ax_d.set_yticks([])

    ref_handle = Line2D([], [], marker="o", ls="none", markersize=8, color=REF_COLOR,
                        markeredgecolor="#22252a", label="reference (top row)")
    null_handle = Line2D([], [], marker="o", ls="none", markersize=8, markerfacecolor="white",
                         markeredgecolor=NULL_COLOR, color=NULL_COLOR, label="null (chance)")
    group_handles = [
        Line2D([], [], marker="o", ls="none", markersize=8, color=CLUSTER_COLORS[(c - 1) % 6],
               label=f"behaviour group {c} (n={int((clusters_ord == c).sum())})")
        for c in sorted(set(clusters_ord))
    ]
    subtitle = ("rows grouped by varied parameter; colour = behaviour cluster"
                if args.order == "param"
                else "rows sorted by gap; colour = behaviour cluster")
    handles = [ref_handle] + group_handles + [null_handle]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(
        f"{args.title} ({len(tags)} conditions, DreamSim)\n{subtitle}",
        fontsize=12, x=0.05, ha="left")

    ensure_directory(args.out.parent)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    print(f"wrote {args.out}.png / .pdf")

    summary = pd.DataFrame({
        "cluster": list(clusters_ord),
        "varied_param": [classify_condition(t)[0] for t in tags_ord],
        "condition": labels,
        "matched": m_mean,
        "null": n_mean,
        "gap": g_mean,
        "gap_sem": g_sem,
        "pixel_sd": [conds[t]["pixel_sd"].mean() for t in tags_ord],
        "tag": tags_ord,
    })
    summary.to_csv(f"{args.out}_summary.csv", index=False)
    print(summary.drop(columns="tag").to_string(index=False))


if __name__ == "__main__":
    main()
