"""Composite 2x2 OAT sampling-parameter figure with example reconstructions.

    A (top-left)   example reconstructions: 5 target images x 5 conditions
                   (reference + one more orange, one green, one purple, one brown)
    B (top-right)  condition x condition correlation of the matched-distance profiles
    C (bottom-left) raw matched vs null DreamSim distance (+/-1 SD)
    D (bottom-right) null - matched gap (+/-1 SEM)

B/C/D reuse the per-condition DreamSim matrices from
``scripts/experiments/oat_dreamsim_matrices.py`` (rows grouped by varied parameter,
coloured by behaviour cluster). A pulls the final reconstructions straight from the
sweep's output tree.

    python scripts/create_figure_assets/FigA5_oat_composite.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from figure_asset_utils import ensure_directory, load_recon_images, load_target_images, project_root
from Fig_oat_dreamsim_matrix import (
    CLUSTER_COLORS, REFERENCE_TAG, cluster_order, load_conditions, parse_fields,
    pretty_label,
)

PROJECT_ROOT = project_root()
# The OAT sweep and the lr_a x T slice are merged into one condition set: they
# share 9 conditions (the cross), so the union is 22 + 25 - 9 = 38 unique
# settings. The 16 slice-only cells vary lr_a AND T together (classify ->
# "multi"), i.e. the interaction diagonal the OAT cross never tested.
DEFAULT_MATRIX_DIRS = [
    PROJECT_ROOT / "results" / "oat_sampling_params" / "dreamsim_matrices",
    PROJECT_ROOT / "results" / "lr_a_T_slice" / "dreamsim_matrices",
]
RECON_ROOTS = {
    "oat": PROJECT_ROOT / "results" / "oat_sampling_params",
    "slice": PROJECT_ROOT / "results" / "lr_a_T_slice",
}
DEFAULT_OUT = PROJECT_ROOT / "assets" / "figA5" / "oat_composite"

N = 25
NULL_COLOR = "#9aa0a8"
REF_COLOR = "#e07b39"

# Five target images (same representative set as Fig4 / Fig5B).
IMAGE_NAMES = (
    "imageryExpStim01_red_smallring.tiff",
    "imageryExpStim08_blue_+.tiff",
    "imageryExpStim18_anat_goldfish.tiff",
    "imageryExpStim21_anat_swan.tiff",
    "imageryExpStim25_inat_stainedglass.tiff",
)
IMAGE_TITLES = ("red ring", "blue +", "goldfish", "swan", "stained glass")

# Reconstruction rows in panel A: (condition tag, ROI subpath, root key); the
# row label is derived from the tag by pretty_label, so A reads with exactly the
# same names as the B/C rows. Ordered healthy -> degraded to match the
# cluster-block order of B/C/D. Each row's frame colour is taken from the
# behaviour-cluster assignment (target = black). Interaction cells (both lr_a and
# T changed) live in the slice run.
RECON_ROWS = (
    (None, None, None),
    ("lr_a0.00015_lr_b0.15_g0.055_T1e-06_woL1000_wL500_nR1000", "VC", "oat"),
    ("lr_a0.00015_lr_b0.15_g0.055_T0.01_woL1000_wL500_nR1000", "VC", "oat"),
    ("lr_a0.1_lr_b0.15_g0.055_T0.1_woL1000_wL500_nR1000", "VC", "slice"),
    ("lr_a10_lr_b0.15_g0.055_T0.0001_woL1000_wL500_nR1000", "VC", "slice"),
    ("lr_a10_lr_b0.15_g0.055_T1e-06_woL1000_wL500_nR1000", "VC", "oat"),
    ("lr_a1_lr_b0.15_g0.055_T0.01_woL1000_wL500_nR1000", "VC", "slice"),
    ("lr_a10_lr_b0.15_g0.055_T0.1_woL1000_wL500_nR1000", "VC", "slice"),
    ("lr_a0.00015_lr_b0.15_g0.055_T1e-06_woL0_wL500_nR1000", "VC", "oat"),
)


def merge_conditions(matrix_dirs):
    """Union of per-condition summaries across runs; first run wins on overlap
    (the 9 cross conditions the OAT sweep and the slice share)."""
    merged = {}
    for d in matrix_dirs:
        for tag, summary in load_conditions(d).items():
            merged.setdefault(tag, summary)
    return merged


def build_order(tags, matched, null, n_clusters):
    """Rows ordered so each behaviour cluster is contiguous (B reads as clean
    blocks): reference cluster first, remaining clusters by size (largest first),
    and within each cluster by descending gap. reference is pinned to the very top.
    The reference cluster is painted orange; the rest take the other palette hues."""
    ref_idx = tags.index(REFERENCE_TAG)
    _, clusters = cluster_order(matched, n_clusters)
    gap_all = (null - matched).mean(1)

    ref_cluster = clusters[ref_idx]
    non_ref = [c for c in set(clusters) if c != ref_cluster]

    # Colour assignment is by size (largest cluster first) so hues stay attached
    # to the same clusters regardless of block order.
    by_size = sorted(non_ref, key=lambda c: -(clusters == c).sum())
    non_orange = [h for h in CLUSTER_COLORS if h != REF_COLOR]
    cmap = {ref_cluster: REF_COLOR}
    for c, hue in zip(by_size, non_orange):
        cmap[c] = hue

    # Block ORDER is by descending mean gap (healthy -> degraded): reference,
    # then a0.1, green, purple, woL0 fall out in that order.
    mean_gap = {c: gap_all[clusters == c].mean() for c in set(clusters)}
    by_gap = sorted(non_ref, key=lambda c: -mean_gap[c])
    rank = {ref_cluster: 0}
    for k, c in enumerate(by_gap, start=1):
        rank[c] = k

    # Manual pins at block boundaries so the gap gradient stays continuous across
    # them: a0.1 at the bottom of the orange block, a10,T0.0001 (much the highest
    # gap in the green block) at the top of the green block.
    a01_tag = "lr_a0.1_lr_b0.15_g0.055_T1e-06_woL1000_wL500_nR1000"
    a10_t0001_tag = "lr_a10_lr_b0.15_g0.055_T0.0001_woL1000_wL500_nR1000"

    def key(i):
        r = rank[clusters[i]]
        if clusters[i] == ref_cluster:
            # reference pinned to the top, a0.1 pinned to the bottom; the rest by
            # descending gap so the big orange block stays a clean uniform block.
            if i == ref_idx:
                return (r, 0, 0.0)
            if tags[i] == a01_tag:
                return (r, 2, 0.0)
            return (r, 1, -gap_all[i])
        # Other clusters: a10,T0.0001 pinned first, then lr_a -> T (a1... before
        # a10..., no temperature interleave).
        f = parse_fields(tags[i]) or {}
        pin = 0 if tags[i] == a10_t0001_tag else 1
        return (r, pin, f.get("lr_a", 0.0), f.get("T", 0.0))

    order = sorted(range(len(tags)), key=key)

    color_of_tag = {tags[i]: cmap[clusters[i]] for i in range(len(tags))}
    clusters_ord = clusters[order]
    colors = [cmap[c] for c in clusters_ord]
    return order, clusters_ord, colors, order.index(ref_idx), cmap, color_of_tag


def draw_panel_a(fig, cell, roots, subject, color_of_tag):
    """Example reconstructions grid: rows = conditions, columns = target images.

    A blank spacer row separates the target row from the reconstruction rows;
    row labels (condition names) are kept, image column names are dropped.
    """
    targets = load_target_images(list(IMAGE_NAMES))
    ncols = len(IMAGE_NAMES)
    # target row, a thin spacer, then the reconstruction rows.
    heights = [1.0, 0.32] + [1.0] * (len(RECON_ROWS) - 1)
    inner = cell.subgridspec(len(RECON_ROWS) + 1, ncols, hspace=0.06, wspace=0.06,
                             height_ratios=heights)
    for r, (tag, roi, root_key) in enumerate(RECON_ROWS):
        grid_r = r if r == 0 else r + 1  # skip the spacer row (grid index 1)
        label = "target" if tag is None else pretty_label(tag, REFERENCE_TAG)
        color = "#222528" if tag is None else color_of_tag.get(tag, "#222528")
        row_imgs = targets if tag is None else load_recon_images(
            roots[root_key] / tag / subject / roi, list(IMAGE_NAMES))
        for c, img in enumerate(row_imgs):
            ax = fig.add_subplot(inner[grid_r, c])
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(1.8 if tag is not None else 1.0)
            if c == 0:
                ax.set_ylabel(label, fontsize=9, color=color,
                              fontweight="bold" if tag == REFERENCE_TAG else "normal",
                              rotation=0, ha="right", va="center", labelpad=8)


def label_axis(axis, ypos, labels, colors, ref_row):
    axis.set_yticks(ypos)
    axis.set_yticklabels(labels, fontsize=8)
    axis.tick_params(axis="y", length=0)
    for i, lbl in enumerate(axis.get_yticklabels()):
        lbl.set_color(colors[i])
    axis.get_yticklabels()[ref_row].set_fontweight("bold")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dirs", type=Path, nargs="+", default=DEFAULT_MATRIX_DIRS,
                        help="one or more DreamSim-matrix dirs; merged by tag (first wins)")
    parser.add_argument("--subject", default="S1")
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    conds = merge_conditions(args.matrix_dirs)
    tags = list(conds)
    matched = np.array([conds[t]["matched"] for t in tags])
    null = np.array([conds[t]["null"] for t in tags])
    order, clusters_ord, colors, ref_row, cmap, color_of_tag = build_order(
        tags, matched, null, args.n_clusters)

    tags_ord = [tags[i] for i in order]
    matched_ord, null_ord = matched[order], null[order]
    labels = [pretty_label(t, REFERENCE_TAG) for t in tags_ord]
    ypos = np.arange(len(tags_ord))

    # A4 portrait width (210 mm = 8.27"); tall so the same point sizes are not
    # crowded at the reduced width.
    fig = plt.figure(figsize=(8.27, 12.6))
    # A (left) is narrow so its 9 image rows are only as tall as the square B on
    # the right; B's column is wider so the square nearly fills the top-row height.
    # The symbol labels ("alpha = 10, T = 0.0001") are wider than the raw tags
    # were, so the left margin and the A|B gutter both have to hold a full label.
    outer = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.55], height_ratios=[1.0, 1.5],
                             hspace=0.19, wspace=0.50,
                             left=0.175, right=0.985, top=0.93, bottom=0.055)

    # --- A: example reconstructions ---
    draw_panel_a(fig, outer[0, 0], RECON_ROOTS, args.subject, color_of_tag)
    fig.text(0.175, 0.94, "A  Example reconstructions",
             fontsize=12, fontweight="bold", ha="left", va="baseline")

    # --- B: condition x condition correlation ---
    ax_b = fig.add_subplot(outer[0, 1])
    corr = np.corrcoef(matched_ord)
    # aspect="auto" stretches the matrix to fill the cell height, matching panel A
    # and spreading the 38 row labels so they no longer overlap.
    im_b = ax_b.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
                       aspect="auto")
    ax_b.set_title("B  Condition correlation", loc="left", fontsize=12,
                   fontweight="bold")
    ax_b.set_xticks([])
    label_axis(ax_b, ypos, labels, colors, ref_row)
    fig.colorbar(im_b, ax=ax_b, orientation="horizontal", location="bottom",
                 fraction=0.05, pad=0.03, label="Pearson r")
    # Outline each contiguous behaviour-cluster block.
    start = 0
    for idx in range(1, len(clusters_ord) + 1):
        if idx == len(clusters_ord) or clusters_ord[idx] != clusters_ord[start]:
            ax_b.add_patch(plt.Rectangle((start - 0.5, start - 0.5), idx - start, idx - start,
                                         fill=False, edgecolor=colors[start], lw=2.2))
            start = idx

    # --- C: raw matched vs null distance (+/-1 SD, matched/null dodged) ---
    ax_c = fig.add_subplot(outer[1, 0])
    m_mean, m_sd = matched_ord.mean(1), matched_ord.std(1, ddof=1)
    n_mean, n_sd = null_ord.mean(1), null_ord.std(1, ddof=1)
    dodge = 0.2
    ax_c.errorbar(n_mean, ypos + dodge, xerr=n_sd, fmt="none", ecolor=NULL_COLOR, lw=1.0, alpha=0.8)
    ax_c.errorbar(m_mean, ypos - dodge, xerr=m_sd, fmt="none", ecolor="#4a4f57", lw=1.0, alpha=0.8)
    ax_c.scatter(n_mean, ypos + dodge, facecolors="white", edgecolors=NULL_COLOR, s=38, linewidths=1.5)
    ax_c.scatter(m_mean, ypos - dodge, c=colors, s=40, edgecolors="white", linewidths=1.1)
    # Wrapped to three lines: the bottom-row titles must not run into each other
    # (C's cell is the narrow one), and C/D carry the same number of lines so
    # both start at the same height.
    ax_c.set_title("C  Reconstruction distance:\ntarget vs non-target\n"
                   "(filled/open; +/-1 SD)",
                   loc="left", fontsize=12, fontweight="bold")
    ax_c.set_xlabel("DreamSim distance", fontsize=10)
    label_axis(ax_c, ypos, labels, colors, ref_row)

    # --- D: null - matched gap (+/-1 SEM) ---
    ax_d = fig.add_subplot(outer[1, 1])
    gap = null_ord - matched_ord
    g_mean = gap.mean(1)
    g_sem = gap.std(1, ddof=1) / np.sqrt(gap.shape[1])
    ax_d.axvline(0, color="#4a4f57", lw=1.0, zorder=0)
    ax_d.errorbar(g_mean, ypos, xerr=g_sem, fmt="none", ecolor="#4a4f57", lw=1.1, zorder=1)
    ax_d.scatter(g_mean, ypos, c=colors, s=44, edgecolors="white", linewidths=1.1, zorder=2)
    ax_d.set_title("D  Distance gap:\nnon-target - target\n(+/-1 SEM)",
                   loc="left", fontsize=12, fontweight="bold")
    ax_d.set_xlabel("distance gap", fontsize=10)
    ax_d.set_yticks([])
    # Fix the axis out to 0.1 so the gap is read on an absolute scale: even the
    # best condition (reference) sits near zero -- a large separation would be
    # needed for genuinely faithful reconstruction, and none of the conditions
    # come close.
    ax_d.set_xlim(-0.006, 0.05)

    for axis in (ax_c, ax_d):
        axis.set_ylim(len(tags_ord) - 0.5, -0.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="x", color="#e2e5e9", lw=0.6)
        axis.set_axisbelow(True)

    ensure_directory(args.out.parent)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    print(f"wrote {args.out}.png / .pdf")


if __name__ == "__main__":
    main()
