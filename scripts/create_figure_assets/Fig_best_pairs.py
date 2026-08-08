"""Paper/report figure: target-reconstruction pairs ordered by perceptual distance —
the best-recovered images of a run, optionally contrasted with the worst ones.

Reads the per-pair CSV written by distance_distribution_general.py
(columns: group,true,recon,matched_distance,z_col,ident_col), keeps one group
(= subject), sorts by matched_distance and draws:

    row 0 = target (true) image
    row 1 = reconstruction
    columns = the --top smallest-distance pairs, then (if --bottom > 0) a gap and
              the --bottom largest-distance pairs

--top <= 0 takes every pair; --per_row N wraps them into stacked target/recon blocks.

Run:
  # 5 best vs 3 worst side by side
  uv run python scripts/create_figure_assets/Fig_best_pairs.py \
      --csv results/rep_recon_image_koide-majima_comparing_SGD_updated_sampling_parameters/original_all/distance_summary_dreamsim/distances_dreamsim.csv \
      --group S2 --top 5 --bottom 3 --width_mm 170

  # all 25 pairs, 5 per row, ranked best -> worst
  uv run python scripts/create_figure_assets/Fig_best_pairs.py \
      --csv .../distance_summary_dreamsim/distances_dreamsim.csv \
      --group S2 --top 0 --per_row 5 --width_mm 170
"""
import os
import csv as csvmod
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

MM = 1.0 / 25.4  # mm -> inch


def is_natural(name):
    """True for the natural-image stimuli (Stim16..Stim25), whose 2nd token is an ImageNet id."""
    parts = os.path.splitext(name)[0].split("_")
    return len(parts) > 1 and parts[1].startswith("n") and parts[1][1:].isdigit()


def short_label(name):
    """Stim22_n02882301_14188.tiff -> 'Stim22'; Stim09_blue_X.tiff -> 'Stim09\\nblue_X'.

    The descriptive part goes on its own line so long geometric names
    (green_smallring) do not run into the neighbouring column.
    """
    stem = os.path.splitext(name)[0]
    parts = stem.split("_")
    if len(parts) > 1 and parts[1].startswith("n") and parts[1][1:].isdigit():
        return parts[0]                      # natural image: ImageNet id is noise
    return parts[0] + "\n" + "_".join(parts[1:]) if len(parts) > 1 else stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="per-pair CSV (group,true,recon,matched_distance,...)")
    ap.add_argument("--group", default=None, help="group/subject to plot (default: all rows)")
    ap.add_argument("--top", type=int, default=4,
                    help="number of smallest-distance pairs (<=0 or 'all' -> every pair)")
    ap.add_argument("--bottom", type=int, default=0,
                    help="also show this many largest-distance pairs, after a gap")
    ap.add_argument("--per_row", type=int, default=0,
                    help="wrap into blocks of this many pairs (0 = one single row of pairs)")
    ap.add_argument("--subset", default="all", choices=["all", "natural", "geometric"],
                    help="restrict ranking to natural (Stim16-25) or geometric (Stim01-15) stimuli")
    ap.add_argument("--true_dir", default="data/ImageryDeeprecon/source")
    ap.add_argument("--recon_dir",
                    default=("results/rep_recon_image_koide-majima_comparing_SGD_updated_sampling_parameters"
                             "/original_all/{group}/VC"),
                    help="recon image dir; '{group}' is substituted with --group")
    ap.add_argument("--metric_label", default=None, help="distance name shown in titles (default: from CSV name)")
    ap.add_argument("--width_mm", type=float, default=210.0, help="figure width in mm (210 = A4 width)")
    ap.add_argument("--font_pt", type=float, default=8.0,
                    help="smallest text size in the figure (column titles / row labels)")
    ap.add_argument("--out", default=None, help="output path without extension (default: next to CSV)")
    args = ap.parse_args()

    metric = args.metric_label or os.path.splitext(os.path.basename(args.csv))[0].replace("distances_", "")

    with open(args.csv) as f:
        rows = [r for r in csvmod.DictReader(f)
                if args.group is None or r["group"] == args.group]
    if not rows:
        raise SystemExit(f"[error] no rows for group={args.group} in {args.csv}")
    if args.subset != "all":
        want = args.subset == "natural"
        rows = [r for r in rows if is_natural(r["true"]) == want]
        if not rows:
            raise SystemExit(f"[error] no {args.subset} stimuli in {args.csv}")
    rows.sort(key=lambda r: float(r["matched_distance"]))
    n_top = len(rows) if args.top <= 0 else args.top
    best = rows[:n_top]
    worst = rows[len(rows) - args.bottom:] if args.bottom > 0 else []
    if len(best) + len(worst) > len(rows):
        raise SystemExit(f"[error] --top {n_top} + --bottom {args.bottom} exceeds {len(rows)} pairs")
    print(f"[info] {args.group or 'all'}: best-{len(best)}"
          + (f" / worst-{len(worst)}" if worst else "") + f" by {metric}")
    for r in best + worst:
        print(f"  {r['true']:40s} <- {r['recon']:35s} d={float(r['matched_distance']):.4f}")

    recon_dir = args.recon_dir.format(group=args.group or "")

    # column stream: pair dicts, "gap" = spacer between the best/worst blocks, None = empty slot
    items = best + (["gap"] + worst if worst else [])
    per_row = args.per_row if args.per_row > 0 else len(items)
    chunks = [items[i:i + per_row] for i in range(0, len(items), per_row)]
    ncols = max(len(c) for c in chunks)
    chunks = [c + [None] * (ncols - len(c)) for c in chunks]
    GAP = 0.35                                   # spacer column width, in cell units
    ratios = [GAP if any(c[j] == "gap" for c in chunks) else 1.0 for j in range(ncols)]

    label_w = 0.55                               # inch reserved on the left for row labels
    fig_w = args.width_mm * MM
    cell = (fig_w - label_w) / sum(ratios)
    FS = args.font_pt                            # smallest text; headings sit one point above it
    FS_HEAD = FS + 1
    # strip heights derived from the type size so the layout follows --font_pt (inches)
    title_h = (3 * FS * 1.15 + 5) / 72           # column titles are up to 3 lines
    head_h = (FS_HEAD * 1.3 + 5) / 72            # figure header (group / metric / stimulus set)
    cap_h = (FS_HEAD * 1.3 + 4) / 72 if (worst and len(chunks) == 1) else 0.0   # block captions
    pad = 0.045 * cell                           # gutter between image cells
    block_h = title_h + 2 * cell
    fig_h = head_h + cap_h + len(chunks) * block_h + 0.02

    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = {}                                    # (chunk, col, row) -> ax, for caption placement
    for k, chunk in enumerate(chunks):
        y_top = head_h + cap_h + k * block_h + title_h   # inches from figure top to the target row
        for j, r in enumerate(chunk):
            if r is None or r == "gap":
                continue
            x = label_w + sum(ratios[:j]) * cell
            w = ratios[j] * cell - pad
            imgs = [os.path.join(args.true_dir, r["true"]), os.path.join(recon_dir, r["recon"])]
            for row_i, p in enumerate(imgs):
                y = y_top + row_i * cell
                ax = fig.add_axes([x / fig_w, 1.0 - (y + cell - pad) / fig_h,
                                   w / fig_w, (cell - pad) / fig_h])
                axes[(k, j, row_i)] = ax
                ax.imshow(Image.open(p).convert("RGB"))
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4)
                if row_i == 0:
                    ax.set_title(f"{short_label(r['true'])}\nd={float(r['matched_distance']):.3f}",
                                 fontsize=FS, linespacing=1.15, pad=2)
                if j == 0:
                    ax.set_ylabel("Target" if row_i == 0 else "Reconstruction",
                                  fontsize=FS, rotation=90, va="center", ha="center", labelpad=8)

    scope = {"all": "images", "natural": "natural images", "geometric": "geometric images"}[args.subset]
    fig.text(label_w / fig_w, 1.0 - 0.03 / fig_h,
             f"{args.group or 'all subjects'}  |  {metric}  |  {len(rows)} {scope}, ranked by distance",
             fontsize=FS_HEAD, ha="left", va="top")

    if cap_h:  # captions over the best / worst blocks (single-row layout only)
        blocks = (("closest", 0, len(best) - 1), ("farthest", len(best) + 1, len(items) - 1))
        for label, j0, j1 in blocks:
            x0 = axes[(0, j0, 0)].get_position().x0
            x1 = axes[(0, j1, 0)].get_position().x1
            fig.text((x0 + x1) / 2, 1.0 - (head_h + 0.01) / fig_h, f"{j1 - j0 + 1} {label}",
                     fontsize=FS_HEAD, ha="center", va="top")

    tag = ("all" if args.top <= 0 else f"top{len(best)}") + (f"_bottom{len(worst)}" if worst else "")
    sub = "" if args.subset == "all" else f"_{args.subset}"
    out = args.out or os.path.join(os.path.dirname(args.csv),
                                   f"best_pairs_{metric}_{args.group or 'all'}{sub}_{tag}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print(f"saved: {out}.pdf / .png  ({fig_w/MM:.0f}x{fig_h/MM:.0f} mm)")


if __name__ == "__main__":
    main()
