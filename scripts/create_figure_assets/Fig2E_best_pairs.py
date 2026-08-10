"""Paper/report figure: target-reconstruction pairs ordered by perceptual distance —
the best-recovered images of a run, optionally contrasted with the worst ones.

Reads the per-image CSV written by scripts/experiments/recon_distance_distribution.py
(columns: subject,stim_id,matched_distance,...,z_col,ident_col), keeps one subject,
sorts by matched_distance and draws:

    row 0 = target (true) image
    row 1 = reconstruction
    columns = the --top smallest-distance pairs, then (if --bottom > 0) a gap and
              the --bottom largest-distance pairs

The CSV identifies a stimulus by its integer id, so the two image paths are derived
from it: the target from --true_dir (imageryExpStim{id}_*.tiff) and the reconstruction
from --recon_dir (recon_img_normalized-Img{id:04d}.jpg). --recon_dir defaults to the
run the CSV came out of, which is its parent-of-parent directory.

--top <= 0 takes every pair; --per_row N wraps them into stacked target/recon blocks.

Run:
  # 5 best vs 3 worst side by side
  uv run python scripts/create_figure_assets/Fig2E_best_pairs.py \
      --csv results/<run>/original_all/distance_summary/distances_dreamsim.csv \
      --subject S2 --top 5 --bottom 3 --width_mm 170

  # all 25 pairs, 5 per row, ranked best -> worst
  uv run python scripts/create_figure_assets/Fig2E_best_pairs.py \
      --csv results/<run>/original_all/distance_summary/distances_dreamsim.csv \
      --subject S2 --top 0 --per_row 5 --width_mm 170
"""
import os
import re
import csv as csvmod
import glob
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from figure_asset_utils import ensure_directory, project_root, resolve_data_dir

MM = 1.0 / 25.4  # mm -> inch

# Stimuli 1-15 are the geometric shapes, 17-26 the natural images (16 is the
# fixation cross, which is never a reconstruction target).
NATURAL_MIN_ID = 17


def is_natural(stim_id):
    return stim_id >= NATURAL_MIN_ID


def short_label(name):
    """imageryExpStim22_inat_airliner.tiff -> 'Stim22\\nairliner'.

    The descriptive part goes on its own line so long geometric names
    (green_smallring) do not run into the neighbouring column. The anat/inat
    prefix carries no information for the reader and is dropped.
    """
    parts = os.path.splitext(name)[0].split("_")
    number = parts[0].replace("imageryExpStim", "Stim")
    rest = [p for p in parts[1:] if p not in ("anat", "inat")]
    return number + "\n" + "_".join(rest) if rest else number


def load_source_index(source_dir):
    """stim id -> target image path, matching recon_distance_distribution.py."""
    out = {}
    for path in glob.glob(os.path.join(source_dir, "*.tiff")):
        m = re.search(r"imageryExpStim(\d+)", os.path.basename(path))
        if m:
            out[int(m.group(1))] = path
    return out


def read_csv(path, subject):
    """Rows for one subject, as {stim_id, matched_distance, z_col, ident_col}."""
    with open(path) as f:
        reader = csvmod.DictReader(f)
        cols = reader.fieldnames or []
        required = {"subject", "stim_id", "matched_distance"}
        if not required.issubset(cols):
            raise SystemExit(
                f"[error] {path} has columns {cols}; expected at least {sorted(required)}. "
                "Regenerate it with scripts/experiments/recon_distance_distribution.py."
            )
        return [{"stim_id": int(r["stim_id"]),
                 "matched_distance": float(r["matched_distance"]),
                 "z_col": float(r.get("z_col", "nan")),
                 "ident_col": float(r.get("ident_col", "nan"))}
                for r in reader if subject is None or r["subject"] == subject]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="per-image CSV from recon_distance_distribution.py "
                         "(subject,stim_id,matched_distance,...)")
    ap.add_argument("--subject", default=None, help="subject to plot (default: all rows)")
    ap.add_argument("--top", type=int, default=4,
                    help="number of smallest-distance pairs (<=0 or 'all' -> every pair)")
    ap.add_argument("--bottom", type=int, default=0,
                    help="also show this many largest-distance pairs, after a gap")
    ap.add_argument("--per_row", type=int, default=0,
                    help="wrap into blocks of this many pairs (0 = one single row of pairs)")
    ap.add_argument("--subset", default="all", choices=["all", "natural", "geometric"],
                    help="restrict ranking to natural (Stim17-26) or geometric (Stim01-15) stimuli")
    ap.add_argument("--true_dir", default=None,
                    help="target stimulus dir (default: data/source, or $IMAGERY_SOURCE_DIR)")
    ap.add_argument("--recon_dir", default=None,
                    help="recon image dir; '{subject}' is substituted with --subject. "
                         "Default: the run the CSV came from, i.e. <csv>/../../{subject}/VC")
    ap.add_argument("--metric_label", default=None, help="distance name shown in titles (default: from CSV name)")
    ap.add_argument("--width_mm", type=float, default=210.0, help="figure width in mm (210 = A4 width)")
    ap.add_argument("--font_pt", type=float, default=8.0,
                    help="smallest text size in the figure (column titles / row labels)")
    ap.add_argument("--out", default=None,
                    help="output path without extension (default: under assets/fig02/)")
    args = ap.parse_args()

    metric = args.metric_label or os.path.splitext(os.path.basename(args.csv))[0].replace("distances_", "")

    src_idx = load_source_index(str(args.true_dir or resolve_data_dir()))
    rows = read_csv(args.csv, args.subject)
    if not rows:
        raise SystemExit(f"[error] no rows for subject={args.subject} in {args.csv}")
    missing = sorted({r["stim_id"] for r in rows} - set(src_idx))
    if missing:
        raise SystemExit(f"[error] no target image for stimulus ids {missing} "
                         f"in {args.true_dir or resolve_data_dir()}")
    if args.subset != "all":
        want = args.subset == "natural"
        rows = [r for r in rows if is_natural(r["stim_id"]) == want]
        if not rows:
            raise SystemExit(f"[error] no {args.subset} stimuli in {args.csv}")
    rows.sort(key=lambda r: r["matched_distance"])
    n_top = len(rows) if args.top <= 0 else args.top
    best = rows[:n_top]
    worst = rows[len(rows) - args.bottom:] if args.bottom > 0 else []
    if len(best) + len(worst) > len(rows):
        raise SystemExit(f"[error] --top {n_top} + --bottom {args.bottom} exceeds {len(rows)} pairs")
    print(f"[info] {args.subject or 'all'}: best-{len(best)}"
          + (f" / worst-{len(worst)}" if worst else "") + f" by {metric}")
    for r in best + worst:
        name = os.path.basename(src_idx[r["stim_id"]])
        print(f"  {name:40s} <- Img{r['stim_id']:04d}  d={r['matched_distance']:.4f}")

    # The CSV lives in <run>/<method>/distance_summary/, so the reconstructions it
    # describes are two levels up -- derive the directory instead of duplicating the
    # run name here, where it would go stale the moment the run is renamed.
    recon_dir_tmpl = args.recon_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(args.csv))), "{subject}", "VC")
    recon_dir = recon_dir_tmpl.format(subject=args.subject or "")

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
            source_path = src_idx[r["stim_id"]]
            recon_path = os.path.join(
                recon_dir, f"recon_img_normalized-Img{r['stim_id']:04d}.jpg")
            for row_i, p in enumerate((source_path, recon_path)):
                y = y_top + row_i * cell
                ax = fig.add_axes([x / fig_w, 1.0 - (y + cell - pad) / fig_h,
                                   w / fig_w, (cell - pad) / fig_h])
                axes[(k, j, row_i)] = ax
                ax.imshow(Image.open(p).convert("RGB"))
                ax.set_xticks([])
                ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_linewidth(0.4)
                if row_i == 0:
                    col_label = short_label(os.path.basename(source_path))
                    ax.set_title(f"{col_label}\nd={r['matched_distance']:.3f}",
                                 fontsize=FS, linespacing=1.15, pad=2)
                if j == 0:
                    ax.set_ylabel("Target" if row_i == 0 else "Reconstruction",
                                  fontsize=FS, rotation=90, va="center", ha="center", labelpad=8)

    scope = {"all": "images", "natural": "natural images", "geometric": "geometric images"}[args.subset]
    fig.text(label_w / fig_w, 1.0 - 0.03 / fig_h,
             f"{args.subject or 'all subjects'}  |  {metric}  |  {len(rows)} {scope}, ranked by distance",
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
    # Figures belong in assets/; the numbers they are drawn from stay in results/.
    out = args.out or str(ensure_directory(project_root() / "assets" / "fig02")
                          / f"best_pairs_{metric}_{args.subject or 'all'}{sub}_{tag}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print(f"saved: {out}.pdf / .png  ({fig_w/MM:.0f}x{fig_h/MM:.0f} mm)")


if __name__ == "__main__":
    main()
