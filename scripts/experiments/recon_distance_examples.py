"""Gallery of the closest (and farthest) matched reconstruction-target pairs,
ranked by a perceptual metric (dreamsim / lpips). Lets us see what a "small
distance" actually looks like.

Reads the per-image CSV produced by recon_distance_distribution.py (no model
reload needed), then renders source/recon image pairs sorted by matched distance.

Run:
  uv run python scripts/experiments/recon_distance_examples.py \
      --summary_dir results/.../original_all/distance_summary \
      --recon_root  results/rep_recon_image_koide-majima_comparing_SGD_updated_sampling_parameters \
      --method original_all --topk 8
"""
import os
import re
import csv
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


def load_source_index(source_dir):
    out = {}
    for p in glob.glob(os.path.join(source_dir, "*.tiff")):
        m = re.search(r"imageryExpStim(\d+)_(.+)\.tiff", os.path.basename(p))
        if m:
            out[int(m.group(1))] = (p, m.group(2))
    return out


def recon_path(recon_root, method, subj, roi, stim_id):
    return os.path.join(recon_root, method, subj, roi,
                        f"recon_img_normalized-Img{stim_id:04d}.jpg")


def read_csv(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({"subject": r["subject"], "stim_id": int(r["stim_id"]),
                         "matched_distance": float(r["matched_distance"]),
                         "z_col": float(r.get("z_col", "nan")),
                         "z_row": float(r.get("z_row", "nan")),
                         "ident_col": float(r.get("ident_col", "nan"))})
    return rows


def render(rows, title, src_idx, recon_root, method, roi, out, fs=8):
    n = len(rows)
    cell, label_w, m_top, m_bot, gap = 0.95, 0.55, 1.05, 0.12, 0.06
    fw = label_w + n * cell + (n - 1) * gap + 0.1
    fh = m_top + 2 * cell + gap + m_bot
    fig = plt.figure(figsize=(fw, fh))
    for c, row in enumerate(rows):
        subj, sid, d = row["subject"], row["stim_id"], row["matched_distance"]
        spath, lab = src_idx[sid]
        rpath = recon_path(recon_root, method, subj, roi, sid)
        left = label_w + c * (cell + gap)
        for r, img_path in enumerate([spath, rpath]):
            b = m_bot + (1 - r) * (cell + gap)
            ax = fig.add_axes([left / fw, b / fh, cell / fw, cell / fh])
            ax.imshow(Image.open(img_path).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                ax.set_title(f"{subj} #{sid}\n{lab}\nd={d:.3f}  z={row['z_col']:+.2f}\n"
                             f"id={row['ident_col']:.2f}", fontsize=fs, pad=2)
    # row labels
    for r, txt in enumerate(["Source", "Recon"]):
        b = m_bot + (1 - r) * (cell + gap)
        fig.text((label_w - 0.06) / fw, (b + cell / 2) / fh, txt,
                 fontsize=fs + 1, ha="right", va="center")
    fig.text(0.5, 1 - 0.16 / fh, title, fontsize=fs + 2, ha="center", va="top")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"saved {out}.png/.pdf  ({fw/0.03937:.0f}x{fh/0.03937:.0f} mm)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_dir", required=True, help="dir with distances_{metric}.csv")
    ap.add_argument("--recon_root",
                    default="results/rep_recon_image_koide-majima_comparing_SGD_updated_sampling_parameters")
    ap.add_argument("--method", default="original_all")
    ap.add_argument("--roi", default="VC")
    ap.add_argument("--source_dir", default="data/source")
    ap.add_argument("--metrics", default="dreamsim,lpips")
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--also_farthest", action="store_true", help="also render the farthest pairs")
    ap.add_argument("--per_subject", action="store_true",
                    help="rank and render within each subject separately")
    ap.add_argument("--rank_by", default="z_col", choices=["distance", "z_col", "ident_col"],
                    help="ranking key: distance=raw (confounded by image difficulty); "
                         "z_col=matched z-scored vs distractor recons on the same source "
                         "(controls source difficulty); ident_col=per-image identification rate")
    args = ap.parse_args()

    # best (most-reconstructed) first; for distance/z_col lower is better, for ident higher is better
    asc = args.rank_by != "ident_col"
    label = {"distance": "raw distance", "z_col": "difficulty-normalized z",
             "ident_col": "per-image identification"}[args.rank_by]
    out_tag = {"distance": "closest", "z_col": "bestz", "ident_col": "bestid"}[args.rank_by]

    src_idx = load_source_index(args.source_dir)
    for metric in [m.strip() for m in args.metrics.split(",") if m.strip()]:
        rows = read_csv(os.path.join(args.summary_dir, f"distances_{metric}.csv"))
        if args.per_subject:
            subjects = sorted({r["subject"] for r in rows})
            groups = [(s, [r for r in rows if r["subject"] == s]) for s in subjects]
        else:
            groups = [(None, rows)]
        for subj, grp in groups:
            grp = sorted(grp, key=lambda r: r[args.rank_by], reverse=not asc)
            k = min(args.topk, len(grp))
            sfx = f"_{subj}" if subj else ""
            ttl = f"{subj}  " if subj else ""
            render(grp[:k],
                   f"{ttl}{metric}: best {k} by {label} (d=raw dist, z=norm, id=per-image identification)",
                   src_idx, args.recon_root, args.method, args.roi,
                   os.path.join(args.summary_dir, f"examples_{out_tag}_{metric}{sfx}"))
            if args.also_farthest:
                render(grp[-k:][::-1],
                       f"{ttl}{metric}: worst {k} by {label}",
                       src_idx, args.recon_root, args.method, args.roi,
                       os.path.join(args.summary_dir, f"examples_{out_tag}_worst_{metric}{sfx}"))


if __name__ == "__main__":
    main()
