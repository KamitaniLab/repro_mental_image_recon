"""Figure 4: feature-matched images (A) and pairwise identification accuracy (B).

The right-hand identification bars show the MEAN over several
independent repetition runs with error bars (run-to-run SD or SEM). Each rep is a
full re-inversion with a different seed (recovery_matrix_invert_reps.py) followed by
recovery_check_eval.py; this script aggregates their recovery_check_identification.pkl.

The left-hand example images come from ONE rep run (--root); it MUST be one of the reps
that the bars summarize so the pictures correspond to a seed in the distribution. If
--root is omitted it defaults to the first rep under --reps_root. Do NOT point it at an
unrelated single run (e.g. the old seed-42 results/recovery_nocrop25) — those source/
recovered images belong to no rep in the bar statistics.

Directory layout expected under --reps_root:
    <reps_root>/rep*/<opt_space>/recovery_check_identification.pkl

Run:
  uv run python scripts/create_figure_assets/Fig4_recon_and_identification_errorbar.py \
      --reps_root results/recovery_from_rand_images --err sd
"""
import os
import glob
import pickle
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from figure_asset_utils import ensure_directory, project_root

ROWS = ["source", "clip_vitb32", "openclip_laion", "alexnet_conv5", "alexnet_rand_conv5"]
OPT_ROWS = ROWS[1:]
ROW_LABELS = {
    "source": "Source",
    "clip_vitb32": "CLIP ViT-B/32",
    "openclip_laion": "OpenCLIP\n(LAION)",
    "alexnet_conv5": "AlexNet conv5",
    "alexnet_rand_conv5": "AlexNet conv5\n(random)",
}
EVALS = ["clip_vitb32", "clip_laion", "alexnet_conv5", "alexnet_rand_conv5", "lpips", "dreamsim"]
EVAL_LABELS = ["CLIP ViT-B/32", "OpenCLIP", "AlexNet conv5", "AlexNet conv5 (rnd)", "LPIPS", "DreamSim"]
DIAG_EVAL = {"clip_vitb32": "clip_vitb32", "openclip_laion": "clip_laion",
             "alexnet_conv5": "alexnet_conv5", "alexnet_rand_conv5": "alexnet_rand_conv5"}
MM = 1.0 / 25.4


def _load_accs(base, sp):  # base/<opt_space>/recovery_check_identification.pkl -> [acc per EVAL]
    d = pickle.load(open(os.path.join(base, sp, "recovery_check_identification.pkl"), "rb"))
    return [d[e]["acc"] for e in EVALS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None,
                    help="run used for the left example-image columns. Should be ONE of the rep runs "
                         "so the pictures correspond to a seed that the bars actually summarize. "
                         "If omitted, the first rep dir under --reps_root is used.")
    ap.add_argument("--reps_root", required=True,
                    help="parent dir of repetition runs: <reps_root>/rep*/<opt_space>/recovery_check_identification.pkl")
    ap.add_argument("--err", choices=["sd", "sem"], default="sd",
                    help="error-bar statistic across reps: sd = run-to-run spread, sem = sd/sqrt(n_reps)")
    ap.add_argument("--n_cols", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="output path without extension (default: under assets/fig04/)")
    ap.add_argument("--width_mm", type=float, default=190.0, help="total figure width (A4=210, with margins=190)")
    ap.add_argument("--fontsize", type=float, default=10.0)
    ap.add_argument("--bar_ratio", type=float, default=3.3, help="bar panel width / image cell")
    args = ap.parse_args()
    fs = args.fontsize
    # Figures belong in assets/; the numbers they summarise stay in results/.
    if args.out is None:
        args.out = str(ensure_directory(project_root() / "assets" / "fig04")
                       / "fig_recon_and_identification_errorbar")

    # Default the example-image run to the first rep, so the pictures come from a seed
    # that is part of the bar distribution (NOT an unrelated single run such as seed 42).
    if args.root is None:
        rep_candidates = sorted(glob.glob(os.path.join(args.reps_root, "rep*")))
        if not rep_candidates:
            raise SystemExit(f"no rep* dirs under {args.reps_root}")
        args.root = rep_candidates[0]
    print(f"[example images] using --root={args.root}")

    # pick example columns
    src_dir = os.path.join(args.root, "clip_vitb32", "source")
    n_imgs = len([f for f in os.listdir(src_dir) if f.endswith(".png")])
    cols = sorted(np.random.default_rng(args.seed).choice(n_imgs, args.n_cols, replace=False).tolist())

    # aggregate identification accuracy over reps: mean and error per (opt_space, eval)
    accs, errs, n_reps = {}, {}, {}
    for sp in OPT_ROWS:
        rep_dirs = sorted(d for d in glob.glob(os.path.join(args.reps_root, "rep*"))
                          if os.path.exists(os.path.join(d, sp, "recovery_check_identification.pkl")))
        if not rep_dirs:
            raise SystemExit(f"no rep*/{sp}/recovery_check_identification.pkl under {args.reps_root}")
        A = np.array([_load_accs(rd, sp) for rd in rep_dirs])  # (n_reps, n_evals)
        accs[sp] = A.mean(0)
        sd = A.std(0, ddof=1) if A.shape[0] > 1 else np.zeros(A.shape[1])
        errs[sp] = sd / np.sqrt(A.shape[0]) if args.err == "sem" else sd
        n_reps[sp] = A.shape[0]
        print(f"[{sp}] n_reps={A.shape[0]}  diag mean={accs[sp][EVALS.index(DIAG_EVAL[sp])]:.3f} "
              f"±{errs[sp][EVALS.index(DIAG_EVAL[sp])]:.3f} ({args.err})")

    # ---- absolute layout in inches; total width fixed (A4), cell derived ----
    nrows, nimg = len(ROWS), args.n_cols
    fig_w = args.width_mm * MM
    label_w, row_gap = 1.02, 0.06       # wide enough for horizontal row labels
    gap_ib, m_right = 0.48, 0.08        # wider gap so bar y-tick labels clear the images
    m_top, m_bot = 0.30, 0.95           # room for column titles / rotated x labels at 10pt
    cell = (fig_w - label_w - gap_ib - m_right) / (nimg + args.bar_ratio)
    bar_w = args.bar_ratio * cell
    fig_h = m_top + nrows * cell + (nrows - 1) * row_gap + m_bot
    fig = plt.figure(figsize=(fig_w, fig_h))

    def row_bottom_in(r):  # r=0 top ... r=nrows-1 bottom
        return m_bot + (nrows - 1 - r) * (cell + row_gap)

    def add_ax(left_in, bottom_in, w_in, h_in):
        return fig.add_axes([left_in / fig_w, bottom_in / fig_h, w_in / fig_w, h_in / fig_h])

    x = np.arange(len(EVALS))
    for r, row in enumerate(ROWS):
        b = row_bottom_in(r)
        # ---- left: image cells ----
        sub = "source" if row == "source" else "recovered"
        d_img = os.path.join(args.root, "clip_vitb32" if row == "source" else row, sub)
        for c, idx in enumerate(cols):
            ax = add_ax(label_w + c * cell, b, cell, cell)
            ax.imshow(Image.open(os.path.join(d_img, f"{idx:02d}.png")).convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if r == 0:
                ax.set_title(f"#{idx}", fontsize=fs, pad=2)
        # horizontal row label, right-aligned just left of the image block
        fig.text((label_w - 0.07) / fig_w, (b + cell / 2) / fig_h, ROW_LABELS[row],
                 fontsize=fs, ha="right", va="center")

        # ---- right: bar panel (skip source row) ----
        bx_left = label_w + nimg * cell + gap_ib
        if row == "source":
            fig.text((bx_left + bar_w / 2) / fig_w, (b + cell * 0.5) / fig_h,
                     f"Pairwise identification\naccuracy (chance=0.5)\n"
                     f"mean±{args.err.upper()}, n={max(n_reps.values())} reps",
                     fontsize=fs, ha="center", va="center")
            continue
        bx = add_ax(bx_left, b, bar_w, cell)
        diag_i = EVALS.index(DIAG_EVAL[row])
        colors = ["0.6"] * len(EVALS)
        colors[diag_i] = "#c0392b"
        bx.bar(x, accs[row], yerr=errs[row], color=colors, width=0.74, edgecolor="black", linewidth=0.3,
               error_kw=dict(ecolor="black", elinewidth=0.6, capsize=2, capthick=0.6))
        bx.axhline(0.5, color="black", lw=0.5, ls="--", dashes=(4, 3))
        bx.set_ylim(0.4, 1.10)
        bx.set_yticks([0.5, 1.0])
        bx.tick_params(axis="y", labelsize=fs, length=2, pad=1)
        for s in ("top", "right"):
            bx.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            bx.spines[s].set_linewidth(0.5)
        bx.text(diag_i, accs[row][diag_i] + errs[row][diag_i] + 0.02, f"{accs[row][diag_i]:.2f}",
                ha="center", va="bottom", fontsize=fs, color="#c0392b")
        bx.set_xticks(x)
        if row == OPT_ROWS[-1]:
            bx.set_xticklabels(EVAL_LABELS, fontsize=fs, rotation=30, ha="right", rotation_mode="anchor")
        else:
            bx.set_xticklabels([])

    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=300)
    print(f"saved: {args.out}.pdf / .png  ({fig_w/MM:.0f}x{fig_h/MM:.0f} mm), cols={cols}, "
          f"reps={max(n_reps.values())} (err={args.err})")


if __name__ == "__main__":
    main()
