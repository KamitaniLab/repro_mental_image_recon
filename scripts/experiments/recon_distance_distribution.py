"""Systematic, quantitative summary of reconstruction quality over ALL subjects
and ALL target images (reviewer asked for more than representative examples).

For a given reconstruction method (e.g. original_all), for every subject and
every target image we compute the perceptual distance between the reconstruction
and the true target image with DreamSim and LPIPS(vgg).

Two distributions are produced per metric:
  - matched   : d(recon_i, target_i)            (same image)
  - mismatched: d(recon_i, target_j), i != j    (null / chance baseline)

If reconstructions are meaningful, matched distances should be systematically
smaller than the mismatched null. We report the full distributions, summary
stats, a Mann-Whitney U test, and AUC (= pairwise-identification accuracy).

Run:
  uv run python scripts/experiments/recon_distance_distribution.py \
      --recon_root results/rep_recon_image_koide-majima_comparing_SGD_updated_sampling_parameters \
      --method original_all
"""

import argparse
import glob
import os
import re

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_source_index(source_dir):
    """map integer stim id -> source image path (imageryExpStim{NN}_*.tiff)."""
    out = {}
    for p in glob.glob(os.path.join(source_dir, "*.tiff")):
        m = re.search(r"imageryExpStim(\d+)", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def load_recon_index(recon_dir):
    """map integer stim id -> recon image path (recon_img_normalized-Img{NNNN}.jpg)."""
    out = {}
    for p in glob.glob(os.path.join(recon_dir, "recon_img_normalized-Img*.jpg")):
        m = re.search(r"Img(\d+)", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def build_dreamsim(device):
    from dreamsim import dreamsim

    model, preprocess = dreamsim(pretrained=True, device=device)

    def dist(a, b):  # PIL, PIL -> float distance
        with torch.no_grad():
            return float(
                model(preprocess(a).to(device), preprocess(b).to(device)).cpu()
            )

    return dist


def build_lpips(device):
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from torchvision import transforms

    net = (
        LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        .to(device)
        .eval()
    )
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def dist(a, b):
        with torch.no_grad():
            return float(
                net(tf(a).unsqueeze(0).to(device), tf(b).unsqueeze(0).to(device)).cpu()
            )

    return dist


def distance_matrix(dist_fn, recons, sources):
    """M[i, j] = dist(recon_i, source_j)."""
    n = len(recons)
    M = np.zeros((n, n), dtype=np.float64)
    for i, r in enumerate(recons):
        for j, s in enumerate(sources):
            M[i, j] = dist_fn(r, s)
    return M


def summarize(matched, mismatched):
    from scipy.stats import mannwhitneyu

    matched, mismatched = np.asarray(matched), np.asarray(mismatched)
    # AUC: P(matched < mismatched) == pairwise identification accuracy
    u, p = mannwhitneyu(matched, mismatched, alternative="less")
    auc = 1.0 - u / (len(matched) * len(mismatched))  # P(matched < mismatched)
    return {
        "n_matched": len(matched),
        "n_mismatched": len(mismatched),
        "matched_mean": float(matched.mean()),
        "matched_std": float(matched.std()),
        "matched_median": float(np.median(matched)),
        "mismatched_mean": float(mismatched.mean()),
        "mismatched_std": float(mismatched.std()),
        "mismatched_median": float(np.median(mismatched)),
        "auc_identification": float(auc),
        "mannwhitney_p": float(p),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--recon_root",
        default="results/rep_recon_image_koide-majima_comparing_SGD_updated_sampling_parameters",
    )
    ap.add_argument("--method", default="original_all")
    ap.add_argument("--subjects", default="S1,S2,S3")
    ap.add_argument("--roi", default="VC")
    ap.add_argument("--source_dir", default="data/source")
    ap.add_argument("--metrics", default="dreamsim,lpips")
    ap.add_argument(
        "--id_min",
        type=int,
        default=None,
        help="only stim ids >= this (e.g. 17 for natural)",
    )
    ap.add_argument("--out_dir", default=None)
    ap.add_argument(
        "--mark_subject", default=None, help="subject to highlight stimuli on (e.g. S2)"
    )
    ap.add_argument(
        "--mark_ids",
        default=None,
        help="comma stim ids to mark on the mark_subject hist",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        args.recon_root, args.method, "distance_summary"
    )
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]

    src_idx = load_source_index(args.source_dir)
    builders = {"dreamsim": build_dreamsim, "lpips": build_lpips}
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # cache loaded source/recon PIL images per subject (RGB)
    data = {}  # subject -> (ids, recon_imgs, source_imgs)
    for subj in subjects:
        rdir = os.path.join(args.recon_root, args.method, subj, args.roi)
        rec_idx = load_recon_index(rdir)
        ids = sorted(
            i
            for i in rec_idx
            if i in src_idx and (args.id_min is None or i >= args.id_min)
        )
        recs = [Image.open(rec_idx[i]).convert("RGB") for i in ids]
        srcs = [Image.open(src_idx[i]).convert("RGB") for i in ids]
        data[subj] = (ids, recs, srcs)
        print(f"[info] {subj}: {len(ids)} image pairs (ids {ids[0]}..{ids[-1]})")

    results = {}
    for metric in metrics:
        dist_fn = builders[metric](device)
        matched_all, mismatched_all = [], []
        per_subject = {}
        per_subject_raw = {}  # subj -> (matched_arr, mismatched_arr)
        matrices = {}  # subj -> full distance matrix (saved as .npz below)
        rows = []  # per-image stats (see header below)
        for subj in subjects:
            ids, recs, srcs = data[subj]
            M = distance_matrix(dist_fn, recs, srcs)  # M[i,j] = dist(recon_i, source_j)
            matrices[subj] = M
            diag = np.diag(M)
            off = M[~np.eye(len(ids), dtype=bool)]
            matched_all.extend(diag.tolist())
            mismatched_all.extend(off.tolist())
            per_subject[subj] = summarize(diag, off)
            per_subject_raw[subj] = (diag.copy(), off.copy())
            for k, i in enumerate(ids):
                row_off = np.delete(
                    M[k], k
                )  # recon_i vs other sources (controls recon)
                col_off = np.delete(
                    M[:, k], k
                )  # other recons vs source_i (controls source difficulty)
                z_col = float((diag[k] - col_off.mean()) / (col_off.std() + 1e-9))
                z_row = float((diag[k] - row_off.mean()) / (row_off.std() + 1e-9))
                ident_col = float(
                    np.mean(diag[k] < col_off)
                )  # matched recon beats distractor recons at source_i
                ident_row = float(
                    np.mean(diag[k] < row_off)
                )  # matched source beats distractor sources for recon_i
                rows.append(
                    (
                        subj,
                        i,
                        float(diag[k]),
                        float(row_off.mean()),
                        float(col_off.mean()),
                        float(col_off.std()),
                        z_col,
                        z_row,
                        ident_col,
                        ident_row,
                    )
                )
        overall = summarize(matched_all, mismatched_all)
        results[metric] = {
            "overall": overall,
            "per_subject": per_subject,
            "per_subject_raw": per_subject_raw,
            "matched": np.asarray(matched_all),
            "mismatched": np.asarray(mismatched_all),
        }
        # write per-image CSV
        csv = os.path.join(out_dir, f"distances_{metric}.csv")
        with open(csv, "w") as f:
            f.write(
                "subject,stim_id,matched_distance,row_null_mean,col_null_mean,"
                "col_null_std,z_col,z_row,ident_col,ident_row\n"
            )
            f.writelines(
                f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f},"
                f"{r[6]:.6f},{r[7]:.6f},{r[8]:.6f},{r[9]:.6f}\n"
                for r in rows
            )
        # Full matrices alongside the CSV: the CSV keeps only the per-image summaries,
        # so anything needing the off-diagonal structure (e.g.
        # scripts/experiments/export_dreamsim_matrices_csv.py) reads these instead of
        # recomputing the distances. Row order matches the CSV rows for that subject.
        np.savez(os.path.join(out_dir, f"matrices_{metric}.npz"), **matrices)
        print(
            f"  [{metric}] matched={overall['matched_mean']:.4f}±{overall['matched_std']:.4f} "
            f"mismatched={overall['mismatched_mean']:.4f}±{overall['mismatched_std']:.4f} "
            f"AUC(id)={overall['auc_identification']:.3f} p={overall['mannwhitney_p']:.2e}"
        )

    # ---- figure: matched vs mismatched distribution per metric ----
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.0))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        m, mm = results[metric]["matched"], results[metric]["mismatched"]
        lo, hi = min(m.min(), mm.min()), max(m.max(), mm.max())
        bins = np.linspace(lo, hi, 30)
        ax.hist(
            mm,
            bins=bins,
            density=True,
            alpha=0.5,
            color="0.6",
            label=f"mismatched (null, n={len(mm)})",
        )
        ax.hist(
            m,
            bins=bins,
            density=True,
            alpha=0.6,
            color="#c0392b",
            label=f"matched (n={len(m)})",
        )
        ax.axvline(m.mean(), color="#c0392b", lw=1.3, ls="--")
        ax.axvline(mm.mean(), color="0.4", lw=1.3, ls="--")
        ov = results[metric]["overall"]
        ax.set_title(
            f"{metric}  (AUC={ov['auc_identification']:.2f}, "
            f"p={ov['mannwhitney_p']:.1e})",
            fontsize=10,
        )
        ax.set_xlabel(f"{metric} distance (lower = more similar)")
        ax.set_ylabel("density")
        ax.legend(fontsize=8, frameon=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    tag = f"{args.method}" + (f"_id{args.id_min}plus" if args.id_min else "")
    fig.suptitle(
        f"Reconstruction-target distance: {tag}, subjects={','.join(subjects)}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(out_dir, f"distance_distribution_{tag}.{ext}"), dpi=200
        )
    plt.close(fig)

    # ---- figure: per-subject histograms (rows = subjects, cols = metrics) ----
    nrow = len(subjects)
    fig, axes = plt.subplots(
        nrow, len(metrics), figsize=(5.2 * len(metrics), 2.6 * nrow), squeeze=False
    )
    for ci, metric in enumerate(metrics):
        m_all, mm_all = results[metric]["matched"], results[metric]["mismatched"]
        lo, hi = min(m_all.min(), mm_all.min()), max(m_all.max(), mm_all.max())
        bins = np.linspace(lo, hi, 26)
        for ri, subj in enumerate(subjects):
            ax = axes[ri][ci]
            m, mm = results[metric]["per_subject_raw"][subj]
            ax.hist(
                mm,
                bins=bins,
                density=True,
                alpha=0.5,
                color="0.6",
                label=f"null (n={len(mm)})",
            )
            ax.hist(
                m,
                bins=bins,
                density=True,
                alpha=0.6,
                color="#c0392b",
                label=f"matched (n={len(m)})",
            )
            ax.axvline(m.mean(), color="#c0392b", lw=1.2, ls="--")
            ax.axvline(mm.mean(), color="0.4", lw=1.2, ls="--")
            s = results[metric]["per_subject"][subj]
            ax.set_title(
                f"{subj} — {metric}  (AUC={s['auc_identification']:.2f}, "
                f"p={s['mannwhitney_p']:.1e})",
                fontsize=9,
            )
            if ri == nrow - 1:
                ax.set_xlabel(f"{metric} distance (lower = more similar)", fontsize=9)
            ax.set_ylabel("density", fontsize=9)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7.5, frameon=False)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.suptitle(f"Per-subject reconstruction-target distance: {tag}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(out_dir, f"distance_distribution_persubject_{tag}.{ext}"),
            dpi=200,
        )
    plt.close(fig)

    # ---- figure: mark specific stimuli on one subject's histogram ----
    if args.mark_subject and args.mark_ids:
        subj = args.mark_subject
        mark_ids = [int(x) for x in args.mark_ids.split(",") if x.strip()]
        ids_subj = data[subj][0]
        id_pos = {i: k for k, i in enumerate(ids_subj)}
        names = {}
        for p in glob.glob(os.path.join(args.source_dir, "*.tiff")):
            mm_ = re.search(r"imageryExpStim(\d+)_(.+)\.tiff", os.path.basename(p))
            if mm_:
                names[int(mm_.group(1))] = mm_.group(2)
        fig, axes = plt.subplots(
            len(metrics), 1, figsize=(7.2, 3.0 * len(metrics)), squeeze=False
        )
        for mi, metric in enumerate(metrics):
            ax = axes[mi][0]
            m, mm = results[metric]["per_subject_raw"][subj]
            lo, hi = min(m.min(), mm.min()), max(m.max(), mm.max())
            bins = np.linspace(lo, hi, 26)
            ax.hist(
                mm,
                bins=bins,
                density=True,
                alpha=0.5,
                color="0.6",
                label="null (other stimuli)",
            )
            ax.hist(
                m,
                bins=bins,
                density=True,
                alpha=0.55,
                color="#c0392b",
                label="matched (all)",
            )
            ymax = ax.get_ylim()[1]
            palette = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#000000", "#17becf"]
            for j, sid in enumerate(mark_ids):
                d = float(m[id_pos[sid]])
                col = palette[j % len(palette)]
                ax.axvline(d, color=col, lw=1.8)
                ax.text(
                    d,
                    ymax * (0.96 - 0.13 * j),
                    f" #{sid} {names.get(sid, '')}\n d={d:.3f}",
                    color=col,
                    fontsize=8,
                    ha="left",
                    va="top",
                    rotation=0,
                )
            s = results[metric]["per_subject"][subj]
            ax.set_title(
                f"{subj} — {metric}  (matched mean={s['matched_mean']:.3f}, "
                f"null mean={s['mismatched_mean']:.3f}, AUC={s['auc_identification']:.2f})",
                fontsize=10,
            )
            ax.set_xlabel(f"{metric} distance (lower = more similar)")
            ax.set_ylabel("density")
            if mi == 0:
                ax.legend(fontsize=8, frameon=False, loc="upper right")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.suptitle(
            f"Marked stimuli on {subj} distance distribution: {tag}", fontsize=11
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        for ext in ("pdf", "png"):
            fig.savefig(
                os.path.join(out_dir, f"distance_marked_{subj}_{tag}.{ext}"), dpi=200
            )
        plt.close(fig)
        print(f"[marked] {subj} ids={mark_ids} -> distance_marked_{subj}_{tag}")

    # ---- figure: box/violin of matched vs null (overall + per subject) ----
    # this is the view that directly matches the Mann-Whitney test (ranks/medians).
    groups = ["all"] + subjects
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics) + 1.0, 4.4))
    if len(metrics) == 1:
        axes = [axes]
    C_M, C_N = "#c0392b", "0.6"
    for ax, metric in zip(axes, metrics):
        # build pairs of (null, matched) columns per group
        positions, data_box, colors, centers, xticklab = [], [], [], [], []
        for gi, g in enumerate(groups):
            if g == "all":
                m, mm = results[metric]["matched"], results[metric]["mismatched"]
                au = results[metric]["overall"]["auc_identification"]
            else:
                m, mm = results[metric]["per_subject_raw"][g]
                au = results[metric]["per_subject"][g]["auc_identification"]
            base = gi * 3.0
            # null (left) then matched (right)
            for off, arr, col in ((0.0, mm, C_N), (1.0, m, C_M)):
                positions.append(base + off)
                data_box.append(arr)
                colors.append(col)
            centers.append(base + 0.5)
            xticklab.append(f"{g}\nAUC={au:.2f}")
        bp = ax.boxplot(
            data_box,
            positions=positions,
            widths=0.8,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black", lw=1.2),
            whiskerprops=dict(color="0.3"),
            capprops=dict(color="0.3"),
        )
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
            patch.set_edgecolor("0.2")
        # overlay matched points (jittered) since matched n is small
        rng = np.random.default_rng(0)
        for pos, arr, col in zip(positions, data_box, colors):
            if col == C_M:  # only matched (few points)
                jit = (rng.random(len(arr)) - 0.5) * 0.45
                ax.scatter(
                    pos + jit,
                    arr,
                    s=7,
                    color=col,
                    edgecolor="white",
                    linewidth=0.2,
                    zorder=3,
                    alpha=0.9,
                )
        ax.set_xticks(centers)
        ax.set_xticklabels(xticklab, fontsize=8)
        ov = results[metric]["overall"]
        ax.set_title(f"{metric}  (overall p={ov['mannwhitney_p']:.1e})", fontsize=10)
        ax.set_ylabel(f"{metric} distance (lower = more similar)", fontsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for gi in range(1, len(groups)):
            ax.axvline(gi * 3.0 - 0.6, color="0.85", lw=0.8)
    # shared legend
    from matplotlib.patches import Patch

    fig.legend(
        handles=[
            Patch(facecolor=C_N, alpha=0.55, label="mismatched (null, other stimuli)"),
            Patch(facecolor=C_M, alpha=0.55, label="matched (own target)"),
        ],
        loc="upper center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        f"Matched vs null distance: {tag}, subjects={','.join(subjects)}",
        fontsize=11,
        y=1.06,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(out_dir, f"distance_box_{tag}.{ext}"),
            dpi=200,
            bbox_inches="tight",
        )
    plt.close(fig)

    # ---- text summary ----
    with open(os.path.join(out_dir, f"summary_{tag}.txt"), "w") as f:
        f.write(
            f"method={args.method} subjects={subjects} roi={args.roi} "
            f"id_min={args.id_min} source={args.source_dir}\n\n"
        )
        for metric in metrics:
            ov = results[metric]["overall"]
            f.write(f"== {metric} ==\n")
            f.write(
                f"  matched    : mean={ov['matched_mean']:.4f} sd={ov['matched_std']:.4f} "
                f"median={ov['matched_median']:.4f} (n={ov['n_matched']})\n"
            )
            f.write(
                f"  mismatched : mean={ov['mismatched_mean']:.4f} sd={ov['mismatched_std']:.4f} "
                f"median={ov['mismatched_median']:.4f} (n={ov['n_mismatched']})\n"
            )
            f.write(
                f"  AUC(identification)={ov['auc_identification']:.4f}  "
                f"Mann-Whitney U p(matched<mismatched)={ov['mannwhitney_p']:.3e}\n"
            )
            for subj, s in results[metric]["per_subject"].items():
                f.write(
                    f"    {subj}: matched={s['matched_mean']:.4f} "
                    f"mismatched={s['mismatched_mean']:.4f} AUC={s['auc_identification']:.3f}\n"
                )
            f.write("\n")
    print(f"[done] -> {out_dir}")


if __name__ == "__main__":
    main()
