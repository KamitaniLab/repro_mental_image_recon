"""Within-participant label-permutation test for matched-vs-null reconstruction distances.

The pooled Mann-Whitney test used by recon_distance_distribution.py /
distance_distribution_general.py compares 25 matched distances against 600
non-target distances PER SUBJECT. Those 600 values are not independent: they come
from the same 25 reconstructions and the same 25 targets, each reused 24 times.
Mann-Whitney treats them as 600 independent samples, so it overstates the
effective sample size and returns an anticonservative p-value.

This script re-tests the same distance matrices with a permutation test that makes
no independence assumption. It reads the matrices_<metric>.npz written by
distance_distribution_general.py --cache (one n x n array per group,
M[i,j] = d(recon_i, target_j)) and uses

  identification accuracy   acc = mean_{i, j != i} 1[ M[i,i] < M[i,j] ]

as the statistic (pairwise 2-AFC accuracy, chance = 0.5; identical to the AUC that
the distribution scripts report). The null is built by shuffling the TARGET LABELS
within each participant -- reconstruction i is assigned target pi(i) -- which
destroys the recon-target correspondence while preserving the full dependency
structure of the matrix. Subjects are permuted independently within each draw, so
the pooled test is a valid group-level null.

  p = (1 + #{acc_perm >= acc_obs}) / (n_perm + 1)     one-sided

The existing pooled Mann-Whitney p is kept and reported next to it, so the two can
be compared directly. The distribution scripts themselves are unchanged.

Run:
  uv run python scripts/experiments/identification_permutation_test.py \
      --npz results/.../distance_summary_CORRECTED_dreamsim/matrices_dreamsim.npz \
      --n_perm 10000
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu


def identification_accuracy(M, perm=None):
    """2-AFC accuracy for one subject.

    M[i,j] = d(recon_i, target_j); perm[i] = target index assigned to recon_i
    (None = the true assignment, i.e. the diagonal).
    """
    n = M.shape[0]
    idx = np.arange(n) if perm is None else perm
    matched = M[np.arange(n), idx]
    # the assigned column compares against itself as (x < x) = False, so it drops out
    wins = (matched[:, None] < M).sum(axis=1)
    return float(wins.sum() / (n * (n - 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="matrices_<metric>.npz from a --cache run")
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None, help="tag for output files (default: metric name)")
    ap.add_argument("--out_dir", default=None, help="default: directory of --npz")
    args = ap.parse_args()

    metric = os.path.basename(args.npz).replace("matrices_", "").replace(".npz", "")
    label = args.label or metric
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.npz))
    os.makedirs(out_dir, exist_ok=True)

    z = np.load(args.npz, allow_pickle=True)
    groups = list(z.files)
    mats = {g: np.asarray(z[g], dtype=np.float64) for g in groups}
    print(f"[info] {metric}: {len(groups)} group(s) {groups}, "
          + ", ".join(f"{g}:{mats[g].shape[0]}x{mats[g].shape[0]}" for g in groups))

    rng = np.random.default_rng(args.seed)
    B = args.n_perm

    obs = {g: identification_accuracy(mats[g]) for g in groups}
    obs_pooled = float(np.mean([obs[g] for g in groups]))

    null = {g: np.empty(B) for g in groups}
    null_pooled = np.empty(B)
    for b in range(B):
        accs = []
        for g in groups:
            pi = rng.permutation(mats[g].shape[0])
            a = identification_accuracy(mats[g], pi)
            null[g][b] = a
            accs.append(a)
        null_pooled[b] = np.mean(accs)

    def perm_p(nulls, o):
        return float((1 + np.sum(nulls >= o)) / (len(nulls) + 1))

    lines = [f"label={label}  metric={metric}  groups={groups}  n_perm={B}  seed={args.seed}",
             "statistic: pairwise identification accuracy (chance 0.5)",
             "null: target labels shuffled within each participant", ""]
    rows = []
    for g in groups + ["pooled"]:
        pooled = g == "pooled"
        o = obs_pooled if pooled else obs[g]
        nl = null_pooled if pooled else null[g]
        p = perm_p(nl, o)
        if pooled:
            diag = np.concatenate([np.diag(mats[k]) for k in groups])
            off = np.concatenate([mats[k][~np.eye(mats[k].shape[0], dtype=bool)] for k in groups])
            n = len(diag)
        else:
            n = mats[g].shape[0]
            diag = np.diag(mats[g])
            off = mats[g][~np.eye(n, dtype=bool)]
        _, p_mw = mannwhitneyu(diag, off, alternative="less")
        rows.append((g, n, float(diag.mean()), float(off.mean()), o, p, float(nl.mean()),
                     float(np.percentile(nl, 95)), float(p_mw)))
        lines += [
            f"== {g} (n={n} images) ==",
            f"  matched mean={diag.mean():.4f}  non-target mean={off.mean():.4f}",
            f"  identification accuracy = {o:.4f}",
            f"  permutation null: mean={nl.mean():.4f}  95th pct={np.percentile(nl, 95):.4f}",
            f"  permutation p = {p:.4g}",
            f"  Mann-Whitney p (pooled {len(diag)} vs {len(off)} distances) = {p_mw:.4g}"
            f"   [assumes independent samples; report next to the permutation p]",
            "",
        ]

    txt = os.path.join(out_dir, f"permutation_test_{label}.txt")
    with open(txt, "w") as f:
        f.write("\n".join(lines))
    csv = os.path.join(out_dir, f"permutation_test_{label}.csv")
    with open(csv, "w") as f:
        f.write("group,n,matched_mean,nontarget_mean,ident_acc,p_perm,"
                "null_mean,null_p95,p_mannwhitney\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6g},"
                    f"{r[6]:.6f},{r[7]:.6f},{r[8]:.6g}\n")
    print("\n".join(lines))

    # ---- figure: permutation null vs observed ----
    panels = groups + ["pooled"]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 2.1), squeeze=False)
    for ax, g in zip(axes[0], panels):
        pooled = g == "pooled"
        nl = null_pooled if pooled else null[g]
        o = obs_pooled if pooled else obs[g]
        ax.hist(nl, bins=40, color="0.75", edgecolor="none")
        ax.axvline(o, color="crimson", lw=1.2)
        ax.axvline(0.5, color="0.4", lw=0.6, ls=":")
        ax.set_title(f"{g}\nacc={o:.3f}, p={perm_p(nl, o):.3g}", fontsize=6.5)
        ax.set_xlabel("identification accuracy", fontsize=6)
        ax.tick_params(labelsize=5.5)
        for s in ax.spines.values():
            s.set_linewidth(0.4)
    axes[0][0].set_ylabel(f"permutations (n={B})", fontsize=6)
    fig.suptitle(f"{metric}: within-participant label-permutation null", fontsize=7.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(out_dir, f"permutation_null_{label}")
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    print(f"saved: {txt}\n       {csv}\n       {out}.pdf / .png")


if __name__ == "__main__":
    main()
