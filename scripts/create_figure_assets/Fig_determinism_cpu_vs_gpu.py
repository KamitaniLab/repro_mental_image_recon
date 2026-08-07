"""Figure: the same seed is bit-reproducible on CPU but not on GPU.

Left  -- image grid, one column per device:
           row 1  run 1
           row 2  run 2
           row 3  |run 1 - run 2|, raw per-channel difference (black = identical)
Right -- mean |pixel difference| between two same-seed runs, pooled over every
         subject and image in the sweep.

No target image is shown: the right panel pools 75 images, so putting one target
beside it would suggest the distribution belongs to that image.

Inputs are produced by scripts/experiments/check_determinism.py --recon (once
with --device cpu, once on GPU) for the image grid, and by
scripts/experiments/determinism_sweep.py for the distribution.

Usage:
    python scripts/create_figure_assets/Fig_determinism_cpu_vs_gpu.py
    python scripts/create_figure_assets/Fig_determinism_cpu_vs_gpu.py --stage sgd
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

# Categorical slots 1 and 2 of the reference palette (fixed order, not cycled).
COLOR = {'cpu': '#2a78d6', 'cuda': '#eb6834'}
LABEL = {'cpu': 'CPU', 'cuda': 'GPU (CUDA)'}
INK = '#0b0b0b'
INK_MUTED = '#52514e'

RESULT_DIR = os.path.join('results', '_determinism_check')
SWEEP_CSV = os.path.join('results', '_determinism_sweep', 'cuda_pairs.csv')
OUT_DIR = os.path.join('assets', 'determinism')


def load_sweep(path):
    """subject -> list of mean|d| values, from determinism_sweep.py's CSV.

    Returns None when the sweep has not been run, so the figure falls back to
    the single-image pair.
    """
    import csv
    if not path or not os.path.exists(path):
        return None
    out = {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            out.setdefault(r['subject'], []).append(float(r['mean_abs_diff']))
    return out or None


def load_pair(device, stage):
    """The two runs' output images for one device, as float arrays."""
    paths = [os.path.join(RESULT_DIR, f'{device}_recon{i}_{stage}.png')
             for i in (1, 2)]
    for p in paths:
        if not os.path.exists(p):
            sys.exit(f'missing {p}\n'
                     'run: python scripts/experiments/check_determinism.py '
                     f'--recon{" --device cpu" if device == "cpu" else ""}')
    return [np.asarray(Image.open(p), dtype=np.float64) for p in paths]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['final', 'sgd'], default='final',
                    help='final = after SGLD (default); sgd = after Adam')
    ap.add_argument('--devices', nargs='+', default=['cpu', 'cuda'])
    ap.add_argument('--sweep', default=SWEEP_CSV,
                    help='determinism_sweep.py CSV; the right panel shows its '
                         'distribution. Pass "" to fall back to the single pair.')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    runs = {d: load_pair(d, args.stage) for d in args.devices}
    diffs = {d: np.abs(runs[d][0] - runs[d][1]) for d in args.devices}
    stats = {d: dict(mean=diffs[d].mean(), max=diffs[d].max(),
                     identical=np.array_equal(runs[d][0], runs[d][1]))
             for d in args.devices}

    ncol = len(args.devices)
    # A blank spacer column keeps the bar chart's y-label off the image grid.
    fig = plt.figure(figsize=(2.6 + 1.5 * ncol + 3.6, 6.2))
    gs = GridSpec(3, ncol + 2, figure=fig,
                  width_ratios=[1] * ncol + [0.5, 1.75 * ncol],
                  hspace=0.08, wspace=0.06,
                  left=0.11, right=0.97, top=0.86, bottom=0.12)

    # No target row: the right panel pools 75 images, so showing one target
    # beside it would imply the distribution belongs to that image.
    rows = ['Run 1', 'Run 2', '|Run 1 - Run 2|']

    for ci, dev in enumerate(args.devices):
        for ri in range(3):
            ax = fig.add_subplot(gs[ri, ci])
            if ri in (0, 1):
                ax.imshow(runs[dev][ri].astype(np.uint8))
            else:
                # Same rendering as Fig5B's difference panel: the raw per-channel
                # |a - b| shown as an image, so 0 is black and larger differences
                # are brighter. No colormap, so the two columns are directly
                # comparable and CPU's all-zero panel reads as solid black.
                ax.imshow(np.clip(diffs[dev], 0, 255).astype(np.uint8))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor('#d8d7d2')
                s.set_linewidth(0.8)
            if ri == 0:
                ax.set_title(LABEL[dev], fontsize=11, color=COLOR[dev],
                             fontweight='bold', pad=6)
            if ci == 0:
                ax.set_ylabel(rows[ri], fontsize=9.5, color=INK_MUTED,
                              rotation=0, ha='right', va='center', labelpad=10)

    # Legend for the diff row: the intensity ramp the raw difference is read on.
    cax = fig.add_axes([0.115, 0.055, 0.105 * ncol, 0.011])
    cax.imshow(np.linspace(0, 1, 256).reshape(1, -1), cmap='gray',
               aspect='auto', vmin=0, vmax=1)
    cax.set_yticks([])
    cax.set_xticks([0, 63.75, 127.5, 191.25, 255])
    cax.set_xticklabels(['0', '64', '128', '192', '255'], fontsize=7,
                        color=INK_MUTED)
    cax.set_xlabel('|difference| per channel  (black = identical)',
                   fontsize=8, color=INK_MUTED, labelpad=2)
    cax.tick_params(length=2, colors=INK_MUTED)
    for s in cax.spines.values():
        s.set_visible(False)

    # ---- right: every GPU same-seed comparison as a point, CPU as the zero line
    axb = fig.add_subplot(gs[:, ncol + 1])
    sweep = load_sweep(args.sweep)

    if sweep is None:
        # No sweep available: fall back to the single-image pair per device.
        xs = np.arange(ncol)
        vals = [stats[d]['mean'] for d in args.devices]
        top = max(vals) * 1.28 if max(vals) > 0 else 1.0
        axb.bar(xs, vals, width=0.42,
                color=[COLOR[d] for d in args.devices], zorder=3)
        for x, d in zip(xs, args.devices):
            v = stats[d]['mean']
            txt = '0\n(bit-identical)' if stats[d]['identical'] else f'{v:.1f}'
            axb.text(x, v + top * 0.02, txt, ha='center', va='bottom',
                     fontsize=10, color=INK, linespacing=1.4,
                     fontweight='bold' if not stats[d]['identical'] else 'normal')
        axb.set_xticks(xs)
        axb.set_xticklabels([LABEL[d] for d in args.devices], fontsize=10, color=INK)
        n_note = 'one image'
    else:
        # Subjects pooled: one distribution over every same-seed comparison.
        subs = sorted(sweep)
        v = np.concatenate([np.asarray(sweep[s]) for s in subs])
        top = v.max() * 1.22
        rng = np.random.default_rng(0)          # jitter only; fixed for stability

        # CPU: every pair was bit-identical, so the whole distribution sits at 0.
        axb.scatter([0], [0], s=46, color=COLOR['cpu'], zorder=4,
                    edgecolor='white', linewidth=0.8)
        axb.text(0, top * 0.03, '0\n(all bit-identical)', ha='center', va='bottom',
                 fontsize=9.5, color=INK, linespacing=1.4)

        x = 1 + (rng.random(len(v)) - 0.5) * 0.34
        axb.scatter(x, v, s=24, color=COLOR['cuda'], alpha=0.5, zorder=3,
                    edgecolor='none')
        # mean, with the full range drawn through it
        axb.plot([1, 1], [v.min(), v.max()], color=INK, linewidth=1.0, zorder=4)
        axb.plot([0.72, 1.28], [v.mean()] * 2, color=INK, linewidth=2.2,
                 zorder=5, solid_capstyle='round')
        for y, lab in ((v.max(), f'max {v.max():.1f}'),
                       (v.min(), f'min {v.min():.1f}')):
            axb.text(1.34, y, lab, ha='left', va='center', fontsize=8.5,
                     color=INK_MUTED)
        axb.text(1.34, v.mean(), f'mean {v.mean():.1f}', ha='left', va='center',
                 fontsize=9.5, color=INK, fontweight='bold')

        axb.set_xticks([0, 1])
        axb.set_xticklabels(['CPU', 'GPU (CUDA)'], fontsize=10, color=INK)
        axb.set_xlim(-0.6, 2.0)
        n_note = (f'{len(v)} same-seed comparisons pooled over '
                  f'{len(subs)} subjects x 25 images')

    axb.set_ylabel('mean |pixel difference| between two same-seed runs',
                   fontsize=10, color=INK_MUTED, labelpad=8)
    axb.set_ylim(0, top)
    axb.set_title(n_note, fontsize=9, color=INK_MUTED, pad=6)
    axb.grid(axis='y', color='#e8e7e2', linewidth=0.8, zorder=0)
    axb.set_axisbelow(True)
    for side in ('top', 'right'):
        axb.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axb.spines[side].set_color('#d8d7d2')
    axb.tick_params(colors=INK_MUTED, labelsize=9, length=3)

    stage_name = ('full schedule (Adam + SGLD)' if args.stage == 'final'
                  else 'Adam phase only')
    fig.suptitle('Same seed, same code: bit-reproducible on CPU, not on GPU',
                 fontsize=13.5, color=INK, fontweight='bold', y=0.978)
    fig.text(0.5, 0.925,
             "seed-controlled version of Koide-Majima's implementation   |   "
             f'two runs, identical seed   |   {stage_name}',
             ha='center', fontsize=9.5, color=INK_MUTED)

    out = args.out or os.path.join(OUT_DIR, f'Fig_determinism_{args.stage}')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out}.{ext}', dpi=200, bbox_inches='tight',
                    facecolor='white')
    print(f'saved {out}.pdf / .png')
    for d in args.devices:
        s = stats[d]
        print(f'  {LABEL[d]:<12} identical={s["identical"]}  '
              f'mean|d|={s["mean"]:.4f}  max|d|={s["max"]:.0f}')


if __name__ == '__main__':
    main()
