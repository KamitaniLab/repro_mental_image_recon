"""Figure: the same seed is bit-reproducible on CPU but not on GPU.

Left  -- image grid, one column per device:
           row 1  target image
           row 2  run 1 (SGLD final)
           row 3  run 2 (SGLD final)
           row 4  |run 1 - run 2|, shared magnitude scale
Right -- mean |pixel difference| between the two runs, per device.

Inputs are produced by scripts/experiments/check_determinism.py --recon
(once with --device cpu, once on GPU), which writes into
results/_determinism_check/.

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
import yaml
from matplotlib.gridspec import GridSpec
from PIL import Image

# Categorical slots 1 and 2 of the reference palette (fixed order, not cycled).
COLOR = {'cpu': '#2a78d6', 'cuda': '#eb6834'}
LABEL = {'cpu': 'CPU', 'cuda': 'GPU (CUDA)'}
INK = '#0b0b0b'
INK_MUTED = '#52514e'
DIFF_CMAP = 'Blues'          # sequential, single hue: magnitude

RESULT_DIR = os.path.join('results', '_determinism_check')
OUT_DIR = os.path.join('assets', 'determinism')


def load_target(targetID):
    from recon_utils import get_target_image
    with open('./scripts/config/demo_params.yaml', 'rb') as f:
        prm = yaml.safe_load(f)
    img, name = get_target_image(targetID, prm['dt_targetimages_path'])
    return np.asarray(img), name


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
    ap.add_argument('--target', type=int, default=18)
    ap.add_argument('--devices', nargs='+', default=['cpu', 'cuda'])
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    target, targetname = load_target(args.target)
    runs = {d: load_pair(d, args.stage) for d in args.devices}
    diffs = {d: np.abs(runs[d][0] - runs[d][1]) for d in args.devices}
    stats = {d: dict(mean=diffs[d].mean(), max=diffs[d].max(),
                     identical=np.array_equal(runs[d][0], runs[d][1]))
             for d in args.devices}

    # One magnitude scale across devices, so the two diff panels are comparable.
    vmax = max(1.0, max(d.max() for d in diffs.values()))

    ncol = len(args.devices)
    # A blank spacer column keeps the bar chart's y-label off the image grid.
    fig = plt.figure(figsize=(2.6 + 1.5 * ncol + 3.6, 7.6))
    gs = GridSpec(4, ncol + 2, figure=fig,
                  width_ratios=[1] * ncol + [0.5, 1.75 * ncol],
                  hspace=0.08, wspace=0.06,
                  left=0.11, right=0.97, top=0.89, bottom=0.10)

    rows = ['Target', 'Run 1', 'Run 2', '|Run 1 - Run 2|']

    for ci, dev in enumerate(args.devices):
        for ri in range(4):
            ax = fig.add_subplot(gs[ri, ci])
            if ri == 0:
                ax.imshow(target)
            elif ri in (1, 2):
                ax.imshow(runs[dev][ri - 1].astype(np.uint8))
            else:
                im = ax.imshow(diffs[dev].mean(axis=2), cmap=DIFF_CMAP,
                               vmin=0, vmax=vmax)
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

    # Colorbar for the diff row, tucked under the image grid.
    cax = fig.add_axes([0.115, 0.055, 0.105 * ncol, 0.011])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('|difference|  (0-255)', fontsize=8, color=INK_MUTED)
    cb.ax.tick_params(labelsize=7, colors=INK_MUTED, length=2)
    cb.outline.set_visible(False)

    # ---- right: the diff magnitude as a bar per device
    axb = fig.add_subplot(gs[:, ncol + 1])
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
    axb.set_ylabel('mean |pixel difference| between the two runs',
                   fontsize=10, color=INK_MUTED, labelpad=8)
    axb.set_ylim(0, top)
    axb.grid(axis='y', color='#e8e7e2', linewidth=0.8, zorder=0)
    axb.set_axisbelow(True)
    for side in ('top', 'right'):
        axb.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axb.spines[side].set_color('#d8d7d2')
    axb.tick_params(colors=INK_MUTED, labelsize=9, length=3)

    stage_name = 'after SGLD' if args.stage == 'final' else 'after Adam'
    fig.suptitle('Same seed, same code: bit-reproducible on CPU, not on GPU',
                 fontsize=13.5, color=INK, fontweight='bold', y=0.975)
    fig.text(0.5, 0.935,
             f'target {targetname}   |   two runs, identical seed   |   '
             f'reconstruction {stage_name}',
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
