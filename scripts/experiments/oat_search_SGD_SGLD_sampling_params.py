
# %%
# One-at-a-time (OAT) sweep over the reconstruction sampling parameters.
#
# Unlike grid_search_SGD_SGLD_sampling_params.py (full Cartesian product, 6,250
# combos), this fixes every parameter at a BASELINE and varies a single parameter
# at a time across its value list, repeating per parameter. The shared baseline
# point is deduped, so the count is:
#     1 (baseline) + sum_p (n_p - 1)
# For the default lists (5,5,5,5,5 + 2) that is 1 + (4+4+4+4+4+1) = 22 combos.
#
# Baseline defaults = original_all (Koide-Majima et al., 2024):
#   lr_a=0.00015, lr_b=0.15, lr_gamma=0.055, T=1e-6, wL=500, woL=1000
#
# The heavy machinery (model loading, per-image reconstruction, tagging, config
# loading, manifest) is reused from grid_search_SGD_SGLD_sampling_params.py so the
# two analyses stay in lock-step; only the combo enumeration and the output root
# differ. Output default: results/oat_sampling_params (separate from the grid run).
import os
import sys
import argparse

# make the sibling grid-search module importable and reuse its machinery
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import grid_search_SGD_SGLD_sampling_params as gs  # noqa: E402


def build_oat(args):
    """OAT combos: baseline first, then vary each parameter across its list."""
    grid_keys = gs.GRID_KEYS
    baseline = {
        'lr_a': args.baseline_lr_a, 'lr_b': args.baseline_lr_b,
        'lr_gamma': args.baseline_lr_gamma, 'T': args.baseline_T,
        'numReps_withLangevin': args.baseline_numReps_withLangevin,
        'numReps_withoutLangevin': args.baseline_numReps_withoutLangevin,
    }
    value_lists = {k: getattr(args, k) for k in grid_keys}

    seen, combos = set(), []

    def _add(d):
        t = tuple(d[k] for k in grid_keys)
        if t not in seen:
            seen.add(t)
            combos.append(t)

    _add(baseline)                       # center point first
    for p in grid_keys:                  # then sweep each axis around it
        for v in value_lists[p]:
            d = dict(baseline)
            d[p] = v
            _add(d)
    return grid_keys, combos


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="One-at-a-time sweep over reconstruction sampling parameters (SGD/SGLD).")
    # swept value lists (same defaults as the grid script)
    parser.add_argument('--lr_a', type=float, nargs='+', default=[0.00015, 0.01, 0.1, 1.0, 10])
    parser.add_argument('--lr_b', type=float, nargs='+', default=[0.05, 0.115533, 0.15, 0.5, 1.0])
    parser.add_argument('--lr_gamma', type=float, nargs='+', default=[0, 0.03, 0.055, 0.1, 0.3])
    parser.add_argument('--T', type=float, nargs='+', default=[1e-6, 1e-4, 1e-2, 0.1, 1.0])
    parser.add_argument('--numReps_withLangevin', type=int, nargs='+', default=[0, 250, 500, 1000, 1500])
    parser.add_argument('--numReps_withoutLangevin', type=int, nargs='+', default=[0, 1000])
    # OAT baseline (defaults = original_all / Koide-Majima)
    parser.add_argument('--baseline_lr_a', type=float, default=0.00015)
    parser.add_argument('--baseline_lr_b', type=float, default=0.15)
    parser.add_argument('--baseline_lr_gamma', type=float, default=0.055)
    parser.add_argument('--baseline_T', type=float, default=1e-6)
    parser.add_argument('--baseline_numReps_withLangevin', type=int, default=500)
    parser.add_argument('--baseline_numReps_withoutLangevin', type=int, default=1000)
    # fixed parameters (single value)
    parser.add_argument('--numReps', type=int, default=1000,
                        help='base numReps; no-op for iteration count, kept single value')
    parser.add_argument('--clip_coef', type=float, default=0.25)
    parser.add_argument('--feat_set', type=str, default='all')
    parser.add_argument('--display_every', type=int, default=50)
    parser.add_argument('--similarity', type=str, default='corr')
    # scope
    parser.add_argument('--subjects', type=str, nargs='+', default=['S01', 'S02', 'S03'])
    parser.add_argument('--targetID', type=int, nargs='+', default=list(range(25)),
                        help='default = all 25 imagery stimuli (targetID 0..24)')
    # io / control  (NOTE: separate default output root from the grid run)
    parser.add_argument('--out', type=str, default='./results/oat_sampling_params')
    parser.add_argument('--dry_run', action='store_true',
                        help='enumerate combos (+ manifest) without running reconstruction')
    parser.add_argument('--resume', action='store_true',
                        help='skip (combo, subject, image) whose done-marker already exists')
    parser.add_argument('--save_trajectory', action='store_true',
                        help='also store per-step latent-vector trajectories in the pkl (heavy)')
    parser.add_argument('--trajectory_stride', type=int, default=1,
                        help='keep every Nth trajectory step (only with --save_trajectory)')
    # sharding for multi-GPU
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_id', type=int, default=0)
    args = parser.parse_args()

    if not (0 <= args.shard_id < args.num_shards):
        parser.error(f'--shard_id must be in [0, {args.num_shards}); got {args.shard_id}')

    prm_demo, dt_cfg = gs.load_config()

    grid_keys, combos = build_oat(args)
    total = len(combos)
    os.makedirs(args.out, exist_ok=True)

    # manifest: only shard 0 (or dry-run) writes it to avoid concurrent clobber
    manifest_path = os.path.join(args.out, 'oat_manifest.csv')
    if args.shard_id == 0 or args.dry_run:
        gs.write_manifest(manifest_path, grid_keys, combos, args)
        print(f'[manifest] wrote {total} OAT combos to {manifest_path}')

    my_indices = [idx for idx in range(total) if idx % args.num_shards == args.shard_id]
    print(f'[shard {args.shard_id}/{args.num_shards}] handling {len(my_indices)}/{total} combos '
          f'| subjects={args.subjects} targetID={args.targetID}')

    if args.dry_run:
        for idx in my_indices:
            params = dict(zip(grid_keys, combos[idx]))
            params.update(numReps=args.numReps, clip_coef=args.clip_coef, feat_set=args.feat_set)
            tag = gs.make_tag(params)
            print(f'[{idx}/{total}] {tag} -> {args.out}/{tag}')
        print(f'[dry-run] {len(my_indices)} combos for this shard '
              f'({len(my_indices) * len(args.subjects) * len(args.targetID)} reconstructions)')
        sys.exit(0)

    # load heavy models ONCE (reused from the grid module)
    models = gs.load_models(dt_cfg)

    done_log = os.path.join(args.out, f'done.shard{args.shard_id}.log')
    errors_log = os.path.join(args.out, f'errors.shard{args.shard_id}.log')

    for n, idx in enumerate(my_indices):
        params = dict(zip(grid_keys, combos[idx]))
        params.update(
            numReps=args.numReps, clip_coef=args.clip_coef, feat_set=args.feat_set,
            display_every=args.display_every, similarity=args.similarity,
        )
        tag = gs.make_tag(params)
        save_base_dir = f'{args.out}/{tag}'
        print(f'[shard {args.shard_id}] [{n+1}/{len(my_indices)}] (global {idx}/{total}) tag={tag}')

        gs.run_recon(models, params, save_base_dir, dt_cfg, prm_demo,
                     subjects=args.subjects, targetID_list=args.targetID,
                     resume=args.resume, save_trajectory=args.save_trajectory,
                     trajectory_stride=args.trajectory_stride, errors_log=errors_log)

        with open(done_log, 'a') as f:
            f.write(f'{idx}\t{tag}\n')
