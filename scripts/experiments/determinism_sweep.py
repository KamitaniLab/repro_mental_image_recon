"""How large is the GPU non-determinism, across many images?

`check_determinism.py --recon` settles the question for one image: two same-seed
runs are bit-identical on CPU and differ on GPU. This widens that to a
distribution -- repeat the run for every target (and optionally every subject)
and report the spread of the mean absolute pixel difference.

Every run everywhere uses the SAME seed (--seed, default 2956797496, the same
value check_determinism.py uses, so the sweep and the single-image CPU/GPU check
are directly comparable). Holding it constant
across conditions keeps it from confounding the comparison: any difference
between two runs is the GPU's arithmetic, and any difference between conditions
is the image, never the seed.

All runs of a condition happen in this one process, so the models are loaded
once instead of once per run. That is sound here because the reconstruction draws
its randomness from a per-run `torch.Generator` seeded from `seed`, and the
augmentation restores the global RNG state it touches -- so a second run with a
freshly seeded generator starts from the same state as the first. Cross-check the
magnitude against `check_determinism.py --recon`, which uses two separate
processes on one target.

--runs sets how many runs per condition (default 2 -> one comparison). More runs
give C(runs, 2) comparisons and show the spread within a single condition.

Writes one CSV row per comparison, plus the images if --save-images.

Usage (from the repo root):
    python scripts/experiments/determinism_sweep.py            # 3 subjects x 25 targets
    python scripts/experiments/determinism_sweep.py --subjects S01
    python scripts/experiments/determinism_sweep.py --device cpu --targets 0 1
"""
import argparse
import csv
import os
import time

import numpy as np
import torch

from repro_mental_image_recon.recon import func_reproducible as R

OUT_DIR = os.path.join('results', 'determinism_sweep')


def load_models(dt_cfg, dev):
    import model_loading
    tt = dt_cfg['file_path']['taming_transformer_dir']
    print('loading VQGAN / VGG19 / CLIP ...', flush=True)
    cfg = model_loading.load_config(
        tt + '/logs/vqgan_imagenet_f16_1024/configs/model.yaml', display=False)
    VQGAN = model_loading.load_vqgan(
        cfg, ckpt_path=tt + '/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt').to(dev)
    VQGAN.eval()
    VGG, _ = model_loading.load_VGG_model(dev)
    CLIP, _ = model_loading.load_CLIP_model(dt_cfg['models']['CLIP']['modeltypes'], dev)
    print('models loaded', flush=True)
    return VQGAN, VGG, CLIP


def load_features(dt_cfg, prm_demo, subject, targetID, method, dev):
    """Per-target decoded/mean features. Models stay loaded across calls."""
    import pickle
    import scipy.io
    from recon_utils import get_target_label, convert_featname

    names = dt_cfg['models']['CLIP']['modelnames']
    clip_layer = dt_cfg['models']['CLIP']['used_layer']
    feat_set = dt_cfg['recon_params'][method]['feat_set']
    vgg_in = dt_cfg['recon_feat_layers'][feat_set]['VGG19']
    vgg_dirs = convert_featname(vgg_in, cvt_to='directory')
    meanDir = dt_cfg['file_path']['mean_feat_dir']
    targetimname = get_target_label(targetID, prm_demo['dt_targetimages_path'])

    def decpath(model, layer):
        p = prm_demo['decfearture_path']
        for a, b in (('__subjectname__', subject), ('__modelname__', model),
                     ('__layername__', layer), ('__targetimname__', targetimname)):
            p = p.replace(a, b)
        return p

    def load_dec(model, layer):
        with open(decpath(model, layer), 'rb') as f:
            dt = pickle.load(f)
        return torch.tensor(dt[0].astype('float32'),
                            dtype=torch.float32).to(dev).unsqueeze(0)

    def load_mean(model, layer):
        x = scipy.io.loadmat(os.path.join(meanDir, model, layer, 'meanFeature_.mat'))
        return torch.tensor(x['mu'], dtype=torch.float32).to(dev)

    w = np.ones(len(vgg_dirs))
    return dict(
        targetVGG=[load_dec('VGG19', name) for name in vgg_dirs],
        meanVGG=[load_mean('VGG19', name) for name in vgg_dirs],
        vggw=w / w.sum(), vgg_in=vgg_in,
        targetCLIP=[load_dec(n, clip_layer) for n in names],
        meanCLIP=[load_mean(n, clip_layer) for n in names],
        name=targetimname)


def run_once(models, feats, dt_cfg, method, seed, dev, n_sgd, n_lang):
    from PIL import Image
    VQGAN, VGG, CLIP = models
    recon = R.imageRecon(
        feats['targetVGG'], feats['meanVGG'], feats['vggw'], VGG, feats['vgg_in'],
        feats['targetCLIP'], feats['meanCLIP'],
        dt_cfg['models']['CLIP']['modelcoefs'], CLIP, VQGAN,
        Image.fromarray(np.uint8(np.ones([240, 240, 3]) * 128)),
        initInputType='PIL',
        similarity=dt_cfg['recon_params'][method]['similarity'],
        disp_every=max(n_sgd, 1),
        numReps=dt_cfg['recon_params'][method]['numReps'],
        CLIPcoef=dt_cfg['recon_params'][method]['clip_coef'],
        DEVICE=dev, seed=seed)

    latent, img = None, None
    for recImg, _, _, _, vec in recon.withoutLangevin(numReps=n_sgd, returnVec=True):
        img = recImg
        if vec is not None:
            latent = vec
    if n_lang > 0:
        L = dt_cfg['recon_params'][method]['Langevin']
        for recImg, _, _, _, vec in recon.Langevin(
                initInput=latent, initInputType='latentVector', numReps=n_lang,
                returnVec=True, lr_gamma=L['lr_gamma'], lr_a=L['lr_a'],
                lr_b=L['lr_b'], T=L['T']):
            img = recImg
            if vec is not None:
                latent = vec
    return img, latent.detach().cpu().numpy()


def main():
    import yaml
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='+', default=['S01', 'S02', 'S03'])
    ap.add_argument('--targets', type=int, nargs='+', default=list(range(25)))
    ap.add_argument('--method', default='original_all')
    ap.add_argument('--seed', type=int, default=2956797496,
                    help='the one seed every run uses, held constant across '
                         'conditions so it cannot confound the comparison. '
                         'Default matches check_determinism.py, so the sweep and '
                         'the single-image CPU/GPU check are the same seed.')
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--n-sgd', type=int, default=None)
    ap.add_argument('--n-lang', type=int, default=None)
    ap.add_argument('--runs', type=int, default=2,
                    help='runs per condition; C(runs,2) comparisons each')
    ap.add_argument('--save-images', action='store_true')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    dev = (torch.device('cpu') if args.device == 'cpu'
           else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
    gpu = torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'
    print(f'host={os.uname().nodename}  device={dev}  gpu={gpu}  '
          f'torch={torch.__version__}  cuda={torch.version.cuda}', flush=True)

    with open('./scripts/config/demo_params.yaml', 'rb') as f:
        prm_demo = yaml.safe_load(f)
    with open('./scripts/config/config_recon.yaml', 'rb') as f:
        dt_cfg = yaml.safe_load(f)

    n_sgd = (args.n_sgd if args.n_sgd is not None
             else dt_cfg['recon_params'][args.method]['numReps_withoutLangevin'])
    n_lang = (args.n_lang if args.n_lang is not None
              else dt_cfg['recon_params'][args.method]['numReps_withLangevin'])

    R.set_seed(args.seed)
    models = load_models(dt_cfg, dev)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = args.out or os.path.join(OUT_DIR, f'{dev.type}_pairs.csv')
    img_dir = os.path.join(OUT_DIR, f'{dev.type}_images')
    if args.save_images:
        os.makedirs(img_dir, exist_ok=True)

    conditions = [(s, t) for s in args.subjects for t in args.targets]
    print(f'{len(conditions)} conditions x {args.runs} runs, seed={args.seed}  '
          f'(n_sgd={n_sgd}, n_lang={n_lang})', flush=True)

    from itertools import combinations
    t0 = time.time()
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subject', 'target_id', 'target_name', 'seed', 'run_i', 'run_j',
                    'identical', 'mean_abs_diff', 'max_abs_diff', 'image_corr',
                    'latent_corr'])
        for k, (subj, tid) in enumerate(conditions, 1):
            feats = load_features(dt_cfg, prm_demo, subj, tid, args.method, dev)
            seed = args.seed

            # All runs share the seed; any difference between them is the GPU's,
            # not the sampling's. --runs 2 gives one comparison per condition;
            # more runs give C(runs, 2) and show the spread within a condition.
            imgs, lats = [], []
            for _ in range(args.runs):
                img, lat = run_once(models, feats, dt_cfg, args.method, seed, dev,
                                    n_sgd, n_lang)
                imgs.append(img)
                lats.append(lat)

            if args.save_images:
                for r, img in enumerate(imgs, 1):
                    img.save(os.path.join(img_dir, f'{subj}_t{tid:02d}_run{r}.png'))

            means = []
            for i, j in combinations(range(args.runs), 2):
                a = np.asarray(imgs[i], dtype=np.float64)
                b = np.asarray(imgs[j], dtype=np.float64)
                d = np.abs(a - b)
                means.append(d.mean())
                w.writerow([subj, tid, feats['name'], seed, i + 1, j + 1,
                            bool(np.array_equal(a, b)),
                            f'{d.mean():.6f}', f'{d.max():.0f}',
                            f'{np.corrcoef(a.ravel(), b.ravel())[0, 1]:.8f}',
                            f'{np.corrcoef(lats[i].ravel(), lats[j].ravel())[0, 1]:.8f}'])
            f.flush()

            el = time.time() - t0
            spread = (f'  (over {len(means)} comparisons: '
                      f'{min(means):.3f}-{max(means):.3f})' if len(means) > 1 else '')
            print(f'[{k}/{len(conditions)}] {subj} target{tid:02d} seed={seed}  '
                  f'mean|d|={np.mean(means):.3f}{spread}  '
                  f'({el/60:.1f} min, eta {el/k*(len(conditions)-k)/60:.0f} min)',
                  flush=True)

    print(f'\nwrote {out_csv}')
    vals = []
    with open(out_csv) as f:
        for r in csv.DictReader(f):
            vals.append(float(r['mean_abs_diff']))
    v = np.array(vals)
    print(f'mean|d| over {len(v)} pairs: mean={v.mean():.3f}  '
          f'min={v.min():.3f}  max={v.max():.3f}  median={np.median(v):.3f}')


if __name__ == '__main__':
    main()
