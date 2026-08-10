"""Is the seeded reconstruction actually reproducible? Run this and read the verdict.

    python scripts/experiments/check_determinism.py                  # GPU, quick
    python scripts/experiments/check_determinism.py --device cpu     # CPU, quick
    python scripts/experiments/check_determinism.py --recon          # + real pipeline

`--seed` fixes every random draw, but that alone does not make two runs
bit-identical on a GPU: some CUDA backward kernels accumulate with atomicAdd,
whose order is not fixed. This script separates "the seeding is wrong" from
"the GPU arithmetic is not reproducible" by running each check twice in two
*separate processes* (it spawns them itself) and comparing the results.

Checks, cheapest first:

  forward   createCrops forward only -- pure sampling, no backward. Must be
            bit-identical on any device. A mismatch means a random draw escapes
            the generator, i.e. the seeding is broken.
  backward  the gradient through createCrops. Isolates the backward kernels
            (grid_sample / interpolate) from the sampling.
  recon     (--recon) the real reconstruction: VQGAN + VGG19 + CLIP, the full
            Adam + Langevin schedule. Records the latent at a set of steps and
            saves the output images, so the divergence can be followed step by
            step. Minutes on GPU, much longer on CPU -- use --n-sgd / --n-lang
            to shorten it.

Run it once on GPU and once with --device cpu: CPU has deterministic kernels for
these ops, so the pair of results tells the two causes apart.

Everything is written to results/determinism_check/ (inside the repo, so it
survives a reboot); pass --out-dir to write somewhere else.
"""
import argparse
import hashlib
import os
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import recon_func_reproducible as R  # noqa: E402

DEFAULT_OUT_DIR = os.path.join('results', 'determinism_check')

# Steps at which the latent is recorded. Langevin steps are numbered n_sgd + t
# so both phases share one axis.
SNAP_SGD = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000]
SNAP_LANG = [1, 2, 5, 10, 20, 50, 100, 200, 300, 500]


def pick_device(name):
    if name == 'cpu':
        return torch.device('cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        sys.exit('CUDA requested but not available')
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def fixed_image(dev, size=224):
    """RNG-free input, so any output difference comes from the probe target."""
    return torch.linspace(-1., 1., 3 * size * size, device=dev).reshape(1, 3, size, size)


# --------------------------------------------------------------------------
# worker side: one process produces one file

def work_crops(out_path, args, dev, backward):
    R.set_seed(args.base_seed)
    if args.list_nondet:
        torch.use_deterministic_algorithms(True, warn_only=True)

    img = fixed_image(dev)
    if backward:
        img.requires_grad_()
    gen = torch.Generator(device=dev).manual_seed(args.seed)

    outs = []
    for _ in range(args.steps):
        crops = R.createCrops(img, num_crops=args.num_crops, DEVICE=dev,
                              generator=gen, augment=not args.no_augment)
        if not backward:
            outs.append(crops.detach().cpu().numpy().copy())
            continue
        # Deterministic weighting -> the differentiated scalar is itself
        # deterministic, so a gradient mismatch can only come from backward.
        w = torch.linspace(0.1, 1.0, crops.numel(), device=dev).reshape(crops.shape)
        loss = (crops * w).sum()
        img.grad = None
        loss.backward()
        outs.append(img.grad.detach().cpu().numpy().copy())

    np.save(out_path, np.stack(outs))


def build_recon(args, dev):
    import pickle
    import scipy.io
    import yaml
    from PIL import Image

    from recon_utils import get_target_label, convert_featname

    with open('./scripts/config/demo_params.yaml', 'rb') as f:
        prm_demo = yaml.safe_load(f)
    with open('./scripts/config/config_recon.yaml', 'rb') as f:
        dt_cfg = yaml.safe_load(f)

    R.set_seed(args.seed)
    sys.path.insert(0, dt_cfg['file_path']['taming_transformer_dir'])
    import model_loading

    print('  loading VQGAN / VGG19 / CLIP ...', flush=True)
    tt = dt_cfg['file_path']['taming_transformer_dir']
    cfg = model_loading.load_config(
        tt + "/logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
    VQGAN = model_loading.load_vqgan(
        cfg, ckpt_path=tt + "/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(dev)
    VQGAN.eval()
    VGGmodel_, _ = model_loading.load_VGG_model(dev)
    names = dt_cfg["models"]["CLIP"]["modelnames"]
    clip_layer = dt_cfg["models"]["CLIP"]["used_layer"]
    CLIPmodelWeight_ = dt_cfg["models"]["CLIP"]["modelcoefs"]
    CLIPmodel_, _ = model_loading.load_CLIP_model(
        dt_cfg["models"]["CLIP"]["modeltypes"], dev)

    feat_set = dt_cfg["recon_params"][args.method]["feat_set"]
    vgg_in = dt_cfg["recon_feat_layers"][feat_set]["VGG19"]
    vgg_dirs = convert_featname(vgg_in, cvt_to='directory')
    meanDir = dt_cfg['file_path']['mean_feat_dir']
    targetimname = get_target_label(args.target, prm_demo['dt_targetimages_path'])

    def decpath(model, layer):
        p = prm_demo['decfearture_path']
        for a, b in (('__subjectname__', args.subject), ('__modelname__', model),
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
    w = w / w.sum()
    return dict(
        args=([load_dec('VGG19', name) for name in vgg_dirs],
              [load_mean('VGG19', name) for name in vgg_dirs],
              w, VGGmodel_, vgg_in,
              [load_dec(n, clip_layer) for n in names],
              [load_mean(n, clip_layer) for n in names],
              CLIPmodelWeight_, CLIPmodel_, VQGAN,
              Image.fromarray(np.uint8(np.ones([240, 240, 3]) * 128))),
        kwargs=dict(initInputType='PIL',
                    similarity=dt_cfg['recon_params'][args.method]['similarity'],
                    disp_every=dt_cfg['recon_params'][args.method]['display_every'],
                    numReps=dt_cfg['recon_params'][args.method]['numReps'],
                    CLIPcoef=dt_cfg['recon_params'][args.method]['clip_coef'],
                    DEVICE=dev),
        n_sgd=(args.n_sgd if args.n_sgd is not None
               else dt_cfg['recon_params'][args.method]["numReps_withoutLangevin"]),
        n_lang=(args.n_lang if args.n_lang is not None
                else dt_cfg['recon_params'][args.method]["numReps_withLangevin"]),
        langevin=dt_cfg['recon_params'][args.method]['Langevin'])


def work_recon(out_path, args, dev):
    import time
    b = build_recon(args, dev)
    recon = R.imageRecon(*b['args'], seed=args.seed, **b['kwargs'])
    recon.snap_at = ({s for s in SNAP_SGD if s <= b['n_sgd']} |
                     {b['n_sgd'] + t for t in SNAP_LANG if t <= b['n_lang']})
    recon.snap_offset = b['n_sgd']

    t0 = time.time()
    latent, img_sgd = None, None
    for recImg, step, lv, lc, vec in recon.withoutLangevin(
            numReps=b['n_sgd'], returnVec=True):
        img_sgd = recImg
        print(f'  [SGD] {step}/{b["n_sgd"]} loss_VGG={lv:.6f} loss_CLIP={lc:.6f} '
              f'{time.time()-t0:.0f}s', flush=True)
        if vec is not None:
            latent = vec

    img_final = img_sgd
    if b['n_lang'] > 0:
        L = b['langevin']
        for recImg, step, lv, lc, vec in recon.Langevin(
                initInput=latent, initInputType='latentVector',
                numReps=b['n_lang'], returnVec=True, lr_gamma=L['lr_gamma'],
                lr_a=L['lr_a'], lr_b=L['lr_b'], T=L['T']):
            img_final = recImg
            print(f'  [Langevin] {step}/{b["n_lang"]} loss_VGG={lv:.6f} '
                  f'loss_CLIP={lc:.6f} {time.time()-t0:.0f}s', flush=True)

    stem = os.path.splitext(out_path)[0]
    if img_sgd is not None:
        img_sgd.save(stem + '_sgd.png')
    if img_final is not None:
        img_final.save(stem + '_final.png')
    np.savez(out_path, snaps=np.stack(recon.snapshots),
             steps=np.array(recon.snap_steps), n_sgd=b['n_sgd'])


# --------------------------------------------------------------------------
# driver side: spawn two workers, compare

def spawn(mode, out_path, args):
    """Run this same script as a worker in a fresh process."""
    cmd = [sys.executable, os.path.abspath(__file__), '--_worker', mode,
           '--_out', out_path,
           '--device', args.device, '--seed', str(args.seed),
           '--base-seed', str(args.base_seed), '--steps', str(args.steps),
           '--num-crops', str(args.num_crops), '--subject', args.subject,
           '--out-dir', args.out_dir,
           '--target', str(args.target), '--method', args.method]
    if args.no_augment:
        cmd.append('--no-augment')
    if args.list_nondet:
        cmd.append('--list-nondet')
    if args.n_sgd is not None:
        cmd += ['--n-sgd', str(args.n_sgd)]
    if args.n_lang is not None:
        cmd += ['--n-lang', str(args.n_lang)]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f'worker failed ({mode}, rc={r.returncode})')


def compare_npy(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    same = np.array_equal(a, b)
    d = np.abs(a - b)
    print(f'    sha A={hashlib.sha256(a.tobytes()).hexdigest()[:16]}  '
          f'B={hashlib.sha256(b.tobytes()).hexdigest()[:16]}')
    print(f'    {"IDENTICAL" if same else "DIFFERS"}   '
          f'max|d|={d.max():.3e}  mean|d|={d.mean():.3e}')
    return same


def compare_recon(a_path, b_path):
    a, b = np.load(a_path), np.load(b_path)
    sa, sb, steps = a['snaps'], b['snaps'], a['steps']
    n_sgd = int(a['n_sgd'])
    order = np.argsort(steps)
    print(f'    {"step":>10}  {"phase":>8}  {"status":>9}  {"max|d|":>11}  '
          f'{"rel":>10}  {"corr":>14}')
    all_same = True
    for i in order:
        st = int(steps[i])
        x, y = sa[i].ravel(), sb[i].ravel()
        ok = np.array_equal(x, y)
        all_same &= ok
        d = np.abs(x - y)
        scale = np.abs(x).mean()
        c = np.corrcoef(x, y)[0, 1] if x.std() and y.std() else float('nan')
        phase = 'SGD' if st <= n_sgd else 'Langevin'
        label = st if st <= n_sgd else f'{n_sgd}+{st - n_sgd}'
        print(f'    {str(label):>10}  {phase:>8}  '
              f'{"IDENT" if ok else "DIFFERS":>9}  {d.max():>11.3e}  '
              f'{d.mean()/scale:>10.3e}  {c:>14.10f}')
    return all_same


def drive(args):
    dev = pick_device(args.device)
    gpu = torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'cpu'
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    tag = dev.type
    print(f'host={os.uname().nodename}  device={dev}  gpu={gpu}  '
          f'torch={torch.__version__}  seed={args.seed}')
    print(f'output -> {out_dir}/\n')

    results = {}

    for mode, title, note in (
            ('forward', 'createCrops forward (sampling only)',
             'must be IDENTICAL on any device; otherwise the seeding is broken'),
            ('backward', 'createCrops backward (gradient)',
             'DIFFERS on CUDA is expected: grid_sample/interpolate backward '
             'use atomicAdd'),
    ):
        print(f'[{mode}] {title}')
        print(f'    ({note})')
        paths = []
        for i in (1, 2):
            p = os.path.join(out_dir, f'{tag}_{mode}{i}.npy')
            spawn(mode, p, args)
            paths.append(p)
        results[mode] = compare_npy(*paths)
        print()

    if args.recon:
        print('[recon] real pipeline (VQGAN + VGG19 + CLIP)')
        print('    (this is the one that matters; images are saved alongside)')
        paths = []
        for i in (1, 2):
            p = os.path.join(out_dir, f'{tag}_recon{i}.npz')
            print(f'  --- run {i}/2 ---', flush=True)
            spawn('recon', p, args)
            paths.append(p)
        results['recon'] = compare_recon(*paths)
        print(f'    images: {out_dir}/{tag}_recon1_sgd.png, '
              f'{tag}_recon1_final.png (and _recon2_)')
        print()

    print('=' * 64)
    for k, v in results.items():
        print(f'  {k:<9} {"reproducible" if v else "NOT bit-identical"}')
    if results.get('forward') and not results.get('backward', True):
        print('\n  読み: 乱数は generator に閉じている（forward が完全一致）。')
        print('        ずれは backward の非決定性で、CPU では消えるはず。')
    return 0 if all(results.values()) else 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto',
                   help='auto uses CUDA when present; cpu is the control')
    p.add_argument('--recon', action='store_true',
                   help='also run the real reconstruction (slow)')
    p.add_argument('--seed', type=int, default=2956797496,
                   help='generator seed (default: the S01/target18/iter1 run seed)')
    p.add_argument('--base-seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=2,
                   help='successive createCrops draws recorded per run; the two runs '
                        'are compared over this whole sequence')
    p.add_argument('--num-crops', type=int, default=32)
    p.add_argument('--no-augment', action='store_true')
    p.add_argument('--list-nondet', action='store_true',
                   help='warn on every op lacking a deterministic kernel')
    p.add_argument('--subject', default='S01')
    p.add_argument('--target', type=int, default=18)
    p.add_argument('--method', default='original_all')
    p.add_argument('--out-dir', default=DEFAULT_OUT_DIR,
                   help='where the .npy / .npz / .png outputs go')
    p.add_argument('--n-sgd', type=int, default=None,
                   help='override numReps_withoutLangevin (CPU runs are slow)')
    p.add_argument('--n-lang', type=int, default=None,
                   help='override numReps_withLangevin')
    # Internal: one worker process produces one file. Not for direct use.
    p.add_argument('--_worker', dest='worker', default=None,
                   choices=['forward', 'backward', 'recon'],
                   help=argparse.SUPPRESS)
    p.add_argument('--_out', dest='out', default=None, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        dev = pick_device(args.device)
        if args.worker == 'recon':
            work_recon(args.out, args, dev)
        else:
            work_crops(args.out, args, dev, args.worker == 'backward')
        return 0
    return drive(args)


if __name__ == '__main__':
    sys.exit(main())
