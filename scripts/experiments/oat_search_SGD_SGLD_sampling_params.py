"""One-at-a-time (OAT) sweep over the reconstruction sampling parameters.

Every parameter is held at a BASELINE and one is varied at a time across its value
list. The shared baseline point is deduped, so the count is

    1 (baseline) + sum_p (n_p - 1)

which for the default lists (5,5,5,5,5 + 2) is 1 + (4+4+4+4+4+1) = 22 combos.

Baseline defaults = original_all (Koide-Majima et al., 2024):
    lr_a=0.00015, lr_b=0.15, lr_gamma=0.055, T=1e-6, wL=500, woL=1000

The reconstruction machinery below (model loading, the per-image loop, tagging,
the manifest) previously lived in a separate full-Cartesian-grid script. That
sweep is not part of the manuscript, so only the OAT sweep is kept and the shared
machinery is inlined here rather than imported from a module that is no longer
published.

Two differences from the plain reconstruction scripts, both for long sweeps:
the per-step latent trajectories are not stored unless --save_trajectory is
given (they cost ~600 MB per image), and a per-(combo, subject, image) done
marker lets --resume restart safely.

NOTE on `numReps`: it only feeds imageRecon.__init__ as a default; the real
iteration counts come from numReps_withoutLangevin and numReps_withLangevin, so
sweeping it alone would produce identical output. It stays a single value.

Output default: results/oat_sampling_params.
"""
import argparse
import csv
import os
import pickle
import random
import sys

import numpy as np
import scipy
import torch
import yaml
from PIL import Image

from recon_utils import get_target_label, convert_featname
import recon_func_mod_KS as recon_func

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def load_config():
    with open('./scripts/config/demo_params.yaml', 'rb') as f:
        prm_demo = yaml.safe_load(f)
    with open('./scripts/config/config_KS_mod.yaml', 'rb') as f:
        dt_cfg = yaml.safe_load(f)
    return prm_demo, dt_cfg


def load_models(dt_cfg):
    """Load the heavy models once and return them + CLIP metadata in a dict."""
    dir_taming_transformer = dt_cfg['file_path']['taming_transformer_dir']
    sys.path.insert(0, dir_taming_transformer)
    import model_loading

    cudaID = "cuda:0"
    DEVICE = torch.device(cudaID if torch.cuda.is_available() else "cpu")

    # load VQGAN model
    config1024 = model_loading.load_config(
        dir_taming_transformer + "/logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
    VQGANmodel1024 = model_loading.load_vqgan(
        config1024, ckpt_path=dir_taming_transformer + "/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(DEVICE)
    VQGANmodel1024.eval()

    # Load VGG19 model
    VGGmodel_, _ = model_loading.load_VGG_model(DEVICE)

    # Load CLIP models to be used.
    CLIP_modelNames = dt_cfg["models"]["CLIP"]["modelnames"]
    CLIP_modelTypes = dt_cfg["models"]["CLIP"]["modeltypes"]
    CLIP_usedLayer = dt_cfg["models"]["CLIP"]["used_layer"]
    CLIPmodelWeight_ = dt_cfg["models"]["CLIP"]["modelcoefs"]
    CLIPmodel_, nameOfSubdirForCLIPfeature = model_loading.load_CLIP_model(
        CLIP_modelTypes, DEVICE)

    return {
        'DEVICE': DEVICE,
        'VQGANmodel1024': VQGANmodel1024,
        'VGGmodel_': VGGmodel_,
        'CLIPmodel_': CLIPmodel_,
        'CLIP_modelNames': CLIP_modelNames,
        'CLIP_usedLayer': CLIP_usedLayer,
        'CLIPmodelWeight_': CLIPmodelWeight_,
    }


def run_recon(models, params, save_base_dir, dt_cfg, prm_demo,
              subjects, targetID_list,
              resume=False, save_trajectory=False, trajectory_stride=1,
              errors_log=None):
    """Run the full subject x image reconstruction loop for one parameter set.

    This mirrors the body of the compare script's main(); only the parameter
    source (params dict) and the trajectory/pkl handling differ.
    """
    DEVICE = models['DEVICE']
    VQGANmodel1024 = models['VQGANmodel1024']
    VGGmodel_ = models['VGGmodel_']
    CLIPmodel_ = models['CLIPmodel_']
    CLIP_modelNames = models['CLIP_modelNames']
    CLIP_usedLayer = models['CLIP_usedLayer']
    CLIPmodelWeight_ = models['CLIPmodelWeight_']

    torch.cuda.empty_cache()

    meanFeatureDir = dt_cfg['file_path']['mean_feat_dir']
    targetimpath = prm_demo['dt_targetimages_path']

    # --- parameters (from the grid, not from a YAML preset) -----------------
    feat_set = params['feat_set']
    CLIPcoef_ = params['clip_coef']
    numReps = params['numReps']
    similarity = params['similarity']
    lr_gamma = params['lr_gamma']
    lr_a = params['lr_a']
    lr_b = params['lr_b']
    T_langevin = params['T']
    numReps_withoutLangevin = params['numReps_withoutLangevin']
    numReps_Langevin = params['numReps_withLangevin']

    used_layers_VGG__in = dt_cfg["recon_feat_layers"][feat_set]["VGG19"]
    used_layers_VGG = convert_featname(used_layers_VGG__in, cvt_to='directory')

    # Mirror the compare script's mapping: 'S01' -> 'S1', 'S02' -> 'S2', ...
    save_subject_list = ['S{}'.format(int(s[1:])) if s[1:].isdigit() else s for s in subjects]
    subject_dict = dict(zip(subjects, save_subject_list))

    def _keep_step(step_idx):
        return save_trajectory and (trajectory_stride <= 1 or (step_idx % trajectory_stride == 0))

    # %%
    for subject in subjects:
        for targetID in targetID_list:
            save_subject = subject_dict[subject]
            save_dir = f'{save_base_dir}/{save_subject}/VC'
            os.makedirs(save_dir, exist_ok=True)
            if targetID > 14:
                tid = targetID + 1
            else:
                tid = targetID
            image_label = 'Img{:04d}'.format(tid + 1)
            done_marker = f'{save_dir}/.done_{image_label}'

            if resume and os.path.exists(done_marker):
                continue

            try:
                targetimname = get_target_label(targetID, targetimpath)
                # VGG
                list_path_vgg = list()
                for t_layername in used_layers_VGG:
                    path_decfeat = prm_demo['decfearture_path']
                    path_decfeat = path_decfeat.replace('__subjectname__', subject)
                    path_decfeat = path_decfeat.replace('__modelname__', 'VGG19')
                    path_decfeat = path_decfeat.replace('__layername__', t_layername)
                    path_decfeat = path_decfeat.replace('__targetimname__', targetimname)
                    list_path_vgg.append(path_decfeat)

                # CLIP
                list_path_clip = list()
                for t_modelname in CLIP_modelNames:
                    path_decfeat = prm_demo['decfearture_path']
                    path_decfeat = path_decfeat.replace('__subjectname__', subject)
                    path_decfeat = path_decfeat.replace('__modelname__', t_modelname)
                    path_decfeat = path_decfeat.replace('__layername__', CLIP_usedLayer)
                    path_decfeat = path_decfeat.replace('__targetimname__', targetimname)
                    list_path_clip.append(path_decfeat)

                # targetCLIPfeature_ (decoded CLIP feature)
                targetCLIPfeature_ = list()
                for mi in range(len(CLIPmodel_)):
                    with open(list_path_clip[mi], 'rb') as f:
                        dt = pickle.load(f)
                    x = dt[0].astype('float32')
                    targetCLIPfeature_.append(torch.tensor(x, dtype=torch.float32).to(DEVICE).unsqueeze(0))
                    del dt, x

                # meanCLIPfeature_ (for centering)
                meanCLIPfeature_ = list()
                for mi in range(len(CLIPmodel_)):
                    x = scipy.io.loadmat(os.path.join(meanFeatureDir, CLIP_modelNames[mi], CLIP_usedLayer, 'meanFeature_.mat'))
                    meanCLIPfeature_.append(torch.tensor(x['mu'], dtype=torch.float32).to(DEVICE))
                del x

                # targetVGGfeature_ (decoded VGG feature)
                targetVGGfeature_ = list()
                for li in range(len(used_layers_VGG)):
                    with open(list_path_vgg[li], 'rb') as f:
                        dt = pickle.load(f)
                    x = dt[0].astype('float32')
                    targetVGGfeature_.append(torch.tensor(x, dtype=torch.float32).to(DEVICE).unsqueeze(0))
                    del dt, x

                meanVGGfeature_ = list()
                for li in range(len(used_layers_VGG)):
                    x = scipy.io.loadmat(os.path.join(meanFeatureDir, 'VGG19', used_layers_VGG[li], 'meanFeature_.mat'))
                    meanVGGfeature_.append(torch.tensor(x['mu'], dtype=torch.float32).to(DEVICE))
                    del x

                ### Main: Reconstruction ---------------------------------------
                VGGlayerWeight_ = np.ones(len(used_layers_VGG))
                VGGlayerWeight_ = VGGlayerWeight_ / VGGlayerWeight_.sum()

                # Set the initial image (uniform gray)
                initialImage_PIL_ = Image.fromarray(np.uint8(np.ones([240, 240, 3]) * 128))

                reconf = recon_func.imageRecon(
                    targetVGGfeature_, meanVGGfeature_, VGGlayerWeight_, VGGmodel_, used_layers_VGG__in,
                    targetCLIPfeature_, meanCLIPfeature_, CLIPmodelWeight_, CLIPmodel_,
                    VQGANmodel1024, initialImage_PIL_, initInputType='PIL',
                    similarity=similarity, disp_every=1, numReps=numReps, CLIPcoef=CLIPcoef_, DEVICE=DEVICE
                )

                print('Reconstruction without Langevin:')
                generator = reconf.withoutLangevin(numReps=numReps_withoutLangevin, returnVec=True)

                currentLatentVec = None
                currentLatentVec_SGLD = None
                currentLatentVec_SGD = None
                recImg = None
                recImg_SGD = None

                woLang_time_step_list = []
                loss_vgg_withoutLangevin_list = []
                loss_clip_withoutLangevin_list = []

                SGLD_currentLatentVec_list = []
                SGD_currentLatentVec_list = []

                if numReps_withoutLangevin > 0:
                    for recImg, time_step, loss_VGG, loss_CLIP, currentLatentVec in generator:
                        woLang_time_step_list.append(time_step)
                        loss_vgg_withoutLangevin_list.append(loss_VGG)
                        loss_clip_withoutLangevin_list.append(loss_CLIP)
                        if _keep_step(time_step):
                            SGLD_currentLatentVec_list.append(currentLatentVec.detach().cpu().numpy())
                            SGD_currentLatentVec_list.append(currentLatentVec.detach().cpu().numpy())
                    # save the without-Langevin image
                    save_wo_lang_dir = f'{save_dir}/wo_lang/'
                    os.makedirs(save_wo_lang_dir, exist_ok=True)
                    save_file_name = f'{save_wo_lang_dir}/recon_img_normalized-{image_label}.jpg'
                    recImg.save(save_file_name)

                wLang_time_step_list = []
                loss_vgg_withLangevin_list = []
                loss_clip_withLangevin_list = []

                SGD_time_step_list = []
                loss_vgg_SGD_list = []
                loss_clip_SGD_list = []

                if numReps_Langevin > 0:
                    generator = reconf.Langevin(initInput=currentLatentVec, initInputType='latentVector', numReps=numReps_Langevin, returnVec=True,
                                                lr_gamma=lr_gamma, lr_a=lr_a, lr_b=lr_b, T=T_langevin)
                    for recImg, time_step, loss_VGG, loss_CLIP, currentLatentVec_SGLD in generator:
                        wLang_time_step_list.append(time_step)
                        loss_vgg_withLangevin_list.append(loss_VGG)
                        loss_clip_withLangevin_list.append(loss_CLIP)
                        if _keep_step(time_step):
                            SGLD_currentLatentVec_list.append(currentLatentVec_SGLD.detach().cpu().numpy())

                    generator_SGD = reconf.SGD(initInput=currentLatentVec, initInputType='latentVector', numReps=numReps_Langevin, returnVec=True,
                                               lr_gamma=lr_gamma, lr_a=lr_a, lr_b=lr_b, T=T_langevin)
                    for recImg_SGD, time_step_SGD, loss_VGG_SGD, loss_CLIP_SGD, currentLatentVec_SGD in generator_SGD:
                        SGD_time_step_list.append(time_step_SGD)
                        loss_vgg_SGD_list.append(loss_VGG_SGD)
                        loss_clip_SGD_list.append(loss_CLIP_SGD)
                        if _keep_step(time_step_SGD):
                            SGD_currentLatentVec_list.append(currentLatentVec_SGD.detach().cpu().numpy())

                # %% Save results ------------------------------------------------
                save_file_name = f'{save_dir}/{image_label}.pkl'
                def to_np(t):
                    return None if t is None else t.cpu().detach().numpy()

                save_dict = {
                    'latent_vec_adam': to_np(currentLatentVec),
                    'woLang_time_step_list': woLang_time_step_list,
                    'wLang_time_step_list': wLang_time_step_list,
                    'SGD_time_step_list': SGD_time_step_list,
                    'loss_vgg_withoutLangevin_list': loss_vgg_withoutLangevin_list,
                    'loss_clip_withoutLangevin_list': loss_clip_withoutLangevin_list,
                    'loss_vgg_withLangevin_list': loss_vgg_withLangevin_list,
                    'loss_clip_withLangevin_list': loss_clip_withLangevin_list,
                    'loss_vgg_SGD_list': loss_vgg_SGD_list,
                    'loss_clip_SGD_list': loss_clip_SGD_list,
                    # record the parameters that produced this reconstruction
                    'params': dict(params),
                }
                # trajectory lists are dropped by default (grid-search light mode)
                if save_trajectory:
                    save_dict['SGLD_currentLatentVec_list'] = SGLD_currentLatentVec_list
                    save_dict['SGD_currentLatentVec_list'] = SGD_currentLatentVec_list

                if currentLatentVec_SGLD is not None:
                    save_dict['latent_vec_SGLD'] = currentLatentVec_SGLD.cpu().detach().numpy()

                if currentLatentVec_SGD is not None:
                    save_dict['latent_vec_SGD'] = currentLatentVec_SGD.cpu().detach().numpy()

                    with open(save_file_name, 'wb') as f:
                        pickle.dump(save_dict, f)

                    # save final images
                    save_name = f'{save_dir}/recon_img_normalized-{image_label}.jpg'
                    recImg.save(save_name)

                    sgd_save_dir = os.path.join(save_dir, 'SGD')
                    os.makedirs(sgd_save_dir, exist_ok=True)
                    save_name = f'{sgd_save_dir}/recon_img_normalized-{image_label}.jpg'
                    recImg_SGD.save(save_name)

                    # save posterior samples (10)
                    for kk in range(10):
                        generator_ = reconf.Langevin(initInput=currentLatentVec, initInputType='latentVector', numReps=1, returnVec=True,
                                                     lr_gamma=0, lr_a=0.1, lr_b=lr_b + 1e-8, T=T_langevin)
                        for recImg_s, _, _, _, _ in generator_:
                            save_dir_ = f'{save_dir}/sampling_{kk:02}'
                            os.makedirs(save_dir_, exist_ok=True)
                            save_name = f'{save_dir_}/recon_img_normalized-{image_label}.jpg'
                            recImg_s.save(save_name)
                else:
                    # No Langevin/SGD phase (numReps_withLangevin == 0): still save the
                    # lightweight pkl so the grid point has a persisted result.
                    with open(save_file_name, 'wb') as f:
                        pickle.dump(save_dict, f)

                # mark this (combo, subject, image) as done for --resume
                with open(done_marker, 'w') as f:
                    f.write('done\n')

            except Exception as e:  # noqa: BLE001 - one failure must not kill the grid
                msg = f'{save_base_dir}\t{subject}\t{image_label}\t{type(e).__name__}: {e}'
                print(f'[ERROR] {msg}', file=sys.stderr)
                if errors_log is not None:
                    with open(errors_log, 'a') as f:
                        f.write(msg + '\n')
                torch.cuda.empty_cache()
                continue

def fmt(v):
    """Filesystem-friendly compact string for a numeric grid value."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


GRID_KEYS = ['lr_a', 'lr_b', 'lr_gamma', 'T',
             'numReps_withLangevin', 'numReps_withoutLangevin']


def make_tag(params):
    return (f"lr_a{fmt(params['lr_a'])}_lr_b{fmt(params['lr_b'])}"
            f"_g{fmt(params['lr_gamma'])}_T{fmt(params['T'])}"
            f"_woL{params['numReps_withoutLangevin']}_wL{params['numReps_withLangevin']}"
            f"_nR{params['numReps']}")


def write_manifest(path, grid_keys, combos, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ['index', 'tag'] + grid_keys + ['numReps', 'clip_coef', 'feat_set']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for idx, combo in enumerate(combos):
            params = dict(zip(grid_keys, combo))
            params.update(numReps=args.numReps, clip_coef=args.clip_coef, feat_set=args.feat_set)
            tag = make_tag(params)
            w.writerow([idx, tag] + [params[k] for k in grid_keys] +
                       [args.numReps, args.clip_coef, args.feat_set])


def build_oat(args):
    """OAT combos: baseline first, then vary each parameter across its list."""
    grid_keys = GRID_KEYS
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


def build_slice(args):
    """The 5x5 lr_a x T plane, every other parameter held at the baseline.

    The OAT sweep only tests the cross through this plane (the row at the baseline
    lr_a and the column at the baseline T), so it never varies the two together --
    the one direction in which the SGLD step size and temperature interact, since
    they scale the gradient term by eps_t / T and the noise term by sqrt(eps_t).
    The two sweeps share those 9 cross conditions, giving 22 + 25 - 9 = 38 unique
    settings in total.
    """
    baseline = {
        'lr_b': args.baseline_lr_b, 'lr_gamma': args.baseline_lr_gamma,
        'numReps_withLangevin': args.baseline_numReps_withLangevin,
        'numReps_withoutLangevin': args.baseline_numReps_withoutLangevin,
    }
    combos = [
        tuple(dict(baseline, lr_a=lr_a, T=T)[k] for k in GRID_KEYS)
        for lr_a in args.lr_a
        for T in args.T
    ]
    return GRID_KEYS, combos


BUILDERS = {'oat': build_oat, 'slice': build_slice}
DEFAULT_OUT = {'oat': './results/oat_sampling_params',
               'slice': './results/lr_a_T_slice'}


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
    parser.add_argument('--mode', choices=sorted(BUILDERS), default='oat',
                        help="'oat' varies one parameter at a time (22 conditions); "
                             "'slice' is the 5x5 lr_a x T plane (25). Figure A5 uses "
                             "their union of 38 settings, so run both.")
    parser.add_argument('--out', type=str, default=None,
                        help='output root (default: depends on --mode)')
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

    prm_demo, dt_cfg = load_config()

    args.out = args.out or DEFAULT_OUT[args.mode]
    grid_keys, combos = BUILDERS[args.mode](args)
    total = len(combos)
    os.makedirs(args.out, exist_ok=True)

    # manifest: only shard 0 (or dry-run) writes it to avoid concurrent clobber
    manifest_path = os.path.join(args.out, f'{args.mode}_manifest.csv')
    if args.shard_id == 0 or args.dry_run:
        write_manifest(manifest_path, grid_keys, combos, args)
        print(f'[manifest] wrote {total} {args.mode} combos to {manifest_path}')

    my_indices = [idx for idx in range(total) if idx % args.num_shards == args.shard_id]
    print(f'[shard {args.shard_id}/{args.num_shards}] handling {len(my_indices)}/{total} combos '
          f'| subjects={args.subjects} targetID={args.targetID}')

    if args.dry_run:
        for idx in my_indices:
            params = dict(zip(grid_keys, combos[idx]))
            params.update(numReps=args.numReps, clip_coef=args.clip_coef, feat_set=args.feat_set)
            tag = make_tag(params)
            print(f'[{idx}/{total}] {tag} -> {args.out}/{tag}')
        print(f'[dry-run] {len(my_indices)} combos for this shard '
              f'({len(my_indices) * len(args.subjects) * len(args.targetID)} reconstructions)')
        sys.exit(0)

    # load heavy models ONCE (reused from the grid module)
    models = load_models(dt_cfg)

    done_log = os.path.join(args.out, f'done.shard{args.shard_id}.log')
    errors_log = os.path.join(args.out, f'errors.shard{args.shard_id}.log')

    for n, idx in enumerate(my_indices):
        params = dict(zip(grid_keys, combos[idx]))
        params.update(
            numReps=args.numReps, clip_coef=args.clip_coef, feat_set=args.feat_set,
            display_every=args.display_every, similarity=args.similarity,
        )
        tag = make_tag(params)
        save_base_dir = f'{args.out}/{tag}'
        print(f'[shard {args.shard_id}] [{n+1}/{len(my_indices)}] (global {idx}/{total}) tag={tag}')

        run_recon(models, params, save_base_dir, dt_cfg, prm_demo,
                     subjects=args.subjects, targetID_list=args.targetID,
                     resume=args.resume, save_trajectory=args.save_trajectory,
                     trajectory_stride=args.trajectory_stride, errors_log=errors_log)

        with open(done_log, 'a') as f:
            f.write(f'{idx}\t{tag}\n')
