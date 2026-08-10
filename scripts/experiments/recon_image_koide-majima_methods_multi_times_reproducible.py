"""Seed-reproducible counterpart of recon_image_koide-majima_methods_multi_times_no_seed.py.

The ``_no_seed`` script is left untouched: it uses the original ``recon_func``
and its results vary run to run (that is the point of the variability
experiment). This script is the reproducible one -- it always uses
``recon_func_reproducible``, so the same ``--seed`` gives the same
reconstruction.

Example:
    python scripts/experiments/recon_image_koide-majima_methods_multi_times_reproducible.py \\
        original_all --seed 42 --subjects S01 --targets 18 --iters 1
"""

# %%
import sys
import argparse
import hashlib
import yaml
from PIL import Image
import numpy as np
import torch
import pickle
import scipy
import os

from recon_utils  import get_target_image, convert_featname
import recon_func_reproducible as recon_func


def derive_seed(base_seed, subject, targetID, iter_n):
    """Deterministic per-reconstruction seed.

    Derived from the run's base seed plus the loop indices, so each of the
    (subject x target x iteration) reconstructions gets its own independent but
    reproducible stream. Because the seed depends on the indices rather than on
    the loop order, a run can be sharded across GPUs / restarted partway and
    still produce identical results.
    """
    key = f'{base_seed}|{subject}|{targetID}|{iter_n}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)


def main(reconMethod='original_all', save_base_dir = './test', seed=None,
         subjects=None, targets=None, iters=None):
    # load config
    with open('./scripts/config/demo_params.yaml', 'rb') as f:
        prm_demo = yaml.safe_load(f)
    with open('./scripts/config/config_KS_mod.yaml', 'rb') as f:
        dt_cfg = yaml.safe_load(f)

    # Resolve the seed: explicit --seed wins, otherwise the `seed` entry in
    # config_KS_mod.yaml (default 42). set_seed pins the global RNGs and cuDNN.
    if seed is None:
        seed = dt_cfg.get('seed', 42)
    recon_func.set_seed(seed)
    print(f'[reproducibility] base seed = {seed} (per-run seeds derived '
          'from subject/target/iteration)')

    dir_taming_transformer = dt_cfg['file_path']['taming_transformer_dir']
    sys.path.insert(0, dir_taming_transformer)
    import model_loading  

    # Device
    cudaID = "cuda:0"
    DEVICE = torch.device(cudaID if torch.cuda.is_available() else "cpu")

    # %%
    ## load DNN models
    # load VQGAN model
    config1024 = model_loading.load_config(
        dir_taming_transformer+"/logs/vqgan_imagenet_f16_1024/configs/model.yaml", display=False)
    VQGANmodel1024 = model_loading.load_vqgan(
        config1024, ckpt_path=dir_taming_transformer+"/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt").to(DEVICE)
    VQGANmodel1024.eval()

    # Load VGG19 model
    VGGmodel_, _ = model_loading.load_VGG_model(DEVICE)

    # Load CLIP models to be used.
    # set CLIPmodelName_
    CLIP_modelNames = dt_cfg["models"]["CLIP"]["modelnames"]
    CLIP_modelTypes = dt_cfg["models"]["CLIP"]["modeltypes"]
    CLIP_usedLayer = dt_cfg["models"]["CLIP"]["used_layer"]
    CLIPmodelWeight_ = dt_cfg["models"]["CLIP"]["modelcoefs"]
    CLIPmodel_, nameOfSubdirForCLIPfeature = model_loading.load_CLIP_model(
        CLIP_modelTypes, DEVICE)

    # %%
    subject_list = ['S01', 'S02', 'S03']
    save_subject_list = ['S1', 'S2', 'S3']
    # Restrict the sweep from the CLI (shard across GPUs, or run a small subset
    # for a reproducibility check). The per-run seed is derived from the indices,
    # so a subset produces exactly the same results as the full sweep would.
    if subjects is not None:
        unknown = [s for s in subjects if s not in subject_list]
        if unknown:
            raise ValueError(f'unknown subject(s) {unknown}; choose from {subject_list}')
        save_subject_list = [save_subject_list[subject_list.index(s)] for s in subjects]
        subject_list = list(subjects)
    subject_dict = dict(zip(subject_list, save_subject_list))
    # select from 0 to 24
    # Here are examples:
    # ID 21: 'Bowling ball (artifact)'
    # ID 20: 'Airplane (artifact)'
    # ID 18: 'Leopard (animal)'
    # ID 19: 'Goat (animal)'
    # ID  7: 'Blue + (symbol)'
    # ID 14: 'Black x (symbol)'
    targetID_list = np.arange(25)
    if targets is not None:
        targetID_list = np.array(targets)
    image_label_list = ['Img{:04d}'.format(i) for i in range(1, 27)]

    #reconMethod = 'original' # select from 'original' (default), 'Langevin', 'withoutLangevin'

    # %%
    # perform reconstruction
    torch.cuda.empty_cache()
    # Set parameters

    meanFeatureDir = dt_cfg['file_path']['mean_feat_dir']  
    targetimpath = prm_demo['dt_targetimages_path']
    CLIP_modelNames = dt_cfg["models"]["CLIP"]["modelnames"]
    CLIP_usedLayer = dt_cfg["models"]["CLIP"]["used_layer"]
    CLIPmodelWeight_ = dt_cfg["models"]["CLIP"]["modelcoefs"]

    # Load decfeat
    feat_set = dt_cfg["recon_params"][reconMethod]["feat_set"]
    used_layers_VGG__in = dt_cfg["recon_feat_layers"][feat_set]["VGG19"]
    used_layers_VGG = convert_featname(used_layers_VGG__in, cvt_to='directory')

    # Set parameters
    CLIPcoef_ = dt_cfg['recon_params'][reconMethod]['clip_coef']
    feat_set = dt_cfg['recon_params'][reconMethod]['feat_set']
    disp_every = dt_cfg['recon_params'][reconMethod]['display_every']
    numReps = dt_cfg['recon_params'][reconMethod]['numReps']
    similarity = dt_cfg['recon_params'][reconMethod]['similarity']

    if reconMethod == 'Langevin' or reconMethod == 'original':
        lr_gamma = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_gamma']
        lr_a = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_a']
        lr_b = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_b']
        T_langevin = dt_cfg['recon_params'][reconMethod]['Langevin']['T']
    if reconMethod == 'Langevin' or reconMethod == 'original':
        lr_gamma = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_gamma']
        lr_a = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_a']
        lr_b = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_b']
        T_langevin = dt_cfg['recon_params'][reconMethod]['Langevin']['T']
    try:
        lr_gamma = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_gamma']
        lr_a = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_a']
        lr_b = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_b']
        T_langevin = dt_cfg['recon_params'][reconMethod]['Langevin']['T']
        
        print(lr_gamma)
        print(lr_a)
        print(lr_b)
        print(T_langevin)
    
    except:
        pass
    # set parameters
    numReps_withoutLangevin = dt_cfg['recon_params'][reconMethod]["numReps_withoutLangevin"]#1000 # (default) 1000
    numReps_Langevin = dt_cfg['recon_params'][reconMethod]["numReps_withLangevin"]#500 # (default) 500

        
    iter_num = 10 if iters is None else iters

    # %%
    for j, subject in enumerate(subject_list):
        for i, targetID in enumerate(targetID_list):
            
            for iter_n in range(iter_num):
                save_subject = subject_dict[subject]
                save_dir = f'{save_base_dir}/{save_subject}/iter{iter_n+1:02}/VC'
                os.makedirs(save_dir, exist_ok=True)
                if targetID >14:
                    tid = targetID +1
                else:
                    tid = targetID
                # %%
                targetImg_, targetimname = get_target_image(targetID, targetimpath)
                # Numbered by targetID, not by the loop position, so a --targets
                # subset produces the same filenames as the full sweep.
                recon_name = f"Stim{targetID+1:02}_{targetimname}"
                true_image_dir = f'./data/ImageryDeeprecon/source'
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
                # %%
                # targetCLIPfeature_: 
                # Prepare target CLIP feature (decoded CLIP feature)
                targetCLIPfeature_ = list()
                for mi in range(len(CLIPmodel_)):
                    with open(list_path_clip[mi], 'rb') as f:
                        dt = pickle.load(f)
                    x = dt[0].astype('float32')
                    targetCLIPfeature_.append(torch.tensor(x, dtype=torch.float32).to(DEVICE).unsqueeze(0))
                    del dt, x
                    
                # Set meanCLIPfeature_:
                # Prepare the mean CLIP feature vector, which is used in the normalization (i.e., centering) process.
                meanCLIPfeature_ = list()
                for mi in range(len(CLIPmodel_)):
                    x = scipy.io.loadmat(os.path.join(meanFeatureDir, CLIP_modelNames[mi], CLIP_usedLayer, 'meanFeature_.mat'))
                    meanCLIPfeature_.append(torch.tensor(x['mu'], dtype=torch.float32).to(DEVICE))
                del x
                
                # %%
                # targetVGGfeature_ (decoded VGG feature):
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

                ### Main: Reconstruction -------------------------------------------------
                # %%
                VGGlayerWeight_ = np.ones(len(used_layers_VGG))
                VGGlayerWeight_ = VGGlayerWeight_/VGGlayerWeight_.sum()
                # Set the initial image
                #initialImage_PIL_ = Image.open('./ref_images/uniformGray.tiff')
                initialImage_PIL_ = Image.fromarray(np.uint8(np.ones([240, 240, 3]) * 128))
                
                
                # %%
                # Reconstruction
                #VGGlayerWeight_ = np.ones(len(used_layers_VGG))
                #feat_norm = torch.tensor([torch.linalg.norm(targetVGGfeature_[i]) for i in range(len(targetVGGfeature_))])
                #VGGlayerWeight_ = VGGlayerWeight_/VGGlayerWeight_.sum()
                #VGGlayerWeight_ = 1. / (feat_norm ** 2)
                # Per-reconstruction seed: independent stream for every
                # (subject, target, iteration) so the 10 repeats still differ
                # while each one stays reproducible on its own.
                run_seed = derive_seed(seed, subject, targetID, iter_n)
                print(f'[reproducibility] {subject} target{targetID} '
                      f'iter{iter_n+1} -> seed {run_seed}')

                reconf = recon_func.imageRecon(
                    targetVGGfeature_, meanVGGfeature_, VGGlayerWeight_, VGGmodel_, used_layers_VGG__in,
                    targetCLIPfeature_, meanCLIPfeature_, CLIPmodelWeight_, CLIPmodel_,
                    VQGANmodel1024, initialImage_PIL_, initInputType='PIL',
                    similarity=similarity, disp_every=disp_every, numReps=numReps, CLIPcoef=CLIPcoef_, DEVICE=DEVICE,
                    seed=run_seed
                )
                # %%
                print('Reconstruction without Langevin:')
                generator = reconf.withoutLangevin(numReps=numReps_withoutLangevin, returnVec=True)
                
                currentLatentVec = None
                woLang_time_step_list = []
                loss_vgg_withoutLangevin_list = []
                loss_clip_withoutLangevin_list = []
                if numReps_withoutLangevin > 0:
                    for recImg, time_step, loss_VGG, loss_CLIP, currentLatentVec in generator:
                        print(time_step)
                        print(loss_VGG, loss_CLIP)
                        # add loss
                        woLang_time_step_list.append(time_step)
                        loss_vgg_withoutLangevin_list.append(loss_VGG)
                        loss_clip_withoutLangevin_list.append(loss_CLIP)
                    # save the results
                    save_wo_lang_dir =  f'{save_dir}/wo_lang/'
                    os.makedirs(save_wo_lang_dir, exist_ok=True)
                    image_label = 'Img{:04d}'.format(tid+1)
                    save_file_name = f'{save_wo_lang_dir}/recon_img_normalized-{image_label}.jpg'
                    recImg.save(save_file_name)
                # %% 
                wLang_time_step_list = []
                loss_vgg_withLangevin_list = []
                loss_clip_withLangevin_list = []
                if numReps_Langevin > 0:
                    #generator = reconf.Langevin(initInput=currentLatentVec, initInputType='latentVector', numReps=numReps_Langevin,  returnVec=True)
                    generator = reconf.Langevin(initInput=currentLatentVec,initInputType='latentVector', numReps=numReps_Langevin, returnVec=True,  
                                            lr_gamma=lr_gamma, lr_a=lr_a, lr_b=lr_b, T=T_langevin)
                    for recImg, time_step, loss_VGG, loss_CLIP, currentLatentVec in generator:
                        print(time_step)
                        print(loss_VGG, loss_CLIP)
                        #add loss
                        wLang_time_step_list.append(time_step)
                        loss_vgg_withLangevin_list.append(loss_VGG)
                        loss_clip_withLangevin_list.append(loss_CLIP)
        
                # %%
                
                # save the results
                
                save_file_name = f'{save_dir}/Img_{tid+1:04d}_{targetimname}.pkl'
                save_dict = {
                    # seed actually used for this reconstruction (None if unseeded)
                    'seed': run_seed,
                    'base_seed': seed,
                    # latent vec
                    'latent_vec': currentLatentVec.cpu().detach().numpy(),
                    #time step
                    'woLang_time_step_list': woLang_time_step_list,
                    'wLang_time_step_list': wLang_time_step_list,
                    #loss
                    'loss_vgg_withoutLangevin_list': loss_vgg_withoutLangevin_list,
                    'loss_clip_withoutLangevin_list': loss_clip_withoutLangevin_list,
                    'loss_vgg_withLangevin_list': loss_vgg_withLangevin_list,
                    'loss_clip_withLangevin_list': loss_clip_withLangevin_list,
                }
                with open(save_file_name, 'wb') as f:
                    pickle.dump(save_dict, f)
                # save images
                save_recon_image = f'{save_dir}/{recon_name}.tiff'
                image_label = 'Img{:04d}'.format(tid+1)
                save_name = f'{save_dir}/recon_img_normalized-{image_label}.jpg'
                recImg.save(save_name)
                    
                    
            

if __name__ == '__main__':
    #load argparse
    parser = argparse.ArgumentParser(description="select reconstruction methods, provided by koide-majima")
    parser.add_argument('method', type=str, help='select the method to use', default="original_all",
                        choices=['original_all'
                                 ])
    parser.add_argument('--seed', type=int, default=None,
                        help='base random seed (default: the `seed` entry in '
                             'config_KS_mod.yaml, else 42). Per-run seeds are '
                             'derived from it plus subject/target/iteration.')
    parser.add_argument('--results_dir', type=str, default=None,
                        help='output directory (default: results/rep_recon_image_koide-majima'
                             '_recon_reproducible_seed<N>/<method>)')
    parser.add_argument('--subjects', type=str, nargs='+', default=None,
                        help='subset of subjects to run, e.g. --subjects S01 (default: all)')
    parser.add_argument('--targets', type=int, nargs='+', default=None,
                        help='subset of target IDs (0-24) to run, e.g. --targets 18 (default: all)')
    parser.add_argument('--iters', type=int, default=None,
                        help='number of repeated reconstructions per target (default: 10)')
    args = parser.parse_args()

    reconMethod = args.method

    if args.results_dir is not None:
        save_base_dir = args.results_dir
    else:
        # Separate directory per seed, and separate from the unseeded
        # `..._recon_variability_no_seed` results, which are never overwritten.
        with open('./scripts/config/config_KS_mod.yaml', 'rb') as f:
            _seed = args.seed if args.seed is not None else yaml.safe_load(f).get('seed', 42)
        save_base_dir = (f'./results/rep_recon_image_koide-majima_recon_reproducible'
                         f'_seed{_seed}/{reconMethod}')
    os.makedirs(save_base_dir, exist_ok=True)
    main(reconMethod, save_base_dir, seed=args.seed,
         subjects=args.subjects, targets=args.targets, iters=args.iters)
    
    
