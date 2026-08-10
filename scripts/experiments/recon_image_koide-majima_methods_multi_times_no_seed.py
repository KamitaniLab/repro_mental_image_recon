# %%
import argparse
import os
import pickle

import numpy as np
import scipy
import torch
import yaml
from PIL import Image
from recon_utils import convert_featname, get_target_label

from repro_mental_image_recon.recon import func_mod as recon_func

RESULT_ROOT = "./results/rep_recon_image_koide-majima_recon_variability_no_seed"

# Subject id in the decoded-feature tree -> subject id used in the output tree.
SUBJECT_DIRNAME = {"S01": "S1", "S02": "S2", "S03": "S3"}
DEFAULT_SUBJECTS = ("S01", "S02", "S03")
DEFAULT_TARGET_IDS = tuple(range(25))
DEFAULT_ITERS = 10


def main(
    reconMethod="original_all",
    save_base_dir=f"{RESULT_ROOT}/original_all",
    subjects=None,
    targets=None,
    iters=None,
):
    # load config
    with open("./scripts/config/demo_params.yaml", "rb") as f:
        prm_demo = yaml.safe_load(f)
    with open("./scripts/config/config_recon.yaml", "rb") as f:
        dt_cfg = yaml.safe_load(f)

    dir_taming_transformer = dt_cfg["file_path"]["taming_transformer_dir"]
    import model_loading

    # Device
    cudaID = "cuda:0"
    DEVICE = torch.device(cudaID if torch.cuda.is_available() else "cpu")

    # %%
    ## load DNN models
    # load VQGAN model
    config1024 = model_loading.load_config(
        dir_taming_transformer + "/logs/vqgan_imagenet_f16_1024/configs/model.yaml",
        display=False,
    )
    VQGANmodel1024 = model_loading.load_vqgan(
        config1024,
        ckpt_path=dir_taming_transformer
        + "/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt",
    ).to(DEVICE)
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
        CLIP_modelTypes, DEVICE
    )

    # %%
    subject_list = list(subjects) if subjects else list(DEFAULT_SUBJECTS)
    unknown = [s for s in subject_list if s not in SUBJECT_DIRNAME]
    if unknown:
        raise ValueError(
            f"unknown subjects {unknown}; choose from {sorted(SUBJECT_DIRNAME)}"
        )
    # select from 0 to 24
    # Here are examples:
    # ID 21: 'Bowling ball (artifact)'
    # ID 20: 'Airplane (artifact)'
    # ID 18: 'Leopard (animal)'
    # ID 19: 'Goat (animal)'
    # ID  7: 'Blue + (symbol)'
    # ID 14: 'Black x (symbol)'
    targetID_list = list(targets) if targets is not None else list(DEFAULT_TARGET_IDS)
    out_of_range = [t for t in targetID_list if not 0 <= t < len(DEFAULT_TARGET_IDS)]
    if out_of_range:
        raise ValueError(f"target ids out of range {out_of_range}; expected 0..24")

    # reconMethod = 'original' # select from 'original' (default), 'Langevin', 'withoutLangevin'

    # %%
    # perform reconstruction
    torch.cuda.empty_cache()
    # Set parameters

    meanFeatureDir = dt_cfg["file_path"]["mean_feat_dir"]
    targetimpath = prm_demo["dt_targetimages_path"]
    CLIP_modelNames = dt_cfg["models"]["CLIP"]["modelnames"]
    CLIP_usedLayer = dt_cfg["models"]["CLIP"]["used_layer"]
    CLIPmodelWeight_ = dt_cfg["models"]["CLIP"]["modelcoefs"]

    # Load decfeat
    feat_set = dt_cfg["recon_params"][reconMethod]["feat_set"]
    used_layers_VGG__in = dt_cfg["recon_feat_layers"][feat_set]["VGG19"]
    used_layers_VGG = convert_featname(used_layers_VGG__in, cvt_to="directory")

    # Set parameters
    CLIPcoef_ = dt_cfg["recon_params"][reconMethod]["clip_coef"]
    feat_set = dt_cfg["recon_params"][reconMethod]["feat_set"]

    numReps = dt_cfg["recon_params"][reconMethod]["numReps"]
    similarity = dt_cfg["recon_params"][reconMethod]["similarity"]

    try:
        langevin = dt_cfg["recon_params"][reconMethod]["Langevin"]
        lr_gamma = langevin["lr_gamma"]
        lr_a = langevin["lr_a"]
        lr_b = langevin["lr_b"]
        T_langevin = langevin["T"]
    except KeyError as e:
        raise ValueError(f"Langevin config does not include: {e}")
    print(f"Langevin: lr_gamma={lr_gamma} lr_a={lr_a} lr_b={lr_b} T={T_langevin}")
    # set parameters
    numReps_withoutLangevin = dt_cfg["recon_params"][reconMethod][
        "numReps_withoutLangevin"
    ]  # 1000 # (default) 1000
    numReps_Langevin = dt_cfg["recon_params"][reconMethod][
        "numReps_withLangevin"
    ]  # 500 # (default) 500

    iter_num = DEFAULT_ITERS if iters is None else iters

    # %%
    for subject in subject_list:
        for targetID in targetID_list:
            for iter_n in range(iter_num):
                save_subject = SUBJECT_DIRNAME[subject]
                save_dir = f"{save_base_dir}/{save_subject}/iter{iter_n + 1:02}/VC"
                os.makedirs(save_dir, exist_ok=True)
                if targetID > 14:
                    tid = targetID + 1
                else:
                    tid = targetID
                # %%
                targetimname = get_target_label(targetID, targetimpath)
                # VGG
                list_path_vgg = list()
                for t_layername in used_layers_VGG:
                    path_decfeat = prm_demo["decfearture_path"]
                    path_decfeat = path_decfeat.replace("__subjectname__", subject)
                    path_decfeat = path_decfeat.replace("__modelname__", "VGG19")
                    path_decfeat = path_decfeat.replace("__layername__", t_layername)
                    path_decfeat = path_decfeat.replace(
                        "__targetimname__", targetimname
                    )
                    list_path_vgg.append(path_decfeat)

                # CLIP
                list_path_clip = list()
                for t_modelname in CLIP_modelNames:
                    path_decfeat = prm_demo["decfearture_path"]
                    path_decfeat = path_decfeat.replace("__subjectname__", subject)
                    path_decfeat = path_decfeat.replace("__modelname__", t_modelname)
                    path_decfeat = path_decfeat.replace("__layername__", CLIP_usedLayer)
                    path_decfeat = path_decfeat.replace(
                        "__targetimname__", targetimname
                    )
                    list_path_clip.append(path_decfeat)
                # %%
                # targetCLIPfeature_:
                # Prepare target CLIP feature (decoded CLIP feature)
                targetCLIPfeature_ = list()
                for mi in range(len(CLIPmodel_)):
                    with open(list_path_clip[mi], "rb") as f:
                        dt = pickle.load(f)
                    x = dt[0].astype("float32")
                    targetCLIPfeature_.append(
                        torch.tensor(x, dtype=torch.float32).to(DEVICE).unsqueeze(0)
                    )
                    del dt, x

                # Set meanCLIPfeature_:
                # Prepare the mean CLIP feature vector, which is used in the normalization (i.e., centering) process.
                meanCLIPfeature_ = list()
                for mi in range(len(CLIPmodel_)):
                    x = scipy.io.loadmat(
                        os.path.join(
                            meanFeatureDir,
                            CLIP_modelNames[mi],
                            CLIP_usedLayer,
                            "meanFeature_.mat",
                        )
                    )
                    meanCLIPfeature_.append(
                        torch.tensor(x["mu"], dtype=torch.float32).to(DEVICE)
                    )
                del x

                # %%
                # targetVGGfeature_ (decoded VGG feature):
                targetVGGfeature_ = list()
                for li in range(len(used_layers_VGG)):
                    with open(list_path_vgg[li], "rb") as f:
                        dt = pickle.load(f)
                    x = dt[0].astype("float32")
                    targetVGGfeature_.append(
                        torch.tensor(x, dtype=torch.float32).to(DEVICE).unsqueeze(0)
                    )
                    del dt, x

                meanVGGfeature_ = list()
                for li in range(len(used_layers_VGG)):
                    x = scipy.io.loadmat(
                        os.path.join(
                            meanFeatureDir,
                            "VGG19",
                            used_layers_VGG[li],
                            "meanFeature_.mat",
                        )
                    )
                    meanVGGfeature_.append(
                        torch.tensor(x["mu"], dtype=torch.float32).to(DEVICE)
                    )
                    del x

                ### Main: Reconstruction -------------------------------------------------
                # %%
                VGGlayerWeight_ = np.ones(len(used_layers_VGG))
                VGGlayerWeight_ = VGGlayerWeight_ / VGGlayerWeight_.sum()
                # Set the initial image
                # initialImage_PIL_ = Image.open('./ref_images/uniformGray.tiff')
                initialImage_PIL_ = Image.fromarray(
                    np.uint8(np.ones([240, 240, 3]) * 128)
                )

                # %%
                # Reconstruction
                # VGGlayerWeight_ = np.ones(len(used_layers_VGG))
                # feat_norm = torch.tensor([torch.linalg.norm(targetVGGfeature_[i]) for i in range(len(targetVGGfeature_))])
                # VGGlayerWeight_ = VGGlayerWeight_/VGGlayerWeight_.sum()
                # VGGlayerWeight_ = 1. / (feat_norm ** 2)
                reconf = recon_func.imageRecon(
                    targetVGGfeature_,
                    meanVGGfeature_,
                    VGGlayerWeight_,
                    VGGmodel_,
                    used_layers_VGG__in,
                    targetCLIPfeature_,
                    meanCLIPfeature_,
                    CLIPmodelWeight_,
                    CLIPmodel_,
                    VQGANmodel1024,
                    initialImage_PIL_,
                    initInputType="PIL",
                    similarity=similarity,
                    disp_every=1,
                    numReps=numReps,
                    CLIPcoef=CLIPcoef_,
                    DEVICE=DEVICE,
                )
                # %%
                print("Reconstruction without Langevin:")
                generator = reconf.withoutLangevin(
                    numReps=numReps_withoutLangevin, returnVec=True
                )

                currentLatentVec = None
                woLang_time_step_list = []
                loss_vgg_withoutLangevin_list = []
                loss_clip_withoutLangevin_list = []
                total_loss_withoutLangevin_list = []
                currentLatentVec_list = [currentLatentVec]
                currentImg_list = []
                if numReps_withoutLangevin > 0:
                    for (
                        recImg,
                        time_step,
                        loss_VGG,
                        loss_CLIP,
                        currentLatentVec,
                    ) in generator:
                        print(time_step)
                        print(loss_VGG, loss_CLIP)
                        # add loss
                        woLang_time_step_list.append(time_step)
                        loss_vgg_withoutLangevin_list.append(loss_VGG)
                        loss_clip_withoutLangevin_list.append(loss_CLIP)
                        currentLatentVec_list.append(
                            currentLatentVec.detach().cpu().numpy()
                        )
                        currentImg_list.append(np.array(recImg))
                        total_loss = loss_VGG + loss_CLIP * CLIPcoef_
                        total_loss_withoutLangevin_list.append(total_loss)
                    # save the results
                    save_wo_lang_dir = f"{save_dir}/wo_lang/"
                    os.makedirs(save_wo_lang_dir, exist_ok=True)
                    image_label = f"Img{tid + 1:04d}"
                    save_file_name = (
                        f"{save_wo_lang_dir}/recon_img_normalized-{image_label}.jpg"
                    )
                    recImg.save(save_file_name)
                # %%
                wLang_time_step_list = []
                loss_vgg_withLangevin_list = []
                loss_clip_withLangevin_list = []
                total_loss_withLangevin_list = []

                current_LatentVec_withLangevin_list = []
                currentImg_withLangevin_list = []
                if numReps_Langevin > 0:
                    # generator = reconf.Langevin(initInput=currentLatentVec, initInputType='latentVector', numReps=numReps_Langevin,  returnVec=True)
                    generator = reconf.Langevin(
                        initInput=currentLatentVec,
                        initInputType="latentVector",
                        numReps=numReps_Langevin,
                        returnVec=True,
                        lr_gamma=lr_gamma,
                        lr_a=lr_a,
                        lr_b=lr_b,
                        T=T_langevin,
                    )
                    for (
                        recImg,
                        time_step,
                        loss_VGG,
                        loss_CLIP,
                        currentLatentVec,
                    ) in generator:
                        print(time_step)
                        print(loss_VGG, loss_CLIP)
                        # add loss
                        wLang_time_step_list.append(time_step)
                        loss_vgg_withLangevin_list.append(loss_VGG)
                        loss_clip_withLangevin_list.append(loss_CLIP)
                        total_loss = loss_VGG + loss_CLIP * CLIPcoef_
                        total_loss_withLangevin_list.append(total_loss)
                        currentLatentVec_list.append(
                            currentLatentVec.detach().cpu().numpy()
                        )
                        currentImg_list.append(np.array(recImg))
                        current_LatentVec_withLangevin_list.append(
                            currentLatentVec.detach().cpu().numpy()
                        )
                        currentImg_withLangevin_list.append(np.array(recImg))

                # %%

                # save the results

                save_file_name = f"{save_dir}/Img_{tid + 1:04d}_{targetimname}.pkl"
                save_dict = {
                    # latent vec
                    "latent_vec": currentLatentVec.cpu().detach().numpy(),
                    # time step
                    "woLang_time_step_list": woLang_time_step_list,
                    "wLang_time_step_list": wLang_time_step_list,
                    # loss
                    "loss_vgg_withoutLangevin_list": loss_vgg_withoutLangevin_list,
                    "loss_clip_withoutLangevin_list": loss_clip_withoutLangevin_list,
                    "total_loss_witoutLangevin_list": total_loss_withoutLangevin_list,
                    "loss_vgg_withLangevin_list": loss_vgg_withLangevin_list,
                    "loss_clip_withLangevin_list": loss_clip_withLangevin_list,
                    "total_loss_withLangevin_list": total_loss_withLangevin_list,
                    "currentLatentVec_list": currentLatentVec_list,
                    "currentImg_list": currentImg_list,
                    "current_LatentVec_withLangevin_list": current_LatentVec_withLangevin_list,
                    "currentImg_withLangevin_list": currentImg_withLangevin_list,
                }
                with open(save_file_name, "wb") as f:
                    pickle.dump(save_dict, f)
                # save images
                image_label = f"Img{tid + 1:04d}"
                save_name = f"{save_dir}/recon_img_normalized-{image_label}.jpg"
                recImg.save(save_name)


if __name__ == "__main__":
    # load argparse
    parser = argparse.ArgumentParser(
        description="select reconstruction methods, provided by koide-majima"
    )
    parser.add_argument(
        "method",
        type=str,
        help="select the method to use",
        default="original_all",
        choices=["original_all"],
    )
    # Defaults reproduce the published run (3 subjects x 25 stimuli x 10 repeats);
    # the flags exist to shard it across GPUs or to redo part of it.
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="+",
        default=None,
        help=f"subjects to run (default: {' '.join(DEFAULT_SUBJECTS)})",
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=None,
        help="target ids 0-24 to run (default: all 25)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help=f"number of seed-free repeats per reconstruction (default: {DEFAULT_ITERS})",
    )
    args = parser.parse_args()

    reconMethod = args.method
    save_base_dir = f"{RESULT_ROOT}/{reconMethod}"
    os.makedirs(save_base_dir, exist_ok=True)
    main(
        reconMethod,
        save_base_dir,
        subjects=args.subjects,
        targets=args.targets,
        iters=args.iters,
    )
