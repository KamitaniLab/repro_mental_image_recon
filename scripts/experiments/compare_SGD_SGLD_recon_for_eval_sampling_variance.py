# %%
import argparse
import os
import pickle
import random

import numpy as np
import scipy
import torch
import yaml
from PIL import Image
from recon_utils import convert_featname, get_target_label

from repro_mental_image_recon.recon import func_mod as recon_func

RESULT_ROOT = (
    "./results/rep_recon_image_koide-majima_comparing_SGD_updated_sampling_parameters"
)
# Independent one-step SGLD draws saved per reconstruction, as sampling_00 .. sampling_NN.
N_SAMPLING_DRAWS = 10

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# %%
def main(reconMethod="original_all", save_base_dir=f"{RESULT_ROOT}/original_all"):
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
    subject_list = ["S01", "S02", "S03"]
    save_subject_list = ["S1", "S2", "S3"]
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
        lr_gamma = dt_cfg["recon_params"][reconMethod]["Langevin"]["lr_gamma"]
        lr_a = dt_cfg["recon_params"][reconMethod]["Langevin"]["lr_a"]
        lr_b = dt_cfg["recon_params"][reconMethod]["Langevin"]["lr_b"]
        T_langevin = dt_cfg["recon_params"][reconMethod]["Langevin"]["T"]
    except KeyError as e:
        raise ValueError(f"Langevin config does not include: {e}")

    # set parameters
    numReps_withoutLangevin = dt_cfg["recon_params"][reconMethod][
        "numReps_withoutLangevin"
    ]  # 1000 # (default) 1000
    numReps_Langevin = dt_cfg["recon_params"][reconMethod][
        "numReps_withLangevin"
    ]  # 500 # (default) 500

    # %%
    for subject in subject_list:
        for targetID in targetID_list:
            save_subject = subject_dict[subject]
            save_dir = f"{save_base_dir}/{save_subject}/VC"
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
                path_decfeat = path_decfeat.replace("__targetimname__", targetimname)
                list_path_vgg.append(path_decfeat)

            # CLIP
            list_path_clip = list()
            for t_modelname in CLIP_modelNames:
                path_decfeat = prm_demo["decfearture_path"]
                path_decfeat = path_decfeat.replace("__subjectname__", subject)
                path_decfeat = path_decfeat.replace("__modelname__", t_modelname)
                path_decfeat = path_decfeat.replace("__layername__", CLIP_usedLayer)
                path_decfeat = path_decfeat.replace("__targetimname__", targetimname)
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
                        meanFeatureDir, "VGG19", used_layers_VGG[li], "meanFeature_.mat"
                    )
                )
                meanVGGfeature_.append(
                    torch.tensor(x["mu"], dtype=torch.float32).to(DEVICE)
                )
                del x

            ### Main: Reconstruction -------------------------------------------------
            VGGlayerWeight_ = np.ones(len(used_layers_VGG))
            VGGlayerWeight_ = VGGlayerWeight_ / VGGlayerWeight_.sum()
            # %%
            # Set the initial image
            # initialImage_PIL_ = Image.open('./ref_images/uniformGray.tiff')
            initialImage_PIL_ = Image.fromarray(np.uint8(np.ones([240, 240, 3]) * 128))

            # %%
            # Reconstruction
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

            SGLD_currentLatentVec_list = []
            SGD_currentLatentVec_list = []

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
                    SGLD_currentLatentVec_list.append(
                        currentLatentVec.detach().cpu().numpy()
                    )
                    SGD_currentLatentVec_list.append(
                        currentLatentVec.detach().cpu().numpy()
                    )
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

            SGD_time_step_list = []
            loss_vgg_SGD_list = []
            loss_clip_SGD_list = []

            # Set only when the Langevin phase runs; kept explicit so the saving code
            # below does not have to inspect locals() to find out.
            currentLatentVec_SGLD = None
            currentLatentVec_SGD = None
            recImg_SGD = None

            if numReps_Langevin > 0:
                # generator = reconf.Langevin(initInput=currentLatentVec, initInputType='latentVector', numReps=numReps_Langevin, returnVec=True)
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
                    currentLatentVec_SGLD,
                ) in generator:
                    print(time_step)
                    print(loss_VGG, loss_CLIP)
                    # add loss
                    wLang_time_step_list.append(time_step)
                    loss_vgg_withLangevin_list.append(loss_VGG)
                    loss_clip_withLangevin_list.append(loss_CLIP)

                    SGLD_currentLatentVec_list.append(
                        currentLatentVec_SGLD.detach().cpu().numpy()
                    )

                generator_SGD = reconf.SGD(
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
                    recImg_SGD,
                    time_step_SGD,
                    loss_VGG_SGD,
                    loss_CLIP_SGD,
                    currentLatentVec_SGD,
                ) in generator_SGD:
                    print(time_step_SGD)
                    print(loss_VGG_SGD, loss_CLIP_SGD)
                    # add loss
                    SGD_time_step_list.append(time_step_SGD)
                    loss_vgg_SGD_list.append(loss_VGG_SGD)
                    loss_clip_SGD_list.append(loss_CLIP_SGD)

                    SGD_currentLatentVec_list.append(
                        currentLatentVec_SGD.detach().cpu().numpy()
                    )

            # %%
            # %%
            image_label = f"Img{tid + 1:04d}"
            save_file_name = f"{save_dir}/{image_label}.pkl"

            def to_np(t):
                return None if t is None else t.cpu().detach().numpy()

            save_dict = {
                # latent vec
                "latent_vec_adam": to_np(currentLatentVec),
                #'latent_vec_SGLD': currentLatentVec_SGLD.cpu().detach().numpy(),
                #'latent_vec_SGD': currentLatentVec_SGD.cpu().detach().numpy(),
                # time step
                "woLang_time_step_list": woLang_time_step_list,
                "wLang_time_step_list": wLang_time_step_list,
                "SGD_time_step_list": SGD_time_step_list,
                # loss
                "loss_vgg_withoutLangevin_list": loss_vgg_withoutLangevin_list,
                "loss_clip_withoutLangevin_list": loss_clip_withoutLangevin_list,
                "loss_vgg_withLangevin_list": loss_vgg_withLangevin_list,
                "loss_clip_withLangevin_list": loss_clip_withLangevin_list,
                "SGLD_currentLatentVec_list": SGLD_currentLatentVec_list,
                "SGD_currentLatentVec_list": SGD_currentLatentVec_list,
                "loss_vgg_SGD_list": loss_vgg_SGD_list,
                "loss_clip_SGD_list": loss_clip_SGD_list,
            }

            # Only present when the Langevin phase ran (numReps_withLangevin > 0).
            if currentLatentVec_SGLD is not None:
                save_dict["latent_vec_SGLD"] = to_np(currentLatentVec_SGLD)
            if currentLatentVec_SGD is not None:
                save_dict["latent_vec_SGD"] = to_np(currentLatentVec_SGD)

            with open(save_file_name, "wb") as f:
                pickle.dump(save_dict, f)

            # save images
            save_name = f"{save_dir}/recon_img_normalized-{image_label}.jpg"
            recImg.save(save_name)

            if recImg_SGD is not None:
                sgd_save_dir = os.path.join(save_dir, "SGD")
                os.makedirs(sgd_save_dir, exist_ok=True)
                save_name = f"{sgd_save_dir}/recon_img_normalized-{image_label}.jpg"
                recImg_SGD.save(save_name)

            # Independent one-step draws from the same starting latent, to show how
            # far a single SGLD step moves the reconstruction.
            for kk in range(N_SAMPLING_DRAWS):
                generator_ = reconf.Langevin(
                    initInput=currentLatentVec,
                    initInputType="latentVector",
                    numReps=1,
                    returnVec=True,
                    lr_gamma=0,
                    lr_a=0.1,
                    lr_b=lr_b + 1e-8,
                    T=T_langevin,
                )
                for recImg_draw, _, _, _, _ in generator_:
                    save_dir_ = f"{save_dir}/sampling_{kk:02}"
                    os.makedirs(save_dir_, exist_ok=True)
                    save_name = f"{save_dir_}/recon_img_normalized-{image_label}.jpg"
                    recImg_draw.save(save_name)


if __name__ == "__main__":
    # load argparse
    parser = argparse.ArgumentParser(
        description="select reconstruction methods, provided by koide-majima"
    )
    # Take the choices from the config rather than restating them: the previous list
    # named six conditions that recon_params does not define, and selecting any of
    # them failed with a KeyError.
    with open("./scripts/config/config_recon.yaml", "rb") as f:
        available_methods = sorted(yaml.safe_load(f)["recon_params"])
    parser.add_argument(
        "method",
        type=str,
        help="select the method to use",
        default="original_all",
        choices=available_methods,
    )
    args = parser.parse_args()

    reconMethod = args.method
    save_base_dir = f"{RESULT_ROOT}/{reconMethod}"
    os.makedirs(save_base_dir, exist_ok=True)
    main(reconMethod, save_base_dir)
