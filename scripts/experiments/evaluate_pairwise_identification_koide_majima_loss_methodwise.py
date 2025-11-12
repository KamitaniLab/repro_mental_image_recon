# %%

import sys
import argparse
import yaml
import pickle
from PIL import Image   
import numpy as np
import torch
import pickle
import scipy
from itertools import product
import os

sys.path.append("./mental_img_recon")
from recon_utils  import get_target_image, convert_featname
import recon_func
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Converting VQGAN output into VGG input format
from torchvision import transforms
def convertImageIntoVGGinput(Image):
    #convert image [0-255, uint8] into [0-1, float32]
    VGGinput = np.array(Image)/255.0
    VGGinput = torch.tensor(VGGinput.astype(np.float32).transpose(2, 0, 1)[np.newaxis])

    if VGGinput.shape[2] == 224 and VGGinput.shape[3] == 224:
        preprocessBeforeVGG = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        preprocessBeforeVGG = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    inputForVGG = preprocessBeforeVGG(VGGinput)

    return inputForVGG


def convertImageIntoCLIPinput(Image):
    #convert image [0-255, uint8] into [0-1, float32]
    CLIPinput = np.array(Image)/255.0
    CLIPinput = torch.tensor(CLIPinput.astype(np.float32).transpose(2, 0, 1)[np.newaxis])

    if CLIPinput.shape[2] == 224 and CLIPinput.shape[3] == 224:
        preprocessBeforeVGG = transforms.Compose([
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])])
    else:
        preprocessBeforeVGG = transforms.Compose([
            transforms.Resize((imageSize[0], imageSize[1])),
            transforms.Normalize(mean=[0.4814, 0.4578, 0.4082], std=[0.2686, 0.2613, 0.2757])])
    inputForCLIP = preprocessBeforeVGG(CLIPinput)

    return inputForCLIP

def compute_loss_CLIP(CLIPmodel, CLIPmodelWeight, input1, input1_type, input2, input2_type, meanCLIPfeature, cosSimilarity, similarity='corr', num_crop=32, DEVICE='cuda'):

    for index_CLIPmodel in range(len(CLIPmodel)):

        # Get x1, x2
        # for input1
        if input1_type == 'img':
            CLIPfeature1 = CLIPmodel[index_CLIPmodel].encode_image(
                recon_func.createCrops(input1,num_crop, DEVICE=DEVICE))
            x1 = CLIPfeature1.reshape(CLIPfeature1.shape[0], -1)
        elif input1_type == 'feat':
            x1 = input1[index_CLIPmodel]
        # for input2
        if input2_type == 'img':
            CLIPfeature2 = CLIPmodel[index_CLIPmodel].encode_image(input2.to(DEVICE))
            x2 = CLIPfeature2.reshape(CLIPfeature2.shape[0], -1)
        elif input2_type == 'feat':
            x2 = input2[index_CLIPmodel]

        # Subtract the mean feature vector
        x1 = x1-meanCLIPfeature[index_CLIPmodel].reshape(1, -1)
        x2 = x2-meanCLIPfeature[index_CLIPmodel].reshape(1, -1)

        # Compute similarity
        if similarity == 'corr':
            loss_thisLayer = -cosSimilarity(x1-x1.mean(dim=1, keepdim=True),
                                            x2-x2.mean(dim=1, keepdim=True)).mean()
        elif similarity == 'cosine':
            loss_thisLayer = -cosSimilarity(x1, x2).mean()
        elif similarity == 'MSE':
            loss_thisLayer = ((x1-x2)**2).mean()
        else:
            print('Error: Similarity metric should be corr, cosine, or MSE.')
            sys.exit(1)

        if index_CLIPmodel == 0:
            loss_CLIP = loss_thisLayer*CLIPmodelWeight[index_CLIPmodel]
        else:
            loss_CLIP = loss_thisLayer * \
                CLIPmodelWeight[index_CLIPmodel]+loss_CLIP

    return loss_CLIP

def calculate_similairty_koidemajima(reconImg, targetImg, meanVGGfeature_, meanCLIPfeature_, cosSimilarity, 
                                        VGGmodel_, VGGlayerWeight_, used_layers_VGG__in, CLIPmodel_, CLIPmodelWeight_, CLIPcoef_, DEVICE, sim_metric='corr'):
    target_VGG_input = convertImageIntoVGGinput(targetImg).to(DEVICE)
    recon_VGG_input = convertImageIntoVGGinput(reconImg).to(DEVICE)

    target_CLIP_input = convertImageIntoVGGinput(targetImg).to(DEVICE)
    recon_CLIP_input = convertImageIntoVGGinput(reconImg).to(DEVICE)    

    if len(used_layers_VGG__in) != 0:
        loss_VGG = recon_func.compute_loss_VGG(VGGmodel_, used_layers_VGG__in, VGGlayerWeight_ , recon_VGG_input, 'img',
                        target_VGG_input, 'img', meanVGGfeature_, cosSimilarity, similarity=sim_metric)
    else:
        loss_VGG = torch.tensor(0.0).to(DEVICE)
    # Added num_crop argument to compute empirical mean
    # (cf Koide-Majima et al., 2024):
    # the mean in Eq. (2) was replaced by the empirical mean with 320 random samples.
    loss_CLIP = compute_loss_CLIP(CLIPmodel_, CLIPmodelWeight_, recon_CLIP_input, 'img',
                                          target_CLIP_input, 'img', meanCLIPfeature_, cosSimilarity,  similarity=sim_metric,
                                          num_crop=320, DEVICE=DEVICE)
    
    loss = (loss_VGG + loss_CLIP * CLIPcoef_).cpu().detach().numpy()
    similarity = - loss
    
    return similarity
# %%


def perform_evaluation(result_dir, reconMethod='original'):
    with open('./scripts/config/demo_params.yaml', 'rb') as f: 
        prm_demo = yaml.safe_load(f)
    with open('./scripts/config/config_KS_mod.yaml', 'rb') as f: 
        dt_cfg = yaml.safe_load(f) 
        
        
    # %%

    #reconMethod = 'original_withoutclip'
    #reconMethod = "Langevin"
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
    subject_list = ['S1', 'S2', 'S3']
    # select from 0 to 24
    # Here are examples:
    # ID 21: 'Bowling ball (artifact)'
    # ID 20: 'Airplane (artifact)'
    # ID 18: 'Leopard (animal)'
    # ID 19: 'Goat (animal)'
    # ID  7: 'Blue + (symbol)'
    # ID 14: 'Black x (symbol)'
    targetID_list = np.arange(25)
    #image_label_list.remove('Img0016')

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

    if reconMethod == 'Langevin' or reconMethod.startswith('original'):
        lr_gamma = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_gamma']
        lr_a = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_a']
        lr_b = dt_cfg['recon_params'][reconMethod]['Langevin']['lr_b']
        T_langevin = dt_cfg['recon_params'][reconMethod]['Langevin']['T']
        
    # set parameters
    numReps_withoutLangevin = dt_cfg['recon_params'][reconMethod]["numReps_withoutLangevin"]#1000 # (default) 1000
    numReps_Langevin = dt_cfg['recon_params'][reconMethod]["numReps_withLangevin"]#500 # (default) 500
    # %%
    ### Main: Reconstruction -------------------------------------------------
    VGGlayerWeight_ = np.ones(len(used_layers_VGG))
    VGGlayerWeight_ = VGGlayerWeight_/VGGlayerWeight_.sum()

    meanVGGfeature_ = list()
    for li in range(len(used_layers_VGG)):
        x = scipy.io.loadmat(os.path.join(meanFeatureDir, 'VGG19', used_layers_VGG[li], 'meanFeature_.mat'))
        meanVGGfeature_.append(torch.tensor(x['mu'], dtype=torch.float32).to(DEVICE))
        del x
        
    meanCLIPfeature_ = list()
    for mi in range(len(CLIPmodel_)):
        x = scipy.io.loadmat(os.path.join(meanFeatureDir, CLIP_modelNames[mi], CLIP_usedLayer, 'meanFeature_.mat'))
        meanCLIPfeature_.append(torch.tensor(x['mu'], dtype=torch.float32).to(DEVICE))
        del x
        
    cosSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    # %%

    save_base_dir =f'{result_dir}/{reconMethod}' #f'./results/recon_image_koide-majima_v2/{reconMethod}'
    eval_id_list = targetID_list[-10:] # nautal image only 
    #eval_id_list = targetID_list
    # %%
    #for j, subject in enumerate(subject_list):
    #subject = subject_list[1]
    pw_iden_results = {}
    pw_iden_matrix = {}
    for subject in subject_list:
        print(subject)    
        target_image_pil_list = []
        recon_image_pil_list = []

        for i, targetID in enumerate(eval_id_list):
            ii = targetID
            save_dir = f'{save_base_dir}/{subject}/VC'
            #os.makedirs(save_dir, exist_ok=True)
            
            targetImg_, targetimname = get_target_image(targetID, targetimpath)
            recon_name = f"Stim{ii+1:02}_{targetimname}"
            if targetID >14:
                tid = targetID +1
            else:
                tid = targetID
                    
            
            targetImg = Image.fromarray(targetImg_)
            #recon_image_path = f'{save_dir}/{recon_name}.tiff'
            image_label = 'Img{:04d}'.format(tid+1)
            recon_image_path = f'{save_dir}/recon_img_normalized-{image_label}.jpg'
            reconImg = Image.open(recon_image_path)
            
            target_image_pil_list.append(targetImg)
            recon_image_pil_list.append(reconImg)

        # create 0 matrix for similarity whose shape is len(targetID_list) x len(targetID_list)
        # %%
        sim_matrix = np.zeros((len(eval_id_list), len(eval_id_list)))

        for i, t_img in enumerate(target_image_pil_list):
            for j, r_img in enumerate(recon_image_pil_list):
                similarity = calculate_similairty_koidemajima(r_img, t_img, meanVGGfeature_, meanCLIPfeature_, cosSimilarity, 
                                                    VGGmodel_, VGGlayerWeight_, used_layers_VGG__in, 
                                                    CLIPmodel_, CLIPmodelWeight_, CLIPcoef_,
                                                    DEVICE)
            
                # save similarity to sim_matrix
                sim_matrix[i, j] = similarity
        # %%
        d = sim_matrix
        cr = np.sum(d - np.diag(d)[:, np.newaxis] < 0, axis=1) / (d.shape[1] -1)

        # %%
        pw_iden_results[subject] = cr
        pw_iden_matrix[subject] = sim_matrix
        print(np.mean(cr))
        
    save_name = 'pairwise_identification_results_koide_majima_optsame_metric.pkl'
    with open(os.path.join(save_base_dir, save_name), 'wb') as f:
        pickle.dump(pw_iden_results, f)
    # sim_matrixの保存
    save_name = 'pairwise_identification_results_koide_majima_optsame_metric_sim_matrix.pkl'
    with open(os.path.join(save_base_dir, save_name), 'wb') as f:
        pickle.dump(pw_iden_matrix, f)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Reconstruct imagined image with DGN')
    parser.add_argument('--method', type=str, help='select the method to use', default="original_all",
                        choices=['original_all', "CLIPonly_all",
                                 ])
    parser.add_argument('--results_dir', type=str, default="results/rep_recon_image_koide-majima")
    
   
    
    args = parser.parse_args()

    result_dir = args.results_dir
    
    perform_evaluation(result_dir, reconMethod=args.method)
    print('done!')