# %%

import sys
import argparse
import yaml
import pickle
from PIL import Image   
import numpy as np
import torch
import torch.nn as nn
import pickle
from itertools import product
import os

from bdpy.dl.torch.domain import image_domain
from bdpy.recon.torch.modules.encoder import SimpleEncoder
from bdpy.recon.torch.modules.critic import LayerWiseAverageCritic

# %%
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
# %%

class CLIPEncoder(SimpleEncoder):
    def __init__(self, feature_network: nn.Module, layer_names, domain):
        
        if "output" in layer_names:
            # exclude output layer from layer_names
            self._include_output = True
            layer_names = [name for name in layer_names if name != "output"]
        else:
            self._include_output = False
        super().__init__(feature_network.visual, layer_names, domain)
        
        self._feature_network = feature_network 
    
    def encode(self, images: torch.Tensor):
        """Encode images.

        Parameters
        ----------
        images : torch.Tensor
            Images to encode.

        Returns
        -------
        Dict[str, torch.Tensor]
            Encoded features.
        """
        images = self._domain.receive(images)
        res = self._feature_extractor(images)
        if self._include_output:
            res["output"] = self._feature_network.encode_image(images)
        return res

# %%
class CorrelationLoss(LayerWiseAverageCritic):
        """
        Correlation between feature and target feature
        """
        
        def compare_layer(
            self, feature: torch.Tensor, target_feature: torch.Tensor, layer_name: str
            ) -> torch.Tensor:
           """Loss function per layer.

           Parameters
           ----------
           feature : torch.Tensor
                Feature tensor of the layer specified by `layer_name`.
           target_feature : torch.Tensor
                Target feature tensor of the layer specified by `layer_name`.
           layer_name : str
                Layer name.

           Returns
           -------
           torch.Tensor
                Loss value of the layer specified by `layer_name`.
           """
           cosSimilarity = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
           feature = feature.reshape(feature.shape[0], -1)
           target_feature = target_feature.reshape(target_feature.shape[0], -1)
           return -cosSimilarity(feature-feature.mean(dim=1, keepdim=True),
                                            target_feature-target_feature.mean(dim=1, keepdim=True)
                                            ).mean()

def perform_evaluation(result_dir, reconMethod='original', loss_name='correlation'):
    
    with open('./scripts/config/demo_params.yaml', 'rb') as f: 
        prm_demo = yaml.safe_load(f)
    with open('./scripts/config/config_KS_mod.yaml', 'rb') as f: 
        dt_cfg = yaml.safe_load(f) 
        
    targetimpath = prm_demo['dt_targetimages_path']
        
    # %%

    #reconMethod = 'original_withoutclip'
    # Device
    cudaID = "cuda:0"
    device = torch.device(cudaID if torch.cuda.is_available() else "cpu")
    dtype= torch.float32
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

    # %%

    save_base_dir =f'{result_dir}/{reconMethod}' #f'./results/recon_image_koide-majima_v2/{reconMethod}'
    eval_id_list = targetID_list[-10:] # targetID_list[:-10] , #targetID_list
    #eval_id_list = targetID_list
    # %%
    
    #encoder = CLIPEncoder(model,["output"], domain=icnn_preprocess)
    pil2common = image_domain.PILDomainWithExplicitCrop()
    
    save_dict = {}
    
    
    # %%
    
    #for j, subject in enumerate(subject_list):
    #subject = subject_list[1]
    pw_iden_results = {}
    pw_iden_matrix = {}
    
    
    for subject in subject_list:
        print(subject)    
        pw_iden_results[subject] = {}
        pw_iden_matrix[subject] = {}
        

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
        
        sim_matrix = np.zeros((len(eval_id_list), len(eval_id_list)))
        for i, r_img in enumerate(recon_image_pil_list):
            recon_image = r_img.resize([224,224])
            #recon_stim_domain = preprocess(recon_image).to(device)
            
            recon_image_array = np.asarray(recon_image).astype(np.float32)[np.newaxis]
            
            #recon_image_tensor = torch.tensor(np.asarray(recon_image).astype(np.float32)[np.newaxis]).to(device)
            #recon_stim_domain = pil2common.send(recon_image_tensor)
            for j, t_img in enumerate(target_image_pil_list):
                cand_image = t_img.resize([224,224])
                cand_image_array = np.asarray(cand_image).astype(np.float32)[np.newaxis]
                #cand_image_tensor = torch.tensor(np.asarray(cand_image).astype(np.float32)[np.newaxis]).to(device)
                
                #cand_stim_domain = pil2common.send(cand_image_tensor)
                
                similarity= np.corrcoef(recon_image_array.flatten(), cand_image_array.flatten())[0,1]
                # save similarity to sim_matrix
                sim_matrix[i, j] = similarity
       
        d = sim_matrix
        cr = np.sum(d - np.diag(d)[:, np.newaxis] < 0, axis=1) / (d.shape[1] -1)

        
        pw_iden_results[subject]["output"] = cr
        pw_iden_matrix[subject]["output"] = sim_matrix
        print(np.mean(cr))
    # %%   
    save_name = f'pairwise_identification_results_pixel_{loss_name}.pkl'
    with open(os.path.join(save_base_dir, save_name), 'wb') as f:
        pickle.dump(pw_iden_results, f)
    # sim_matrixの保存
    save_name = f'pairwise_identification_results_pixel_{loss_name}_sim_matrix.pkl'
    with open(os.path.join(save_base_dir, save_name), 'wb') as f:
        pickle.dump(pw_iden_matrix, f)

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Reconstruct imagined image with DGN')
    parser.add_argument('--method', type=str, help='select the method to use', default="CLIPonly_all",
                        choices=['original_all', "CLIPonly_all",
                                 'original_middle', "wo_SGLD_CLIP_all",
                                 "wo_SGLD_CLIP_middle", 
                                 'original_conv1', "wo_SGLD_CLIP_conv1", 
                                 'original_conv2', "wo_SGLD_CLIP_conv2",
                                 'original_conv3', "wo_SGLD_CLIP_conv3",
                                 'original_conv4', "wo_SGLD_CLIP_conv4",
                                 'original_conv5', "wo_SGLD_CLIP_conv5",
                                 'original_fc6', "wo_SGLD_CLIP_fc6",
                                 'original_fc7', "wo_SGLD_CLIP_fc7",
                                 'original_fc8', "wo_SGLD_CLIP_fc8",
                                 "AdamOnly_middle", "SGLDOnly_middle",
                                 "VGGonly_middle", "8_reconstruct_imagined_image_withDGN_19_layer_conv2_fc6"
                                 ])
    
    #parser.add_argument('--eval_model', type=str, default="alexnet")
    parser.add_argument('--loss_name', type=str, default="correlation",
                        choices = ['correlation', 'MSE'])
    
   
    parser.add_argument('--results_dir', type=str, default="results/rep_recon_image_koide-majima")
    
    
    args = parser.parse_args()

    result_dir = args.results_dir
    
    perform_evaluation(result_dir, reconMethod=args.method, loss_name=args.loss_name)
    print('done!')