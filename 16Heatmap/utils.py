import numpy as np
import cv2
import torch
from torch import nn
import torchvision
from torchvision.models import resnet50
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login
import gc
import os
from torchvision import transforms
from vit import ViTAggregation

class ViT_Regression_model(nn.Module):
    def __init__(self, path2MLPmodel=r'..\15ViTAggregator\results\result_0_0_6\model_trained.pth', batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.Feature_Extraction = Feature_Extraction()
        self.ViTmodel = ViTAggregation(heads=2, dim_head=8, mlp_dim=128, dim=1024,
                         depth=1, aggr="gap",n_classes=18272,dropout=0.2,
                        )
        self.ViTmodel.load_state_dict(torch.load(path2MLPmodel, map_location=torch.device('cpu')))
        self.ViTmodel.eval()
        
    def forward(self, features):
        output = []
        x_sample = self.ViTmodel(features)

        return x_sample
    
#======================================================================================================
#Based on UNI(downloaded)
class Feature_Extraction(nn.Module):
    def __init__(self, model_name="hf-hub:MahmoodLab/uni", pretrained=True,batch_size=1):
        super().__init__()

        self.batch_size = batch_size
        
        # Load the UNI model
        self.model = timm.create_model(model_name, pretrained=pretrained, init_values=1e-5, dynamic_img_size=True, img_size=224)
        
        # Resolve data config to setup the transformations appropriately
        self.transform = create_transform(**resolve_data_config(self.model.pretrained_cfg, model=self.model))
        
        self.model.eval()  # Set the model to evaluation mode

    def forward(self, x):
        with torch.no_grad():
            print(x.shape)
            if isinstance(x, list):
                x = torch.stack(x)
    
            x = x.squeeze(0)
            features = []
                
            for idx_start in range(0, len(x), self.batch_size):
                idx_end = min(idx_start + self.batch_size, len(x))
                batch_features = self.model(x[idx_start:idx_end])
                features.append(batch_features)
    
            features = torch.cat(features, dim=0)
            features = features.view(features.size(0), -1)
            
        return features
    
#======================================================================================================

class MLP_regression(nn.Module):
    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout, bias_init):
        super(MLP_regression, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.Dropout(dropout)
        )
        self.layer1 = nn.Linear(n_hiddens, n_outputs)
        if bias_init is not None:
            self.layer1.bias = bias_init

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = torch.mean(x, dim=0)
        return x

#=====================================================================================================
def evaluate_tile(img_np, edge_mag_thrsh, edge_fraction_thrsh):

    select = 1

    tile_size = img_np.shape[0]

    img_gray=cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0)
    sobely = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1)

    sobelx1 = cv2.convertScaleAbs(sobelx)
    sobely1 = cv2.convertScaleAbs(sobely)

    mag = cv2.addWeighted(sobelx1, 0.5, sobely1, 0.5, 0)

    unique, counts = np.unique(mag, return_counts=True)

    edge_mag = counts[np.argwhere(unique < edge_mag_thrsh)].sum()/(tile_size*tile_size)

    if edge_mag > edge_fraction_thrsh:
        select = 0
    
    return select
        
##================================================================================================
def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
