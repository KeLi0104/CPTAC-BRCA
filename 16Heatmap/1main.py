import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time,platform
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import resnet50

from utils import *
import utils_color_norm

from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients

OPENSLIDE_PATH = r'C:\Users\cocol\anaconda3\Lib\site-packages\openslide\openslide-bin-4.0.0.3-windows-x64\bin'  # 替换为你解压OpenSlide文件的实际路径

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

#%%
color_norm = utils_color_norm.macenko_normalizer()
init_random_seed(random_seed=42)

path2storage = '../'#sys.argv[1]
project = 'CPTAC-BRCA' #sys.argv[2]
i_slide = int(23)#sys.argv[3])

print(f"project: {project}, i_slide: {i_slide}")
##-----------------Preprocessing Configuration-------------------
mag_assumed = 40      
evaluate_edge = True   
evaluate_color = False 
save_tile_file = True 
extract_pretrained_features = True

mag_selected = 20      
tile_size = 512

##-----------------------------------------------------------------
# Paths for saving processed data
path2slide = path2storage + f"{project}_slides_data/"
print("path2slide:", path2slide)
path2meta = "../10metadata/"
    
path2mask = f"{project}_mask/"
path2features = f"{project}_features/"
path2target = "../10metadata/"

metadata = pd.read_csv(f"{path2meta}{project}_slide_matched.csv")#gene_slides match file

## evaluate tile
edge_mag_thrsh = 15  
edge_fraction_thrsh = 0.5

##======================================================================================================
slide_file_names = metadata.slide_file_name.values
slide_names = metadata.slide_name.values

start_time = time.time()

slide_file_name = slide_file_names[i_slide]
slide_name = slide_names[i_slide]
print(f"slide_file_name: {slide_file_name}, slide_name: {slide_name}")

if save_tile_file:    #whether to save the tile file
    ## create tile_folder:
    tile_folder = f"{project}_tiles/" + slide_name
    print(f"tile_folder: {tile_folder}")

    if not os.path.exists(tile_folder):
        os.makedirs(tile_folder)

##======================================================================================================
gene_file = f"{path2target}{project}_genes.npy"
genes = np.load(gene_file, allow_pickle=True)
genes = pd.DataFrame(data=genes[1:], columns=genes[0]).iloc[i_slide][6:]
targets = list(pd.read_csv("../13Classification/analysis_results/coef_sorted.csv")["gene"].iloc[0:1])
targets = genes.index.get_loc(str(targets[0]))
##====================================================================================================== 
slide = openslide.OpenSlide(f"../CPTAC-BRAC_slides_data/{slide_file_name}")
print('Slide:',f"../CPTAC-BRAC_slides_data/{slide_file_name}")

## magnification max
#determine whether the magnification level is available in the slide metadata, if not, use 'mag_assumed' as default one.
if openslide.PROPERTY_NAME_OBJECTIVE_POWER in slide.properties: 
    mag_max = slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER]
    print("mag_max:", mag_max)
    mag_original = mag_max
else:
    print("[WARNING] mag not found, assuming: {mag_assumed}")
    mag_max = mag_assumed
    mag_original = 0

## downsample_level
downsampling = int(int(mag_max)/mag_selected) #calculate the downsampling factor
print(f"downsampling: {downsampling}")

##======================================================================================================
px0, py0 = slide.level_dimensions[0]
tile_size0 = int(tile_size*downsampling)
print(f"px0: {px0}, py0: {py0}, tile_size0: {tile_size0}")

n_rows,n_cols = int(py0/tile_size0), int(px0/tile_size0) 
print(f"n_rows: {n_rows}, n_cols: {n_cols}")

n_tiles_total = n_rows*n_cols   
print(f"n_tiles_total: {n_tiles_total}")

##====================================================================================================== 
i_tile = 0
tiles_list = [] 
tiles_postion = []

downsampling = 16
downsampled_tile_size = int(np.ceil(tile_size/downsampling)) #=32
downsampled_slide = np.zeros((n_rows*downsampled_tile_size,n_cols*downsampled_tile_size,3),dtype = float)

for row in range(n_rows): 
    print(f"row: {row}/{n_rows}")
    for col in range(n_cols):
        
        tile = slide.read_region((col*tile_size0, row*tile_size0), 
                                 level=0, size=[tile_size0, tile_size0]) 
        tile = tile.convert("RGB")
        
        downsampled_tile = np.array(tile.resize((downsampled_tile_size, downsampled_tile_size)))
        
        downsampled_slide[row*downsampled_tile_size:(row+1)*downsampled_tile_size,col*downsampled_tile_size:(col+1)*downsampled_tile_size,:] = downsampled_tile 
        
        if tile.size[0] == tile_size0 and tile.size[1] == tile_size0:

            tile = tile.resize((tile_size, tile_size))
            tile = np.array(tile)
            select = evaluate_tile(tile, edge_mag_thrsh, edge_fraction_thrsh)

            if select == 1:
                tile_norm = Image.fromarray(color_norm.transform(tile))
                tiles_list.append(tile_norm)
                tiles_postion.append([row,col])
                if save_tile_file: 
                    tile_name = "tile_" + str(row).zfill(5)+"_" + str(col).zfill(5) + "_" \
                             + str(i_tile).zfill(5) + "_" + str(downsampling).zfill(3)

                    tile_norm.save(f"{tile_folder}/{tile_name}.png")

        i_tile += 1
##====================================================================================================== 
batch_size = 1
n_tiles = len(tiles_list)
#Resize
torch_resize = transforms.Resize(224) 
tiles_list_resized = []
for i in range(n_tiles):
    tiles_list_resized.append(torch_resize(tiles_list[i]))
tiles_list = tiles_list_resized #resized tiles list (224*224)

#Normalization by ImageNet mean and std:
data_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])
                                    ])

normed_tiles = []
for i in range(n_tiles):
    normed_tiles.append(data_transform(tiles_list[i]).unsqueeze(0))
normed_tiles = torch.stack(normed_tiles)
normed_tiles = normed_tiles.unsqueeze(0)
normed_tiles = normed_tiles.squeeze(2)

#feature extraction
feature_extarctor = Feature_Extraction()
features = feature_extarctor(normed_tiles)

model = ViT_Regression_model(path2MLPmodel = r'..\15ViTAggregator\results\result_0_0_6\model_trained.pth',batch_size = batch_size)

#%%
ig = IntegratedGradients(model)

attributions, delta = ig.attribute(features, target=targets, return_convergence_delta=True,internal_batch_size = 1)

attributions = attributions.squeeze().cpu().detach().numpy()

tile_attributions = np.mean(abs(attributions),axis=1)




attribution_plot = np.zeros((n_rows*downsampled_tile_size,n_cols*downsampled_tile_size),dtype = float)
for i in range(len(tiles_postion)):
    row,col = tiles_postion[i]
    attribution_plot[row*downsampled_tile_size:row*downsampled_tile_size + downsampled_tile_size, col*downsampled_tile_size:col*downsampled_tile_size+downsampled_tile_size] = tile_attributions[i]*np.ones((downsampled_tile_size,downsampled_tile_size),dtype = float)
max_attribution = attribution_plot.max()
attribution_plot = attribution_plot/max_attribution

#plot
nx,ny = 2,1
fig, ax = plt.subplots(ny,nx)

ax[0].imshow(np.uint8(downsampled_slide))

ax[1].imshow(attribution_plot, cmap='hot')
ax[1].set_title('Heatmap')
ax[1].set_xlabel('X-axis')
ax[1].set_ylabel('Y-axis')

plt.tight_layout(h_pad=0.4, w_pad=0.5)
plt.savefig(f'heatmap_slide{i_slide}.pdf', format="pdf", dpi=300)
plt.show()

