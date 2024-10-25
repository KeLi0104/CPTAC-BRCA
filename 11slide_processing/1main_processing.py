import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time,platform
from PIL import Image

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

from utils_preprocessing import *
import utils_color_norm
import os


OPENSLIDE_PATH = r'C:\Users\cocol\anaconda3\Lib\site-packages\openslide\openslide-bin-4.0.0.3-windows-x64\bin'  # Path to Openslide package

if hasattr(os, 'add_dll_directory'):
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

#%%
color_norm = utils_color_norm.macenko_normalizer()

init_random_seed(random_seed=42)
##======================================================================================================

path2storage = '../'#sys.argv[1]
project = 'CPTAC-BRCA' #sys.argv[2]
i_slide = int(16)#sys.argv[3])

print(f"project: {project}, i_slide: {i_slide}")
##-----------------Preprocessing Configuration-------------------
mag_assumed = 40       #magnification level(used as a fallback when the slide's actual magnification level is not available in the slide metadata)
evaluate_edge = True   #whether the tiles extracted from the slides should be evaluated for edge content or color variance
evaluate_color = False 
save_tile_file = False #whether to save the individual tiles extracted from the slides
extract_pretrained_features = True

mag_selected = 20      #desired magnification level 
tile_size = 512
#The Determining Downsample Factor can be calculated by downsampling_factor = mag_assumed / mag_selected

mask_downsampling = 16
#This mask image is a lower-resolution version of the original slide.If the original slide is very large (common with WSIs), 
#analyzing it at full resolution for certain tasks (like identifying regions of interest) can be computationally expensive and unnecessary. 
mask_tile_size = int(np.ceil(tile_size/mask_downsampling)) #=32
#print("mask_tile_size:", mask_tile_size)

##---------------------------------------
# Paths for saving processed data
path2slide = path2storage + f"{project}_slides_data/"
print("path2slide:", path2slide)
path2meta = "../10metadata/"
    
path2mask = f"{project}_mask/"
path2features = f"{project}_features/"

#metadata = pd.read_csv(f"../10match_slide_rna/{project}_slide_matched.csv")
metadata = pd.read_csv(f"{path2meta}{project}_slide_matched.csv")#gene_slides match file

## evaluate tile
edge_mag_thrsh = 15       #Edge Magnitude Threshold. 
#It's a measure of how pronounced or clear the edges within the tile must be for the tile to be considered informative or relevant for further analysis.
#During tile evaluation, an edge detection algorithm (such as the Sobel operator) might be applied to each tile to identify edges within the tissue. 
#The edge_mag_thrsh value sets a minimum threshold for the edge magnitudes that are counted towards the tile being considered as having significant edge content. 
#Tiles with edge magnitudes below this threshold might be considered too smooth or lacking in detail.(e.g., large expanses of whitespace or very uniform regions).

edge_fraction_thrsh = 0.5 #Edge Fraction Threshold(proportion)
#The minimum fraction of a tile that must meet the edge magnitude threshold for the tile to be selected for further analysis. 

##--------------------------
##Ensure the necessary directories for saving mask images and extracted features exist, if not, create one
if not os.path.exists(path2mask):
    os.makedirs(path2mask)

if extract_pretrained_features:
    if not os.path.exists(path2features):
        os.makedirs(path2features)

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
## slide size at largest level (level=0)
px0, py0 = slide.level_dimensions[0]
tile_size0 = int(tile_size*downsampling) # size of the tiles at the original magnificationv,scaled by the downsampling factor(eg.1024)
print(f"px0: {px0}, py0: {py0}, tile_size0: {tile_size0}")

n_rows,n_cols = int(py0/tile_size0), int(px0/tile_size0) #dividing the original image to a tile matrix that has n_rows rows and n_cols columns.
print(f"n_rows: {n_rows}, n_cols: {n_cols}")

n_tiles_total = n_rows*n_cols    #total number of tiles
print(f"n_tiles_total: {n_tiles_total}")

##====================================================================================================== 
#initialize mask image(RGB) by np.full to create arrays filled with the value 255 (white in RGB), (255,255,255)for each pixel
# the size of mask tiles is 'mask_downsampling' times smaller than 512, but the number of tiles is the same
img_mask = np.full((int((n_rows)*mask_tile_size),int((n_cols)*mask_tile_size),3),255).astype(np.uint8) 
mask = np.full((int((n_rows)*mask_tile_size),int((n_cols)*mask_tile_size),3),255).astype(np.uint8)

i_tile = 0
tiles_list = [] #initial the tiles list of the slide
for row in range(n_rows): #iterate over the grid of potential tiles
    print(f"row: {row}/{n_rows}")
    for col in range(n_cols):
        
        tile = slide.read_region((col*tile_size0, row*tile_size0), 
                                 level=0, size=[tile_size0, tile_size0]) ## RGBA image(Red, Green, Blue, and Alpha, where Alpha stands for transparency) #(col*tile_size0, row*tile_size0) is the current position of the traverse
        tile = tile.convert("RGB")
        
        if tile.size[0] == tile_size0 and tile.size[1] == tile_size0: #make sure the size is correct
            # downsample to target tile size
            tile = tile.resize((tile_size, tile_size))
            mask_tile = np.array(tile.resize((mask_tile_size, mask_tile_size)))
            
            img_mask[int(row*mask_tile_size):int((row+1)*mask_tile_size),\
                     int(col*mask_tile_size):int((col+1)*mask_tile_size),:] = mask_tile #define the corresponding part of the mask image

            tile = np.array(tile)
            #print(tile.shape)

            ## evaluate tile (whether the tile is selected for further analysis)
            # the definition of func 'evaluate_tile' is in util_preprocessing.py 
            select = evaluate_tile(tile, edge_mag_thrsh, edge_fraction_thrsh) # bool

            if select == 1:
                ## 2022.09.08: color normalization:
                tile_norm = Image.fromarray(color_norm.transform(tile))
                #Color normalization: correct for variations in staining across different slides or even within the same slide.

                #resize the normalized tile to a mask tile
                mask_tile_norm = np.array(tile_norm.resize((mask_tile_size, mask_tile_size)))
                mask[int(row*mask_tile_size):int((row+1)*mask_tile_size),\
                     int(col*mask_tile_size):int((col+1)*mask_tile_size),:] = mask_tile_norm   #define the part of the mask image to the normalized mask tile 


                #tiles_list.append(np.array(tile_norm).astype(np.uint8))
                tiles_list.append(tile_norm) #add the processed tile to tiles_list

                if save_tile_file: 
                    tile_name = "tile_" + str(row).zfill(5)+"_" + str(col).zfill(5) + "_" \
                             + str(i_tile).zfill(5) + "_" + str(downsampling).zfill(3)

                    tile_norm.save(f"{tile_folder}/{tile_name}.png")

        i_tile += 1

##====================================================================================================== 
## plot: Drawing Grid Lines on Masks
## visualizing and saving the mask images
line_color = [0,255,0] #green

n_tiles = len(tiles_list)

img_mask[:,::mask_tile_size,:] = line_color
img_mask[::mask_tile_size,:,:] = line_color
mask[:,::mask_tile_size,:] = line_color
mask[::mask_tile_size,:,:] = line_color

fig, ax = plt.subplots(1,2,figsize=(40,20))
ax[0].imshow(img_mask)
ax[1].imshow(mask)

ax[0].set_title(f"{slide_name}, mag_original: {mag_original}, mag_assumed: {mag_assumed}")
ax[1].set_title(f"n_rows: {n_rows}, n_cols: {n_cols}, n_tiles_total: {n_tiles_total}, n_tiles_selected: {n_tiles}")

plt.tight_layout(h_pad=0.4, w_pad=0.5)
plt.savefig(f"{path2meta}{path2mask}{slide_name}.pdf", format="pdf", dpi=50)
plt.close()

img_mask = 0 ; mask = 0

print("completed cleaning")

##======================================================================================================
model = Feature_Extraction()
batch_size = 64
##----------
## resize:
torch_resize = transforms.Resize(224) 

tiles_list_resized = []
for i in range(n_tiles):
    tiles_list_resized.append(torch_resize(tiles_list[i]))
tiles_list = tiles_list_resized #resized tiles list (224*224)

##----------
## normalize by ImageNet mean and std:
data_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])
                                    ])#the UNI model uses the same transform, check get_encoder.py/get_eval_transforms()

##----------------------------------------
def extract_features_from_tiles(tiles_list):

    ## transform to torch and normalize
    tiles = []
    for i in range(n_tiles):
        tiles.append(data_transform(tiles_list[i]).unsqueeze(0))
    tiles = torch.cat(tiles, dim=0)
    print("tiles.shape:", tiles.shape)   ## [n_tiles, 3, 224, 224]
    #tiles_list = 0

    ##------------------------------------
    ## extract feature from tile image
    features = []
    for idx_start in range(0, n_tiles, batch_size):
        idx_end = idx_start + min(batch_size, n_tiles - idx_start) #last batch (if n_tiles is not divisible by batch_size)

        feature = model(tiles[idx_start:idx_end])
        
        features.append(feature.detach().cpu().numpy())

    features = np.concatenate(features)

    return features


##----------------------------------------
if extract_pretrained_features:
    features = extract_features_from_tiles(tiles_list)
    print("features.shape:", features.shape)
    np.save(f"{path2features}{slide_name}.npy", features)

##======================================================================================================
print(f"finished -- i_slide: {i_slide}, total time: {int(time.time() - start_time)}")


