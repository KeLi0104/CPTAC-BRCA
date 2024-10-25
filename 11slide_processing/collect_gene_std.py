import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time

path2storage = '../'
project = 'CPTAC-BRCA' 

path2meta = "../10metadata/"
    
path2mask = f"{project}_mask/"
path2features = f"{project}_features/"
path2target = "../10metadata/"

##======================================================================================================
gene_file = f"{path2target}{project}_genes.npy"
genes = np.load(gene_file, allow_pickle=True)
genes = pd.DataFrame(data=genes[1:], columns=genes[0]).iloc[:,6:]
genes = np.array(genes)

genes_list = pd.read_csv(f"../10metadata/{project}_genes.csv")
genes_list = genes_list["gene"].values
genes_list = np.array(genes_list).reshape(1,-1)

#%%
genes_std = []
for i in range(genes.shape[1]):
    genes_std.append(np.std(genes[:,i]))
genes_std = np.array(genes_std).reshape(1,-1)
genes_std = np.vstack([genes_list,genes_std])
genes_std = pd.DataFrame(data={"gene": genes_std[0], "std": genes_std[1]})

genes_std = genes_std.sort_values(by="std", ascending=False)                   
genes_std.to_csv(f"../10metadata/{project}_genes_std.csv", index=None)

