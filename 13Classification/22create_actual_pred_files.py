import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time
##===========================================================================
project = "CPTAC-BRCA"
path2meta = "../10metadata/"

rna_type = "beta"

genes_file = f"{project}_genes.csv"

df_meta = pd.read_csv(f"{path2meta}{project}_slide_matched.csv")

genes = pd.read_csv(f"{path2meta}{genes_file}")
genes = genes["gene"].values
n_genes = len(genes)
print("n_genes:", n_genes)

patients_split_file = f"{path2meta}{project}_train_valid_test_idx.npz"

train_valid_test_idx = np.load(patients_split_file, allow_pickle=True)

##=============================================================================
il_fold = 0
ik_folds = np.arange(5)

for ik_fold in ik_folds:

    test_idx = train_valid_test_idx["test_idx"][ik_fold]
    
    if ik_fold == 0:
        test_idx_all = test_idx
    else:
        test_idx_all = np.hstack((test_idx_all,test_idx))

df_meta_test = df_meta.loc[test_idx_all].reset_index(drop=True)

actual = np.load(f"analysis_results/test_actual_all.npy")
pred = np.load(f"analysis_results/test_pred_all.npy")
print(actual.shape, pred.shape)

df_actual = pd.DataFrame(actual, columns=genes)
df_actual = pd.concat([df_meta_test,df_actual],axis=1)

df_pred = pd.DataFrame(pred, columns=genes)
df_pred = pd.concat([df_meta_test,df_pred],axis=1)

df_actual.to_pickle(f"analysis_results/{project}_{rna_type}_test_actual.pkl")
df_pred.to_pickle(f"analysis_results/{project}_{rna_type}_test_pred.pkl")

print(" --- completed creating files--- ")
