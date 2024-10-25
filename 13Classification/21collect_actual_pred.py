import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time

##===========================================================================
print(" ")
print("--- Collecting labels and preds ---")

project = "CPTAC-BRCA"
rna_type = "beta"

result_folder = "analysis_results/"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

##===========================================================================
##
genes = pd.read_csv(f"../10metadata/{project}_genes.csv")
genes = genes["gene"].values

n_genes = len(genes)
print("n_genes:", n_genes)

##===========================================================================
n_genes_each = 4000
n_models = int(n_genes/n_genes_each) + 1
print("n_models:", n_models)

i_steps = np.array([i*n_genes_each for i in range(n_models)])
print("i_steps.shape:", i_steps.shape)
print(i_steps)

##===========================================================================

ik_folds = [0,1,2,3,4]
il_folds = [0,1,2,3,4]

for ik,ik_fold in enumerate(ik_folds):
    print(" ")
    print("ik_fold:", ik_fold)
    
    ##=====================================
    ## labels
    labels = []
    for i in i_steps:
        label = np.loadtxt(f"../12Indirect_Regression/results/result_{ik_fold}_{il_folds[0]}_6/test_labels.txt")
        labels.append(label)

    labels = np.concatenate(labels,axis=1)
    
    ##=====================================
    ## preds
    preds_mean = np.zeros((len(il_folds), labels.shape[0], labels.shape[1]))
    for il_fold in il_folds:
        preds = []
        for i in i_steps:
            pred = np.loadtxt(f"../12Indirect_Regression/results/result_{ik_fold}_{il_fold}_6/test_preds.txt")
            preds.append(pred)
        
        preds = np.concatenate(preds,axis=1)

        preds_mean[il_fold,:,:] = preds
    
    preds_mean = np.mean(preds_mean,axis=0)

    if ik == 0:
        labels_all = labels
        preds_all = preds_mean

    else:
        labels_all = np.vstack((labels_all, labels))
        preds_all = np.vstack((preds_all, preds_mean))

    ##=====================================

print(f"labels_all.shape: {labels_all.shape}, preds_all.shape: {preds_all.shape}")

##===========================================================================
np.save(f"{result_folder}test_actual_all.npy", labels_all)
np.save(f"{result_folder}test_pred_all.npy", preds_all)

print(" --- completed ---")
