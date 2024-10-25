import pandas as pd
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the provided pickle file
actual_file_path = '../13Classification/analysis_results/CPTAC-BRCA_beta_test_actual.pkl'
actual_genes = pd.read_pickle(actual_file_path)

pred_file_path = '../13Classification/analysis_results/CPTAC-BRCA_beta_test_pred.pkl'
pred_genes = pd.read_pickle(pred_file_path)

# Extract gene expression data, excluding metadata columns
actual_genes = actual_genes.iloc[:, 6:]
pred_genes = pred_genes.iloc[:, 6:]

# Read the actual gene expression data of “whole population” – TCGA dataset
actual_genes_whole_population = pd.read_pickle('../10metadata/BRCA_actual_patient.pkl').iloc[:,1:]

# Calculate the median expression level for each gene
gene_medians = actual_genes_whole_population.median(axis=0)
gene_min = actual_genes_whole_population.min(axis=0)
gene_max = actual_genes_whole_population.max(axis=0)
gene_list = actual_genes.columns
gene_medians = gene_medians[gene_list]
gene_min = gene_min[gene_list]
gene_max = gene_max[gene_list]

#get predicted probs
pred_probs_values = np.zeros(pred_genes.shape)
for i in range(pred_genes.shape[1]):
    pred_probs_values[:,i] = 2*(pred_genes.iloc[:,i]-gene_min.iloc[i])/(gene_max.iloc[i]-gene_min.iloc[i])-1
    pred_probs_values[:,i] = np.array(torch.sigmoid(torch.tensor(pred_probs_values[:,i])))
    
pred_probs = pd.DataFrame(pred_probs_values, columns=gene_list)

# Mark genes as mutated (1) if their expression is greater than the median, otherwise non-mutated (0)
actual_mutation_status = (actual_genes > gene_medians).astype(int)
pred_mutation_status = (pred_genes > gene_medians).astype(int)
#pred_mutation_status = (pred_probs > 0.5).astype(int)

#get gene-wise accuracy
accuracy = (actual_mutation_status == pred_mutation_status).mean(axis=0)
targets = ['BIRC5','CCNB1','CDC20','CEP55','MKI67','NDC80','NUF2','PTTG1','RRM2','TYMS','UBE2C','BRCA1','BRCA2']
accuracy_13genes = accuracy[targets]

# Output the result
output_dir ='../13Classification/mutation_data'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
actual_mutation_status.to_pickle('../13Classification/mutation_data/CPTAC-BRCA_actual_mutation.pkl')
pred_mutation_status.to_pickle('../13Classification/mutation_data/CPTAC-BRCA_pred_mutation.pkl')
pred_probs.to_pickle('../13Classification/mutation_data/CPTAC-BRCA_pred_probs.pkl')

# Display the first few rows of the mutation status DataFrame
print(actual_mutation_status.head())
print(pred_mutation_status.head())

# Display the accuracy
print("Accuracy for each gene:\n", accuracy.head())

# Plot the distribution of gene accuracy
plt.figure(figsize=(12, 6))
plt.hist(accuracy, bins=50, color='blue', alpha=0.7)
plt.title('Distribution of Gene Accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Number of Genes')
plt.grid(axis='y', alpha=0.75)
plt.show()

from collections import Counter
count_accuracy = Counter(accuracy)