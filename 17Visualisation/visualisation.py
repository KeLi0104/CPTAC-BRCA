import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score,average_precision_score
from sklearn.metrics import roc_curve, auc as calculate_auc

result_dir = 'plots/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

#%%Indirect model--coef distribution and cumulative_coef plot
# Load the provided pickle file
actual_file_path = '../13Classification/analysis_results/CPTAC-BRCA_beta_test_actual.pkl'
actual_genes = pd.read_pickle(actual_file_path)

pred_file_path = '../13Classification/analysis_results/CPTAC-BRCA_beta_test_pred.pkl'
pred_genes = pd.read_pickle(pred_file_path)

# Extract gene expression data, excluding metadata columns
actual_genes = actual_genes.iloc[:, 6:]
pred_genes = pred_genes.iloc[:, 6:]

# Calculate Pearson correlation coefficient for each gene
coef = np.array([pearsonr(actual_genes.iloc[:, i], pred_genes.iloc[:, i])[0] for i in range(actual_genes.shape[1])])
mean_coef = np.mean(coef)
print('Average coef of 18272 genes:',mean_coef)
median_coef = np.median(coef)
print('Median coef of 18272 genes:',median_coef)
sorted_coef = sorted(coef, reverse=True)
# Calculate cumulative coefficients
thresholds = np.linspace(0, 1, 21)
cumulative_counts_coef = [np.sum(coef >= t) for t in thresholds]
#print('Cumulative coef:',cumulative_counts)

#plot
nx,ny = 2,1
fig, ax = plt.subplots(ny,nx,figsize=(nx*3.5,ny*3))

bins1 = np.linspace(-0.2,0.8,41, endpoint=True)
ax[0].hist(coef,bins=bins1,histtype='bar',color="#A5C2E2",edgecolor="#6B7EB9",rwidth=0.85)
#ax[0].plot([R0_min, R0_min],[0,50000], "--", color="red", label="p-adj=0.05")
#ax[0].plot([0, 0],[0,2000], "--", color="red", label="p-adj=0.05")
ax[0].vlines(0,0,1000,linestyles="dashed", colors="red")
ax[0].set_xlabel("Correlation")
ax[0].set_ylabel("Number of genes")
ax[0].set_xticks([-0.2,0,0.2,0.4,0.6,0.8])
#ax[0].set_ylim(0,2000)
#ax[0].set_yticks([500,1000,1500,2000])
#ax[0].legend()

ax[1].plot(thresholds,cumulative_counts_coef, "o-", label="DeepPT")
ax[1].set_xlabel("Correlation Threshold")
ax[1].set_ylabel("Number of Genes")
ax[1].set_xlim(0.4,0.8)
ax[1].set_ylim(0,3000)
#ax[1].plot(0.4,786, "^", label="HE2RNA")

#ax[1].legend()
plt.tight_layout(h_pad=1, w_pad= 1.5)
plt.savefig(f"{result_dir}indirect_regression.pdf", format='pdf', dpi=50)

#%%Indirect model--ACC and AUC 
#get mutation status
actual_mutation_status = pd.read_pickle('../13Classification/mutation_data/CPTAC-BRCA_actual_mutation.pkl')
pred_mutation_status = pd.read_pickle('../13Classification/mutation_data/CPTAC-BRCA_pred_mutation.pkl')
pred_probs = pd.read_pickle('../13Classification/mutation_data/CPTAC-BRCA_pred_probs.pkl')

#get gene-wise accuracy
accuracy = (actual_mutation_status == pred_mutation_status).mean(axis=0)
sorted_accuracy = sorted(accuracy, reverse=True)

#calculate indirect auc
auc = { gene:
    roc_auc_score(actual_mutation_status[gene], pred_probs[gene]) 
    for gene in actual_mutation_status.columns
    if len(set(actual_mutation_status[gene])) > 1}
auc = pd.Series(auc)
sorted_auc = sorted(auc, reverse=True)

#threshold count
cumulative_counts_acc = [np.sum(accuracy >= t) for t in thresholds]
cumulative_counts_auc = [np.sum(auc >= t) for t in thresholds]

#print average ACC and AUC
print('Average ACC of 18272 genes:', np.mean(accuracy))
print('Median ACC of 18272 genes:', np.median(accuracy))
print('Average AUC of 18272 genes:', np.mean(auc))
print('Median AUC of 18272 genes:', np.median(auc))

#plot-ACC
nx,ny = 2,1
fig, ax = plt.subplots(ny,nx,figsize=(nx*3.5,ny*3))

ax[0].hist(accuracy, np.linspace(0.4,1,30, endpoint=True), histtype='bar',color="#A5C2E2",edgecolor="#6B7EB9")
ax[0].set_title('Distribution of Gene Accuracy')
ax[0].set_xlabel('Accuracy')
ax[0].set_ylabel('Number of Genes')
ax[0].grid(axis='y', alpha=0.75)

ax[1].plot(thresholds,cumulative_counts_acc, "o-", label="DeepPT")
ax[1].set_xlabel("Accuracy Threshold")
ax[1].set_ylabel("Number of Genes")
ax[1].set_xlim(0.7,1)
ax[1].set_ylim(0,8000)

plt.tight_layout(h_pad=1, w_pad= 1.5)
plt.savefig(f"{result_dir}indirect_classification_ACC.pdf", format='pdf', dpi=50)

#plot-AUC
nx,ny = 2,1
fig, ax = plt.subplots(ny,nx,figsize=(nx*3.5,ny*3))
ax[0].hist(auc, np.linspace(0.3,0.9,50, endpoint=True), histtype='bar',color="#A5C2E2",edgecolor="#6B7EB9")
ax[0].set_title('Distribution of Gene AUC')
ax[0].set_xlabel('AUC')
ax[0].set_ylabel('Number of Genes')
ax[0].grid(axis='y', alpha=0.75)
#ax[0].plot(0.4,786, "^", label="HE2RNA")
#ax[0].legend()

ax[1].plot(thresholds,cumulative_counts_auc, "o-", label="DeepPT")
ax[1].set_xlabel("AUC Threshold")
ax[1].set_ylabel("Number of Genes")
ax[1].set_xlim(0.7,1)
ax[1].set_ylim(0,4000)

plt.tight_layout(h_pad=1, w_pad= 1.5)
plt.savefig(f"{result_dir}indirect_classification_AUC.pdf", format='pdf', dpi=50)

#%%Direct vs Indirect model--13 genes ACC,AUC
gene_list = ['BIRC5','CCNB1','CDC20','CEP55','MKI67','NDC80','NUF2','PTTG1','RRM2','TYMS','UBE2C','BRCA1','BRCA2']
direct_acc_auc_ap = pd.read_csv('../14Direct_ViT/analysis_results/acc_auc_ap.csv').iloc[:,1:]


#acc
direct_acc = direct_acc_auc_ap[gene_list].iloc[0]
indirect_acc = accuracy[gene_list]
print("Average ACC of 13 genes(indirect):", np.mean(indirect_acc))

#print(np.mean(indirect_acc))

x = np.arange(len(gene_list))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 8))

rects1 = ax.bar(x - width/2, direct_acc, width, label='Direct model accuracy',color='#FFE0C1')
rects2 = ax.bar(x + width/2, indirect_acc, width, label='Indirect model accuracy',color='#FEA040')

ax.set_xlabel('Genes')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy by Gene and Model')
ax.set_xticks(x)
ax.set_xticklabels(gene_list)
ax.legend()
plt.savefig(f"{result_dir}ACC_comparison.pdf", format='pdf', dpi=50)



#auc
direct_auc = direct_acc_auc_ap[gene_list].iloc[1]
indirect_auc = auc[gene_list]
print("Average AUC of 13 genes(indirect):", np.mean(indirect_auc))

x = np.arange(len(gene_list))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 8))

rects1 = ax.bar(x - width/2, direct_auc, width, label='Direct model AUC',color='#FFE0C1')
rects2 = ax.bar(x + width/2, indirect_auc, width, label='Indirect model AUC',color='#FEA040')

ax.set_xlabel('Genes')
ax.set_ylabel('AUC')
ax.set_title('AUC by Gene and Model')
ax.set_xticks(x)
ax.set_xticklabels(gene_list)
ax.legend()
plt.savefig(f"{result_dir}AUC_comparison.pdf", format='pdf', dpi=50)



#AP
direct_ap = direct_acc_auc_ap[gene_list].iloc[2]
#calculate indirect AP
AP = { gene:
    average_precision_score(actual_mutation_status[gene], pred_probs[gene]) 
    for gene in actual_mutation_status.columns
    if len(set(actual_mutation_status[gene])) > 1}
AP = pd.Series(AP)
indirect_ap = AP[gene_list]
print("Average AP of 13 genes(indirect):", np.mean(indirect_ap))

x = np.arange(len(gene_list))
width = 0.35

fig, ax = plt.subplots(figsize=(15, 8))

rects1 = ax.bar(x - width/2, direct_ap, width, label='Direct model AP',color='#FFE0C1')
rects2 = ax.bar(x + width/2, indirect_ap, width, label='Indirect model AP',color='#FEA040')

ax.set_xlabel('Genes')
ax.set_ylabel('AP')
ax.set_title('AP by Gene and Model')
ax.set_xticks(x)
ax.set_xticklabels(gene_list)
ax.legend()
plt.savefig(f"{result_dir}AP_comparison.pdf", format = 'pdf', dpi = 50)

#%%ROC for each gene
pred_probs_all = pd.read_pickle('../14Direct_ViT/analysis_results/pred_probs_all.pkl')

for gene in gene_list:
    plt.figure()
    print(gene)

    fpr1, tpr1, _ = roc_curve(actual_mutation_status[gene], pred_probs[gene])#indirect
    roc_auc1 = calculate_auc(fpr1, tpr1)

    fpr2, tpr2, _ = roc_curve(actual_mutation_status[gene], pred_probs_all[gene])#direct
    roc_auc2 = calculate_auc(fpr2, tpr2)

    plt.plot(fpr1, tpr1, color='#FF6100', lw=2, label='Indirect Model (AUC = %0.2f)' % roc_auc1)

    plt.plot(fpr2, tpr2, color='#FEA040', lw=2, label='Direct Model (AUC = %0.2f)' % roc_auc2)

    plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {gene}')
    plt.legend(loc="lower right")
    plt.savefig(f"{result_dir}ROC{gene}.pdf", format = 'pdf', dpi = 50)
    plt.show()
