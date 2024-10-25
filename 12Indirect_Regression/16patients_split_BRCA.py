import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,KFold
import warnings 

warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

## initialize
random_seed = 0

n_outers = 5
n_inners = 5

project = "CPTAC-BRCA"
extension = ""
patient_col = "patient_id"

print(" ")
print(project)

np.random.seed(random_seed)

## read info:
metadata = pd.read_csv(f"../10metadata/{project}_slide_matched{extension}.csv")

n_slides = metadata.shape[0]
print("n_slides:", n_slides)
slide_idx = np.arange(n_slides)

## patients = metadata.patient_id.values
patients = metadata[patient_col].values
print("patients:", patients)

patients_unique = np.unique(patients)
n_patients_unique = len(patients_unique)
print("n_patients_unique:", n_patients_unique)

## inner loop:
def inner_cv(slide_idx, patients, patients_train_valid, n_inners, random_seed):
    
    skf = KFold(n_splits = n_inners, shuffle=True, random_state=random_seed)

    slide_idx_train_list = []
    slide_idx_valid_list = []
    for patient_idx_train, patient_idx_valid in skf.split(patients_train_valid):

        patients_train = patients_train_valid[patient_idx_train]
        patients_valid = patients_train_valid[patient_idx_valid]

        slide_idx_train = np.array([slide_idx[i] for i in range(n_slides) if patients[i] in patients_train])
        slide_idx_valid = np.array([slide_idx[i] for i in range(n_slides) if patients[i] in patients_valid])        

        slide_idx_train_list.append(slide_idx_train)
        slide_idx_valid_list.append(slide_idx_valid)
    
    return slide_idx_train_list, slide_idx_valid_list



## outer loop
skf = KFold(n_splits = n_outers, shuffle=True, random_state=random_seed)

slide_idx_test_list = []
slide_idx_train_list2 = []
slide_idx_valid_list2 = []
for patient_idx_train_valid, patient_idx_test in skf.split(patients_unique):
    patients_train_valid = patients_unique[patient_idx_train_valid]
    patients_test = patients_unique[patient_idx_test]

    ## patients --> slide_idx
    slide_idx_test = np.array([slide_idx[i] for i in range(n_slides) if patients[i] in patients_test])

    slide_idx_test_list.append(slide_idx_test)

    ## ---- inner k-fold ----
    slide_idx_train_list, slide_idx_valid_list = inner_cv(slide_idx, patients, patients_train_valid, n_inners, random_seed)
    slide_idx_train_list2.append(slide_idx_train_list)
    slide_idx_valid_list2.append(slide_idx_valid_list)
    
    

## use test set for patients that have drug response data
try:
    patients_test = np.loadtxt("%s_patient_test.txt"%project, dtype="str")
    print("patients_test.shape:", patients_test.shape)

    patients_train_valid = np.setdiff1d(patients_unique, patients_test)
    print("patients_train_valid.shape:", patients_train_valid.shape)

    ## patients --> slide_idx
    slide_idx_test = np.array([slide_idx[i] for i in range(n_slides) if patients[i] in patients_test])
    slide_idx_test_list.append(slide_idx_test)

    slide_idx_train_list, slide_idx_valid_list = inner_cv(slide_idx, patients, patients_train_valid, n_inners, random_seed)
    slide_idx_train_list2.append(slide_idx_train_list)
    slide_idx_valid_list2.append(slide_idx_valid_list)
    
except:
    print("cannot find patient_test.txt file")
    
    
np.savez(f"{project}_train_valid_test_idx{extension}.npz", train_idx=slide_idx_train_list2,\
         valid_idx=slide_idx_valid_list2, test_idx=slide_idx_test_list)