import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix,average_precision_score
import torch.optim.lr_scheduler as lr_scheduler

from vit import SimpleMLP
from vit import ViTAggregation
from utils_vit import *
from model_vit import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

init_random_seed(random_seed=42)
project = "CPTAC-BRCA"

# %%
try:
    ik_fold = int(sys.argv[1])
    il_fold = int(sys.argv[2])
    max_epochs,patience = 150,20
    
except:
    ik_fold = 0
    il_fold = 0
    max_epochs,patience = 150,20
    
    
for ik_fold in range(5):
    for il_fold in range(5):
        path2target = f"../13Classification/mutation_data/{project}_actual_mutation.pkl" #use actual genes mutation data as target(top 10 correlation)
        path2split = f"../10metadata/{project}_train_valid_test_idx.npz"
        path2features = f"../10metadata/{project}_features.npy"
        
        print("ik_fold:", ik_fold)
        print("il_fold:", il_fold)
        print(f"max_epochs: {max_epochs}, patience: {patience}")
        
        # %%
        targets = ['BIRC5','CCNB1','CDC20','CEP55','MKI67','NDC80','NUF2','PTTG1','RRM2','TYMS','UBE2C','BRCA1','BRCA2']
        targets = targets[12:13]
        print('gene:',targets)
        
        n_outputs = 2
        print("n_outputs:", n_outputs)
        
        # %%
        # train configurations:
        class_weight_based_slide = True
        learning_rate = 1.0e-5
        
        # model configurations
        n_blocks = 1
        n_heads = 2
        dim_head = 8
        
        mlp_dim = 128
        n_inputs = 1024
        aggr = "gap" 
        dropout = 0.2
        batch_size = 16
        
        ## create result directory
        target = str(targets[0])
        result_dir = f"results/{target}/result_{ik_fold}_{il_fold}/"
        os.makedirs(result_dir,exist_ok=True)
        
       #%%
        ## load data
        train_set, valid_set, test_set = load_dataset(path2features, path2target, path2split, 
                                                               ik_fold, il_fold, targets)
        # %%
        ## model
        vit_model_dict = {"heads":n_heads, "dim_head":dim_head, "mlp_dim":mlp_dim, "dim":n_inputs,
                         "depth":n_blocks, "aggr":aggr,"n_classes":n_outputs,"dropout":dropout,
                        }
        
        model = SimpleMLP(n_inputs=1024, n_hiddens=32, n_outputs=2, dropout=0)
        
        # relocate
        model.to(device)
        
        
         
        #%%
        if class_weight_based_slide:
            train_labels = np.array([train_set[i][1].numpy().flatten() if isinstance(train_set[i][1], torch.Tensor) else train_set[i][1] for i in range(len(train_set))])
            class_weights = sk_class_weights(train_labels)
            
            print("class_weights:", class_weights)
            np.savetxt(f"{result_dir}slide_weights.txt", class_weights, fmt="%s")
        
            loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor(class_weights).to(device))
        else:
            loss_fn = nn.CrossEntropyLoss()
            
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
        
        #%%
        print(" --- fit --- ")
        start_time = time.time()
        
        model, train_losses, valid_losses, train_accs, valid_accs, train_aucs, valid_aucs = fit(model, optimizer, loss_fn, max_epochs, patience, device, train_set, valid_set, batch_size)

        #%%
        print("Model parameters:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.data}")
        
        
        # %%
        ## plot
        plot_result(result_dir,model,train_losses,valid_losses,train_accs,valid_accs,train_aucs,valid_aucs)
        
        # %%
        ## predict on the test set
        _, test_label, test_prob = predict(model, test_set, loss_fn, device)
        
        test_pred = np.argmax(test_prob,axis=1)
        test_acc = np.mean(test_label == test_pred)
        test_auc = roc_auc_score(test_label, test_prob[:,1])
        test_ap = average_precision_score(test_label, test_prob[:,1])
        
        print(f"test_acc: {round(test_acc,4)}, test_auc: {round(test_auc,4)}, test_ap: {round(test_ap,4)}")
        
        ## save result
        np.savetxt(f"{result_dir}label_pred.txt", np.hstack((np.array((test_label, test_pred)).T, test_prob)), fmt="%s")
        np.savetxt(f"{result_dir}acc_auc.txt", np.array((round(test_acc,4),round(test_auc,4),round(test_ap,4))), fmt="%s")
        
        # %%
        print(f"fit -- completed --: {round(time.time() - start_time, 2)}s")
