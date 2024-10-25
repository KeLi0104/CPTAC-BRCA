# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Subset,Dataset
from scipy.stats import pearsonr
import argparse

##===================================================================================================
class slide_target_dataset(Dataset):

    def __init__(self, features, targets):
        
        self.features = features
        self.targets = targets    
        self.dim = self.features[0][1].shape[1] ## n_features

    def __getitem__(self, index):
        sample = torch.Tensor(self.features[index][1]).float()
        target = torch.Tensor(self.targets[index]).float()
        
        return sample, target

    def __len__(self):
        return len(self.features)

##===================================================================================================
def load_dataset(path2features, path2target, path2split, ik_fold, il_fold, target_cols):
    
    ## load image feature
    features = np.load(path2features, allow_pickle=True)
    print("len(features):", len(features))

    ## load target
    df_target = pd.read_pickle(path2target)[target_cols]
    targets = df_target[target_cols].values
    print("targets.shape:", targets.shape)

    ## create dataset
    dataset = slide_target_dataset(features, targets)

    ## load_train_valid_test_idx:
    train_valid_test_idx = np.load(path2split, allow_pickle=True)

    train_idx = train_valid_test_idx["train_idx"][ik_fold][il_fold]
    valid_idx = train_valid_test_idx["valid_idx"][ik_fold][il_fold]
    test_idx = train_valid_test_idx["test_idx"][ik_fold]

    ## split train, valid, test dataset
    train_set = Subset(dataset, train_idx)
    valid_set = Subset(dataset, valid_idx)
    test_set = Subset(dataset, test_idx)
    
    return train_set, valid_set, test_set

##===================================================================================================
def print_network(net):
    num_params = 0
    num_params_train = 0
    print(net)
    
    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

##===================================================================================================
class GenericTensorNamespace(argparse.Namespace):
    def __init__(self):
        super(GenericTensorNamespace,self).__init__()
    def to(self,device,non_blocking=False) -> "GenericTensorNamespace":
        for nm in dir(self):
            obj = getattr(self,nm)
            if type(obj)==torch.Tensor:
                setattr(self,nm,obj.to(device,non_blocking=non_blocking))
            
        return self
    def numpy(self) -> "GenericTensorNamespace":
        new_ns=GenericTensorNamespace()
        for nm in dir(self):
            obj = getattr(self,nm)
            if type(obj)==torch.Tensor:
                setattr(new_ns,nm,obj.cpu().numpy())
        return new_ns

##===================================================================================================
def plot_result(result_dir,model,train_loss,valid_loss,valid_coef,valid_slope):

    fig, ax = plt.subplots(ny,nx,figsize=(nx*4,ny*3))

    ax[0].plot(train_loss, 'k-', label="train")
    ax[0].plot(valid_loss, 'b--', label="valid")


    for i in range(nx):
        ax[i].set_xlabel("n_epochs")
        ax[i].legend()

    ax[0].set_ylabel("loss")

    plt.tight_layout(h_pad=1, w_pad= 1.0)

    plt.savefig(f"{result_dir}loss.pdf", format='pdf', dpi=50)

##===================================================================================================
def pearson_r_and_p(label, pred):
    R,p = pearsonr(label, pred)
    if R> 0:
        p_1side = p/2.
    else:
        p_1side = 1-p/2

    return p_1side
    
def compute_coef_slope(labels, preds):
    coef = np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])
    slope = np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])
    
    return coef, slope

def compute_coef_slope_p(labels, preds):
    coef = np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])
    slope = np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])
    p_value = np.array([pearson_r_and_p(labels[:,i], preds[:,i]) for i in range(preds.shape[1])])

    return coef, slope, p_value

def compute_coef_slope_padj(labels, preds):
    coef = np.array([pearsonr(labels[:,i], preds[:,i])[0] for i in range(labels.shape[1])])
    slope = np.array([np.polyfit(labels[:,i], preds[:,i], 1)[0] for i in range(labels.shape[1])])
    p_value = np.array([pearson_r_and_p(labels[:,i], preds[:,i]) for i in range(preds.shape[1])])

    p_adj = smt.multipletests(p_value, alpha=0.05, method='hs', is_sorted=False, returnsorted=False)[1]

    return coef, slope, p_adj

##===================================================================================================
def sk_class_weights(labels):
    u, c = np.unique(labels, return_counts=True)
    n_samples = sum(c)

    weights = n_samples/(len(c)*c)
    
    return weights
##===================================================================================================
def init_random_seed(random_seed=42):
    # Python RNG
    np.random.seed(random_seed)

    # Torch RNG
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False