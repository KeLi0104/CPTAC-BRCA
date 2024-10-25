# %%
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from utils_vit import *

##===================================================================================================
def training_epoch(model, optimizer, train_set, loss_fn, device, batch_size):
    model.train()
    n_slides_train = len(train_set)

    ## shuffle training set
    idx_list = np.arange(n_slides_train)
    np.random.shuffle(idx_list)

    loss_list = []
    labels = []
    probs = []

    for i_batch in range(0, n_slides_train, batch_size):    
        n_slides_batch = min(batch_size, n_slides_train - i_batch)

        ## for each batch
        batch_loss = 0
        batch_labels = []
        batch_probs = []

        optimizer.zero_grad()

        for k in range(n_slides_batch):
            idx = idx_list[i_batch + k]

            x, y = train_set[idx]

            x = x.float().to(device)
            y = y.long().to(device)

            ret_val = model(x)
            logit, y_prob, y_hat = ret_val["logits"], ret_val["Y_prob"], ret_val["Y_hat"]
            loss = loss_fn(logit, y)
            loss.backward()
            batch_loss += loss.item()

            batch_labels.append(y.detach().cpu().numpy())
            batch_probs.append(y_prob.detach().cpu().numpy())

        optimizer.step()

        batch_loss /= n_slides_batch
        loss_list.append(batch_loss)

        labels.extend(batch_labels)
        probs.extend(batch_probs)

    labels = np.concatenate(labels)
    probs = np.concatenate(probs)

    return np.mean(loss_list), labels, probs

##===================================================================================================
def predict(model, valid_set, loss_fn, device):
    model.eval()
    n_slides_valid = len(valid_set)
    
    labels = [] ; probs = []
    loss_list = []
    with torch.no_grad():
        for i in range(n_slides_valid):
            x,y = valid_set[i]
            ret_val = model(x)
            logit,y_prob,y_hat = ret_val["logits"],ret_val["Y_prob"],ret_val["Y_hat"]
            
            loss = loss_fn(logit,y.long().to(device)) 
            loss_list.append(loss)
            labels.append(y.detach().cpu().numpy())
            probs.append(y_prob.detach().cpu().numpy())
            
    valid_loss = np.mean(loss_list)
    labels = np.concatenate(labels)
    probs = np.concatenate(probs)
    
    return valid_loss, labels, probs
##===================================================================================================
def fit(model, optimizer, loss_fn, max_epochs, patience, device,\
    train_set, valid_set, batch_size):
    
    train_losses = [] ; train_accs = [] ; train_aucs = []
    valid_losses = [] ; valid_accs = [] ; valid_aucs = []
    epoch_since_best = 0 ; valid_acc_old = -1. 
    for epoch in range(max_epochs):
        epoch_since_best += 1

        ## train
        train_loss, train_label, train_prob = training_epoch(model,optimizer,train_set,loss_fn,device,batch_size)

        ## predict on the valid set
        valid_loss, valid_label, valid_prob = predict(model, valid_set, loss_fn, device)

        train_pred = np.argmax(train_prob,axis=1)
        valid_pred = np.argmax(valid_prob,axis=1)

        train_acc = np.mean(train_label == train_pred)
        valid_acc = np.mean(valid_label == valid_pred)

        train_auc = roc_auc_score(train_label, train_prob[:,1])
        valid_auc = roc_auc_score(valid_label, valid_prob[:,1])

        print(f"{epoch}, train_loss: {train_loss:.4f}, acc: {train_acc:.4f}, auc: {train_auc:.4f},\
        valid_loss: {valid_loss:.4f}, acc: {valid_acc:.4f}, auc: {valid_auc:.4f}")

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        train_aucs.append(train_auc)

        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)
        valid_aucs.append(valid_auc)

        if valid_acc > valid_acc_old:
            epoch_since_best = 0
            valid_acc_old = valid_acc

        if epoch_since_best == patience:
            print('Early stopping at epoch {}'.format(epoch + 1))
            break

    return model,train_losses,valid_losses,train_accs,valid_accs,train_aucs,valid_aucs

##===================================================================================================
def plot_result(result_dir,model,train_losses,valid_losses,train_accs,valid_accs,train_aucs,valid_aucs):
    
    ## save trained model
    torch.save(model.state_dict(), f"{result_dir}model_trained.pth")

    nx,ny = 3,1
    fig, ax = plt.subplots(ny,nx,figsize=(nx*4,ny*3))

    ax[0].plot(train_losses, 'k-', label="train")
    ax[0].plot(valid_losses, 'b--', label="valid")

    ax[1].plot(train_accs, 'k-', label="train")
    ax[1].plot(valid_accs, 'b--', label="valid")

    ax[2].plot(train_aucs, 'k-', label="train")
    ax[2].plot(valid_aucs, 'b--', label="valid")

    ax[1].set_title(f"ACC train: {round(train_accs[-1],4)}, valid: {round(valid_accs[-1],4)}")
    ax[2].set_title(f"AUC train: {round(train_aucs[-1],4)}, valid: {round(valid_aucs[-1],4)}")

    for i in range(nx):
        ax[i].set_xlabel("n_epochs")
        ax[i].legend()

    ax[0].set_ylabel("loss")
    ax[1].set_ylabel("Accuracy")
    ax[2].set_ylabel("AUC")

    plt.tight_layout(h_pad=1, w_pad= 1.0)

    plt.savefig(f"{result_dir}loss.pdf", format='pdf', dpi=50)
##===================================================================================================
