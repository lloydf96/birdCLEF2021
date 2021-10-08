import colorednoise as cn
import numpy as np # linear algebra
import pandas as pd
import torch
import torchaudio 
from torch.utils.data import Dataset
import torchaudio.transforms as transform
import torch.nn.functional as F
import os
import torch.nn as nn
import torch.optim as optim
import librosa
import torchaudio.transforms as transform
from sklearn.model_selection import train_test_split
import random
from audio_dataset import *

def save_model(i,net,optimizer,loss_func,file_name):

    torch.save({'epoch': i,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_func
                }, os.path.join(os.getcwd(),file_name+'_'+str(i)+'.h5py'))
    torch.save(net,file_name+'_'+str(i)+'.pt')
    

def fit_and_evaluate(net, optimizer, loss_func, train_ds,dev_ds, n_epochs,sig_to_mel, batch_size=1):
    
    train_losses = []
    test_losses = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    
    loss_func = loss_func.cuda()
    file_log = train_ds.file_log
    train_dataset_len = len(train_ds)  
    data_path = train_ds.data_path
    label_to_idx = train_ds.label_to_idx
    sig = nn.Sigmoid()
  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = n_epochs,eta_min = 1e-6)
    test_dl = torch.utils.data.DataLoader(dataset=dev_ds, shuffle=True, batch_size=batch_size, pin_memory = True)
        
    for i in range(n_epochs):
        
        if i%4 == 1:
            save_model(i,net,optimizer,loss_func,'model')
        
        train_dl = torch.utils.data.DataLoader(dataset=train_ds, shuffle=True, batch_size=batch_size,pin_memory = True)
           
        cum_loss = 0
        net.train()
        
        for step,(train_x,train_y) in enumerate(train_dl):

            train_x,train_y = sig_to_mel(train_x).to(device),train_y.to(device)
            optimizer.zero_grad()
        
            net_output= net(train_x)
            loss = loss_func(net_output, train_y.type_as(net_output))
            
            with torch.no_grad():
                cum_loss += loss.detach()
                
            loss.backward()
            optimizer.step()
                
        scheduler.step()
        
        if (i%2== 1):
            net.eval()
            with torch.no_grad():
                #train_risk,train_f1,train_p,train_rc = epoch_loss(net, loss_func, train_dl)
                test_risk,test_f1,test_p,test_rc = epoch_loss(net, loss_func, test_dl)
                train_losses.append((i+1,cum_loss))
                test_losses.append((i+1,test_risk.detach()))
                
                print("Epoch: %s, Training loss: %s, Testing loss: %s " %(i,cum_loss/len(train_dl),test_risk))
                print("Epoch: %s, Testing f1: %s " %(i,test_f1))
                print("Epoch: %s, Testing precision: %s " %(i,test_p))
                print("Epoch: %s, Testing recall: %s " %(i,test_rc))
            net.train()
        
    return train_losses, test_losses

def fast_f1_score(predictions, target):
    tp = (predictions * target).sum(1)
    fp = (predictions * (1 - target)).sum(1)
    fn = ((1 - predictions) * target).sum(1)
    f1 = tp / (tp + (fp + fn) / 2)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    
    return f1.mean(), precision.mean(), recall.mean()
    
def epoch_loss(net, loss_func, dl):

        loss = 0   
        dl.dataset.add_noise = False
        f1_running = 0
        precision_running = 0
        recall_running = 0

        for x,y in dl:
            x,y = sig_to_mel(x).to(device),y.to(device)
            
            y_op= net(x).detach()
            loss += loss_func(y_op,y.type_as(y_op)).detach()
            f1,precision,recall = fast_f1_score(sig(y_op), y)
            f1_running +=f1
            precision_running +=precision
            recall_running += recall

        len_dl = len(dl)
       
        return  loss/len_dl,f1_running/len_dl,precision_running/len_dl,recall_running/len_dl
    
class process_signal:
    def __init__(self,melspec):
        self.adb =  torchaudio.transforms.AmplitudeToDB()
        self.melspec = melspec
        self.eps = 1e-5
    
    def __call__(self,x):
        mels = self.adb(self.melspec(x) + 1e-7)
        min_mels = torch.min(mels.reshape(-1,128*256),dim = 1)[0][(None,)*3].T
        mels_diff = torch.max(mels.reshape(-1,128*256),dim = 1)[0][(None,)*3].T - min_mels
        mels_diff[mels_diff == 0] = self.eps
        
        mels =  (mels - min_mels)/mels_diff
        return mels 