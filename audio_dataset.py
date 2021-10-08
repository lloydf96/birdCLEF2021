
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

class AudioDataset(Dataset):
    def __init__(self,data_path,csv_path,sr,file_log=None,label_to_idx = None,add_noise = True,num_of_steps = None):
        super().__init__()
        self.data_path = data_path
        self.csv_path = csv_path
        self.sr = sr
        self.add_noise = add_noise
        self.num_of_steps = num_of_steps
        
        if label_to_idx is None:
            self.file_log = pd.read_csv(csv_path)
            labels = self.file_log.primary_label.unique()
            self.label_len = len(labels)
            self.label_to_idx = dict(zip(labels,range(self.label_len) ))
            self.idx_to_label = dict(zip(range(self.label_len ),labels))
            
        else:
            self.label_to_idx = label_to_idx
            self.label_len = len(label_to_idx)
            self.idx_to_label = {y:x for x,y in label_to_idx.items()}
            
        if file_log is None:
            self.file_log = pd.read_csv(csv_path)
            self.file_log['secondary_labels'] = self.file_log['secondary_labels'].apply(lambda x: x.replace("[","").replace("]","").replace("'","").replace(" ","").split(","))
            valid_labels = list(self.label_to_idx.keys())
            self.file_log['secondary_labels'] = self.file_log['secondary_labels'].apply(lambda x: list(set(x) & set(valid_labels)))

        else:
            self.file_log = file_log.reset_index(drop = True)
        
        if num_of_steps is not None:
            self.file_log = self.file_log.sample(num_of_steps).reset_index(drop = True)
                               
    def __len__(self):
        return len(self.file_log)

    def __getitem__(self, idx):
        
        label = self.file_log.primary_label[idx]
        path = os.path.join(self.data_path,label,self.file_log.filename[idx])
        y,sr = torchaudio.load(path,normalization = True)
        secondary_label = self.file_log.secondary_labels[idx]
        label_list = torch.ones(self.label_len )*0.0002
        
        if (len(secondary_label) > 0):
            label_list_sec = [self.label_to_idx[j] for j in secondary_label]
            label_list = torch.sum(F.one_hot(torch.tensor(label_list_sec) , num_classes = self.label_len),axis = 0)*0.3 + label_list
        
        if (torch.isnan(y).sum()>=1) or (torch.sum(y ** 2) == 0):
            y,label = self.conv_nan(y.numpy().squeeze())
            y = torch.unsqueeze(torch.tensor(y),dim = 0)
        
        if self.add_noise == True:
            y,label_list= self.AddNoise(y.numpy().squeeze(),label_list) 
            y = torch.unsqueeze(torch.tensor(y),dim = 0)
            
        label_list[self.label_to_idx['nocall']] = 0.1
        label_list[self.label_to_idx[label]] = 0.9
        
        return y,label_list
    
    def AddNoise(self,y,label_list):
        #GaussianNoiseSNR
        rand_noise = random.uniform(0,1)

        if rand_noise <0.15:
            y = self.AddGauss(y,5,10)

        elif rand_noise <0.35:
            y = self.AddPink(y,5,10)

        elif rand_noise <= 0.6:
            
            noise_id = random.randint(0,self.label_len-1)           
            folder_loc = os.path.join(self.data_path,self.idx_to_label[noise_id])
            bird_samples = [name for name in os.listdir(folder_loc) ]
            bird_sample = random.choice(bird_samples)
            y_noise,sr = librosa.load(os.path.join(folder_loc,bird_sample),sr = 32000)
            
            if (np.isnan(y_noise).sum()>=1) or (np.sum(y_noise ** 2) == 0):
                y_noise,label = self.conv_nan(y_noise)
                noise_id = self.label_to_idx[label]
                
            y = self.AddOtherSample(y,y_noise,3,7)
            label_list = label_list + F.one_hot(torch.tensor(noise_id) , num_classes = self.label_len)*0.3

        else:
            y = self.AddGauss(y,10,20)
            
        return y,label_list
    
    @staticmethod
    def conv_nan(y):
        snr = np.random.uniform(5,10)
        a_signal = 0.08
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        y = (white_noise * 1 / a_white * a_noise).astype(y.dtype)
        label = 'nocall'

        return y,label

    @staticmethod
    def AddGauss(y,min_snr,max_snr):
        snr = np.random.uniform(min_snr, max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        y = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return y
    
    @staticmethod
    def AddPink(y,min_snr,max_snr):
        snr = np.random.uniform(min_snr, max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        y = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return y
    
    @staticmethod
    def AddOtherSample(y,y_noise,min_snr,max_snr):
        a_other = np.sqrt(y_noise ** 2).max()
        snr = np.random.uniform(min_snr, max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))
        y = (y + y_noise * 1 / a_other * a_noise).astype(y.dtype)
        return y
    
    