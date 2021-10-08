
import numpy as np 
import pandas as pd
import torch

import torchaudio.transforms as transform
import torch.nn.functional as F
import os
import torch.nn as nn
import torch.optim as optim
import librosa
import random

class AudioConvNet(nn.Module):
    def __init__(self,melspec,op_size):
      
        super().__init__()
        self.eps = 1e-5
        self.adb = torchaudio.transforms.AmplitudeToDB()
        self.melspec = melspec
        self.conv = nn.Sequential(nn.Conv2d(1,32,(5,5),padding= 2),
                                  nn.ReLU(),
                                  
                                  nn.MaxPool2d((2,2)),
                                  nn.Conv2d(32,64,(3,3),padding = 1),
                                  nn.ReLU(),
                                  
                                  nn.MaxPool2d((2,2)),
                                  
                                  nn.Conv2d(64,128,(3,3),padding = 1),
                                  nn.ReLU(),
                                  
                                  nn.MaxPool2d((2,2)),
                                  nn.Conv2d(128,256,(3,3),padding = 1),
                                  nn.ReLU(),
                                  
                                  nn.MaxPool2d((2,2)),
                                  nn.Conv2d(256,512,(3,3),padding = 1),
                                  nn.ReLU(),
                                  
                                  nn.MaxPool2d((2,2)),
                                  nn.Conv2d(512,1024,(3,3),padding = 1),
                                  nn.ReLU(),
                                  
                                  nn.MaxPool2d((2,2)),
                                  nn.Conv2d(1024,2048,(3,3),padding = 1),
                                  nn.ReLU(),
                                  
                                  nn.MaxPool2d((2,2))
                                  )
        self.conv = self.conv.apply(self.init_he)
        
        self.globalMaxPool = nn.AdaptiveMaxPool2d((1,1))
        self.dense = nn.Sequential(nn.Linear(2048,2048),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(2048,1024),
                                   nn.ReLU(),
                                   nn.Dropout(0.5),
                                   nn.Linear(1024,op_size))
                                   

    def forward(self, x):
 
        mels = self.adb(self.melspec(x) + 1e-7)
        min_mels = torch.min(mels.reshape(-1,128*256),dim = 1)[0][(None,)*3].T
        mels_diff = torch.max(mels.reshape(-1,128*256),dim = 1)[0][(None,)*3].T - min_mels
        mels_diff[mels_diff == 0] = self.eps
        
        mels =  (mels - min_mels)/mels_diff
        
        conv_op = self.conv(mels)
        means = torch.squeeze(self.globalMaxPool(conv_op))
        
        dense_op = self.dense(means)
        
        
        return dense_op
    
    @staticmethod
    def init_he(layer):
        if type(layer) == torch.nn.Conv2d:
            torch.nn.init.kaiming_uniform_(layer.weight)
        
