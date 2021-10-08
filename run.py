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
from audionet import *
from audio_dataset import *
from train import *
import random


if __name__ == "__main__":
    base_dir = os.getcwd()
    H = 128
    W = 256
    fmin = 500
    fmax = 15000
    sample_size = 5
    hop_length = int(32000 * sample_size / (W - 1))
    
    n_epochs = 2
    batch_size = 4

    melspec = transform.MelSpectrogram(sample_rate=32000, n_mels=H,
                                                n_fft=1024, hop_length=hop_length,f_min = fmin,f_max = fmax)
    torch.cuda.empty_cache()
    loss_func = nn.BCEWithLogitsLoss()
    
    sig_to_mel = process_signal(melspec)

    train_path = os.path.join(base_dir,'train_test_dev_set','train')
    train_csv_path = os.path.join(train_path,'train.csv')
    
    dev_path = os.path.join(base_dir,'train_test_dev_set','dev')
    dev_csv_path = os.path.join(dev_path,'dev.csv')

    train_dataset = AudioDataset(train_path,train_csv_path,32000,add_noise = True)
    dev_dataset = AudioDataset(dev_path,dev_csv_path,32000,label_to_idx = train_dataset.label_to_idx)
    
    model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet101', pretrained=True)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=2048, out_features=train_dataset.label_len, bias=True)
    optimizer = optim.Adam(model.parameters(), lr=9e-5)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    
    train_loss,dev_loss = fit_and_evaluate(model, optimizer, loss_func, train_dataset, dev_dataset, n_epochs,sig_to_mel,batch_size)
    train_loss.to_csv('train_loss.csv')
    dev_loss.to_csv('test_loss.csv')