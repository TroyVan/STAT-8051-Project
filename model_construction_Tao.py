# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:16:51 2019

@author: Tao Tao
"""

#%% load packages and datasets

import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn

#set default folder
import os
os.chdir(r'C:\UMN\Course\2019_fall\STAT 8051\FP\STAT-8051-Project')

train = pd.read_csv('train.csv', delimiter=",")
test = pd.read_csv('test.csv', delimiter=",")

# check the missing values
train.isnull().sum()
test.isnull().sum()

# drop missing values for train dataset
train = train.fillna(0)
test = test.fillna(0)

train_np = train.to_numpy()
test_np = test.to_numpy()

#%% Nerual Network

train_tensor = torch.from_numpy(train_np)

class Net(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

roc_auc_score(y_train, y_pred[:,1])
