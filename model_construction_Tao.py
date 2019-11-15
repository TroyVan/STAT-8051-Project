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
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

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
test_np = test.iloc[:, 1:41].to_numpy()

for i in range(1, 41):
    if sum(train_np[:,i]) == 0:
        continue
    else:
        train_np[:,i] = (train_np[:,i] - train_np[:,i].mean()) / train_np[:,i].std()

for i in range(40):
    if sum(test_np[:,i]) == 0:
        continue
    else:
        test_np[:,i] = (test_np[:,i] - test_np[:,i].mean()) / test_np[:,i].std()

#%% Nerual Network

inputs = torch.from_numpy(train_np[:, 1:41]).type(torch.FloatTensor)
labels = torch.from_numpy(train_np[:, 0]).type(torch.LongTensor)
tests = torch.from_numpy(test_np).type(torch.FloatTensor)

class Net(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(40, 80)
        self.fc2 = nn.Linear(80, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        softm = F.softmax(x, dim = 1)
        return softm

network = Net()
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr = 0.01)

epochs = 100
losses = []

for i in range(epochs):
    #Precit the output for Given input
    y_pred = network.forward(inputs)
    #Compute Cross entropy loss
    loss = loss_fun(y_pred, labels)
    loss = loss/len(inputs)
    #Add loss to the list
    losses.append(loss.item())
    #Clear the previous gradients
    optimizer.zero_grad()
    #Compute gradients
    loss.backward()
    #Adjust weights
    optimizer.step()

plt.figure(figsize=(8,6))
plt.plot(losses, marker='', linewidth = 2, linestyle='--', label = 'Train Loss')
plt.legend(fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Train Loss', fontsize=20)
plt.yticks(fontsize=20)
plt.savefig('fig.png')

roc_auc_score(labels.numpy(), torch.tensor(network(inputs)).numpy()[:, 1])

test['conv_prob'] = torch.tensor(network(tests)).numpy()[:, 1]

test[['policy_id', 'conv_prob']].to_csv('test_result_tao.csv', index = False)
