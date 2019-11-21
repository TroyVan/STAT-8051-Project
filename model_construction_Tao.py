# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:16:51 2019

@author: Tao Tao
"""

#%% load packages and datasets

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
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
test_np = test.iloc[:, 1:74].to_numpy()

for i in range(1, 74):
    if sum(train_np[:,i]) == 0:
        continue
    else:
        train_np[:,i] = (train_np[:,i] - train_np[:,i].mean()) / train_np[:,i].std()

for i in range(73):
    if sum(test_np[:,i]) == 0:
        continue
    else:
        test_np[:,i] = (test_np[:,i] - test_np[:,i].mean()) / test_np[:,i].std()

inputs = torch.from_numpy(train_np).type(torch.FloatTensor)
testset = torch.from_numpy(test_np).type(torch.FloatTensor)

#%% Nerual Network one
class SimpleNet(nn.Module):
   
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(73, 2)
        self.dropout1 = nn.Dropout(0.1)
        self.relu1 = nn.Tanh()
        self.norm1 = nn.BatchNorm1d(2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        y = self.out_act(x)
        return y
    
#%% Nerual Network two

class Net(nn.Module):
   
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(73, 512)
        self.dropout1 = nn.Dropout(0.1)
        self.relu1 = nn.PReLU()
        self.norm1 = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 180)
        self.relu2 = nn.PReLU()
        self.norm2 = nn.BatchNorm1d(180)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(180, 210)
        self.relu3 = nn.PReLU()
        self.norm3 = nn.BatchNorm1d(210)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(210, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.norm3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        y = self.out_act(x)
        return y

#%% cross validataion
epochs = 300
torch.manual_seed(8051)
np.random.seed(8051)
loss_fun = nn.BCELoss()
batch_size = 256
auc_table = np.zeros((epochs, 3))
loss_table = np.zeros((epochs, 3))

for i in range(3):
    tests = inputs[inputs[:,74] == i + 1]
    trains = inputs[inputs[:,74] != 1 + 1]
    network = SimpleNet()
    optimizer = optim.SGD(network.parameters(), lr = 0.001, momentum = 0.9)
    auc_list = []
    loss_list = []
    permutation = torch.randperm(len(trains))
    X = trains[:, 1:74]
    Y = trains[:, 0]
    for j in range(epochs):
        total_loss = 0
        for m in range(0, len(trains), batch_size):
            optimizer.zero_grad()
            
            indices = permutation[m:m+batch_size]
            batch_x, batch_y = X[indices], Y[indices]
            
            y_pred = network.forward(batch_x)
            loss = loss_fun(y_pred, batch_y.reshape(len(batch_y), 1))
            total_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        auc = roc_auc_score(tests[:, 0].numpy(), network(tests[:, 1:74]).clone().detach().numpy())
        auc_list.append(auc)
        loss_list.append(total_loss)
    auc_table[:, i] = auc_list
    loss_table[:, i] = loss_list

plt.figure(figsize=(16,10))
plt.plot(auc_table.mean(1), marker='', linewidth = 2, linestyle='--', label = 'Train Loss')
plt.legend(fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Test AUC', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
#plt.savefig('fig.png')

plt.figure(figsize=(16,10))
plt.plot(loss_table.mean(1), marker='', linewidth = 2, linestyle='--', label = 'Train Loss')
plt.legend(fontsize=20)
plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Loss', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)


#%% predict for the test part
roc_auc_score(inputs[:, 0].numpy(), network(inputs[:, 1:74]).clone().detach().numpy())

test['conv_prob'] = network(testset).clone().detach().numpy()

test[['policy_id', 'conv_prob']].to_csv('test_result_tao.csv', index = False)
