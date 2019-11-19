# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:25:11 2019

@author: Tao Tao
"""

#%% load data and packages

from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_auc_score
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

#%% variable selection



#%% xgboost classification
auc_list = []
avg_auc_list  =[]

for tree_depth in range(1, 10):

    for i in range(3):
    
        X = train[train['cv_index'] != i+1].iloc[:, 1:74]
        Y = train[train['cv_index'] != i+1].iloc[:, 0]
        dtrain = xgb.DMatrix(X, label = Y)
        
        X_test = train[train['cv_index'] == i+1].iloc[:, 1:74]
        Y_test = train[train['cv_index'] == i+1].iloc[:, 0]
        dtest = xgb.DMatrix(X_test, label = Y_test)
    
        clf = XGBClassifier(max_depth = tree_depth + 1, random_state = 1, learning_rate = 0.01)
        clf.fit(dtrain)
    
        y_pred = clf.predict(X_test)
        auc = roc_auc_score(Y_test, y_pred)
        auc_list.append(auc)

    avg_auc = sum(auc_list)/len(auc_list)
    avg_auc_list.append(avg_auc)
    
plt.figure(figsize=(16,10))
plt.plot(avg_auc_list, marker='', linewidth = 2, linestyle='--', label = 'AUC')
plt.legend(fontsize=20)
plt.xlabel(fontsize=20)
plt.ylabel(fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)