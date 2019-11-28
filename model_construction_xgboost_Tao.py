# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 08:25:11 2019

@author: Tao Tao
"""

#%% load data and packages

from xgboost import XGBClassifier
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

#set default folder
import os
os.chdir(r'C:\UMN\Course\2019_fall\STAT 8051\FP\STAT-8051-Project')

train = pd.read_csv('train.csv', delimiter=",")
test = pd.read_csv('test.csv', delimiter=",")

# check the missing values
train.isnull().sum()
test.isnull().sum()

# drop missing values for train dataset
train = train.fillna(-1)
test = test.fillna(-1)

#%% variable selection



#%% xgboost classification
X = train.iloc[:, 1:87]
y = train.iloc[:, 0]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 8051)

#eval_set = [(X_test, y_test)]

X_train = xgb.DMatrix(data = X, label = y)

param = {
        'max_depth':3,
        'eta':0.1,
        'silent':1,
        'objective':'binary:logistic',
        'n_jobs': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.6
        }

cv = xgb.cv(param, X_train,
       num_boost_round = 2000,
       nfold = 3,
       metrics = {'auc'},
       seed = 8051,
       early_stopping_rounds = 500,
       verbose_eval  = False)


param_list  = {
        'max_depth': range(1, 5),
#        'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001]
        }

search = GridSearchCV(model, param_list, 'roc_auc', cv = 3, iid = False)

search.fit(X, y)
