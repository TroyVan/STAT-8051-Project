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
import xgbfir

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

#%% find interaction term

X = train.iloc[:, 1:77]
y = train.iloc[:, 0]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 8051)

#eval_set = [(X_test, y_test)]

X_train = xgb.DMatrix(data = X, label = y)

param = {
        'max_depth':5,
        'eta':0.0275,
        'objective':'binary:logistic',
        'n_jobs': 4,
        'subsample': 0.8,
        'colsample_bytree': .85,
        'min_child_weight': 31,
        'max_delta_step': 8,
        'num_parallel_tree': 7
        }

cv = xgb.cv(param, X_train,
       num_boost_round = 1000,
       nfold = 4,
       metrics = {'auc'},
       seed = 8051,
       early_stopping_rounds = 200,
       verbose_eval  = True)

model = XGBClassifier(
            max_depth = 8,
            eta = 0.02279751,
            objective = 'binary:logistic',
            n_jobs = 4,
            subsample = 0.7133214,
            colsample_bytree = 0.7673029,
            min_child_weight = 31,
            max_delta_step = 8,
            n_estimators = 260
        )

fit = model.fit(X, y)

xgbfir.saveXgbFI(fit, feature_names = list(X.columns), OutputXlsxFile = 'interaction.xlsx')

#param_list  = {
#        'max_depth': range(1, 5),
##        'learning_rate': [0.1, 0.05, 0.01, 0.005, 0.001]
#        }
#
#search = GridSearchCV(model, param_list, 'roc_auc', cv = 3, iid = False)
#
#search.fit(X, y)
