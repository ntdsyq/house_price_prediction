# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:54:22 2019

@author: yanqi
"""
import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 3 - Machine Learning\\Housing Price Prediction\\house_price_prediction\\code\\change_02'
os.chdir(proj_path)

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from helper import plot_cv_traintestscores_log, make_prediction_dummy, load_traindata

X, y = load_traindata()
cols = X.columns 

# standardize x_train data
scaler = RobustScaler()
X = scaler.fit_transform(X)

n_folds_i = 5
n_folds_o = 5
rs = 42
inner_cv = KFold(n_splits=n_folds_i, shuffle=True, random_state = rs )   
outer_cv = KFold(n_splits=n_folds_o, shuffle=True, random_state = rs)  

ridge = Ridge(max_iter = 10000)
alphas = np.logspace(0, 3, 100)
tuned_parameters = [{'alpha': alphas}]

best_params = []

# scores with the best model in each outer CV
best_train_score = []
best_test_score = []
best_val_score = []
trainscores = []
testscores = []
i = 0

for train_idx, test_idx in outer_cv.split(X):
    #print(test_idx)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    tune_ridge = GridSearchCV(ridge, tuned_parameters, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    tune_ridge.fit(X_tr, y_tr)
    best_params.append(tune_ridge.best_params_['alpha'])
    best_model = tune_ridge.best_estimator_
    best_val_score.append( np.sqrt(mean_squared_error(y_te, best_model.predict(X_te))) )
    best_train_score.append( np.sqrt( - tune_ridge.cv_results_['mean_train_score'][tune_ridge.best_index_]) )
    best_test_score.append( np.sqrt( - tune_ridge.cv_results_['mean_test_score'][tune_ridge.best_index_]) )
    trainscores.append(  np.sqrt(- tune_ridge.cv_results_['mean_train_score']) )
    testscores.append(  np.sqrt(- tune_ridge.cv_results_['mean_test_score']) )
    #print("fold ",i, " best model is ", best_model)
    #print("validation score on best model is ", best_val_score[i])
    i += 1
    
plt.plot(np.arange(n_folds_o), best_train_score, 'go--', label = 'train score')
plt.plot(np.arange(n_folds_o), best_test_score, 'r+--', label = 'test score')
plt.plot(np.arange(n_folds_o), best_val_score, 'bs--', label = 'validation score (outer CV)')
plt.legend(loc = "upper right")
plt.xlabel('Outer CV fold')
plt.ylabel('RMSE from best model')
print(best_params)

u_train, u_test, u_val = np.average(best_train_score), np.average(best_test_score), np.average(best_val_score)
std_train, std_test, std_val = np.std(best_train_score), np.std(best_test_score), np.std(best_val_score)
print(u_train, std_train)
print(u_test, std_test)
print(u_val, std_val)

for i in np.arange(n_folds_o):
    plot_cv_traintestscores_log(trainscores[i], testscores[i], alphas)

# re-train on entire data to get final model: 
ridge.set_params(alpha = 10.0)
ridge.fit(X, y)
submit_num = 31
coefs = sorted(list(zip(cols,ridge.coef_)),key=lambda t: abs(t[1]), reverse = True)
coefs = pd.DataFrame( coefs, columns = ['Feature', 'Coef'] )
coefs.to_csv("bestridge_coefs_submit"+str(submit_num)+".csv",header=True)

# make prediction file for Kaggle submission
submissiondata = make_prediction_dummy(ridge, scaler)
submissiondata.to_csv("yq_submission"+str(submit_num)+"_ridge.csv",index = False)








