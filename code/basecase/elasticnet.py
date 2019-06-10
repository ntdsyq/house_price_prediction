# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 15:53:00 2019

@author: yanqi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:54:22 2019

@author: yanqi
"""
import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 3 - Machine Learning\\Housing Price Prediction\\house_price_prediction\\code\\basecase'
os.chdir(proj_path)

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, KFold

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
from helper import plot_cv_traintestscores, make_prediction

with open('train.pickle', 'rb') as f:
    temp, Xd_train, y_train = pickle.load(f)
X = Xd_train

# loading it as y_train gives a panda series, and causes error in indexing by CV split
# convert to numpy array solves it
y = y_train.values  

enet = ElasticNet(max_iter = 10000)
alphas = np.logspace(-3, 3, 50)
l1_ratio = np.linspace(0.001,1.0,30)
tuned_parameters = [{'alpha': alphas, 'l1_ratio': l1_ratio }]
n_folds_i = 5
n_folds_o = 5
rs = 1

# standardize x_train data
scaler = StandardScaler()
X = scaler.fit_transform(X)

inner_cv = KFold(n_splits=n_folds_i, shuffle=True, random_state = rs )   
outer_cv = KFold(n_splits=n_folds_o, shuffle=True, random_state = rs)  


#tune_enet = GridSearchCV(enet, tuned_parameters, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
#tune_enet.fit(X,y)

# best param
# {'alpha': 0.022229964825261943, 'l1_ratio': 0.10434482758620689}

# np.sqrt(-tune_enet.best_score_)
# Out[52]: 0.11468088652724845

best_params = []

# scores with the best model in each outer CV
best_train_score = []
best_test_score = []
best_val_score = []

i = 0

for train_idx, test_idx in outer_cv.split(X):
    #print(test_idx)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    
    tune_enet = GridSearchCV(enet, tuned_parameters, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    tune_enet.fit(X_tr, y_tr)
    best_params.append(tune_enet.best_params_)
    best_model = tune_enet.best_estimator_
    best_val_score.append( np.sqrt(mean_squared_error(y_te, best_model.predict(X_te))) )
    best_train_score.append( np.sqrt( - tune_enet.cv_results_['mean_train_score'][tune_enet.best_index_]) )
    best_test_score.append( np.sqrt( - tune_enet.cv_results_['mean_test_score'][tune_enet.best_index_]) )

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

# best alphas and l1
best_alphas = []
best_l1s = []
a_lasso = []
b_ridge = []
for p in best_params:
    best_alphas.append(p['alpha'])
    best_l1s.append(p['l1_ratio'])
    a_lasso.append(p['alpha']*p['l1_ratio'])
    b_ridge.append(p['alpha'] - p['alpha']*p['l1_ratio'])

# make two submissions to compare
# model 1: alpha = 0.09102981779915217, l1 = 0.035448275862068966
# equivalent to a_lasso = 0.003226850093018222, b_ridge = 0.08780296770613395
    
# model 2: alpha = 0.005428675439323859, l1 = 0.8277586206896551,
# equivalent to a_lasso = 0.004493632893826525, b_ridge = 0.0009350425454973344
enet.set_params(alpha = best_alphas[2], l1_ratio = best_l1s[2] )
enet.fit(X, y)
cols = Xd_train.columns 
coefs = sorted(list(zip(cols,enet.coef_)),key=lambda t: abs(t[1]), reverse = True)
coefs = pd.DataFrame( coefs, columns = ['Feature', 'Coef'] )
print(len(coefs[ np.abs(coefs['Coef']) > 0 ]))
submissiondata = make_prediction(enet, scaler)
submissiondata.to_csv("yq_submission8_enet2.csv",index = False)








