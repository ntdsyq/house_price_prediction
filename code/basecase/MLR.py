# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 14:49:37 2019

@author: yanqi
"""

import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 3 - Machine Learning\\Housing Price Prediction\\house_price_prediction\\code\\basecase'
os.chdir(proj_path)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.metrics import mean_squared_error
from helper import load_traindata, make_prediction_dummy, root_mean_squared_error

X, y = load_traindata()
cols = X.columns 

n_folds_o = 5
rs = 42

# standardize x_train data
scaler = RobustScaler()
X = scaler.fit_transform(X)
outer_cv = KFold(n_splits=n_folds_o, shuffle=True)  

ols = LinearRegression()
train_scores= []
test_scores = []
y_te_pred = []
y_te_true = []
X_te_cv = []
models = []
for train_idx, test_idx in outer_cv.split(X):
    #print(test_idx)
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    ols.fit(X_tr, y_tr)
    train_scores.append( root_mean_squared_error(ols, X_tr, y_tr) )
    test_scores.append( root_mean_squared_error(ols, X_te, y_te) )
    y_te_true.append( y_te )
    y_te_pred.append( ols.predict(X_te) )
    models.append( ols )
    X_te_cv.append( X_te )
    
plt.plot(np.arange(n_folds_o), test_scores, 'r+--')
plt.xlabel('ith fold in 5 fold CV')
plt.ylabel('RMSE on holdout test data')

for idx in np.arange(5):
    X_mat = np.matrix(X_te_cv[idx])
    print(np.linalg.matrix_rank(X_mat.T*X_mat))

idx = np.argmax(test_scores)
y_pred_chk = y_te_pred[idx]
y_te_chk = y_te_true[idx]
ydiff_chk = abs(y_te_chk - y_pred_chk)
X_te_chk = X_te_cv[idx]
srow = X_te_chk.sum(axis=1)
scol = X_te_chk.sum(axis = 0)
np.sum(scol == 0)
np.sum(srow == 0)

X_te_chk = X_te_cv[3]
srow = X_te_chk.sum(axis=1)
scol = X_te_chk.sum(axis = 0)
np.sum(scol == 0)
np.sum(srow == 0)


idx1 = np.argwhere(ydiff_chk > 1)[0][0]
print(y_te_chk[idx1])
print(y_pred_chk[idx1])
ols_chk = models[idx]
coefs = sorted(list(zip(cols,ols_chk.coef_)),key=lambda t: abs(t[1]), reverse = True)
coefs = pd.DataFrame( coefs, columns = ['Feature', 'Coef'] )
coefs.to_csv("ols_chk_coefs.csv",header=True)

ols.fit(X,y)
print(root_mean_squared_error(ols, X, y))
submitdata = make_prediction_dummy(ols,scaler)
submitdata.to_csv("yq_submission19_MLR.csv",index = False)
coefs = sorted(list(zip(cols,ols.coef_)),key=lambda t: abs(t[1]), reverse = True)
coefs = pd.DataFrame( coefs, columns = ['Feature', 'Coef'] )
#coefs.to_csv("ols_chk_coefs.csv",header=True)


