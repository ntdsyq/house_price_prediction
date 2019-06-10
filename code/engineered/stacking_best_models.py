# -*- coding: utf-8 -*-
"""
Created on Sun May  5 14:44:00 2019

@author: yanqi
"""
import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 3 - Machine Learning\\Housing Price Prediction\\house_price_prediction\\code\\change_02'
os.chdir(proj_path)

from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.base import clone
from xgboost.sklearn import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from helper import load_traindata, plot_cv_traintestscores_log, make_prediction_dummy
from sklearn import tree
from sklearn import ensemble
import pickle

# initiate base models with optimized hyperparameters
lasso = Lasso(max_iter = 10000, alpha =  0.00047148663634573947)
ridge = Ridge(max_iter = 10000, alpha = 10.0)
rf = ensemble.RandomForestRegressor(n_estimators = 250, max_depth = 14, max_features = 16, min_samples_split = 3, min_samples_leaf = 1)
xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.65, gamma=0, importance_type='total_gain',
       learning_rate=0.01, max_delta_step=0, max_depth=5,
       min_child_weight=3, missing=None, n_estimators=1996, n_jobs=-1,
       nthread=None, objective='reg:linear', random_state=0, reg_alpha=0,
       reg_lambda=0.9, scale_pos_weight=1, seed=27, silent=True,
       subsample=0.55)

# load training data
X, y = load_traindata()
cols = X.columns 

X_le, tmp = load_traindata(encodetype ='le')
cols_le = X_le.columns

scaler1 = RobustScaler()
X = scaler1.fit_transform(X)
scaler2 = RobustScaler()
X_le = scaler2.fit_transform(X_le)

# create 5-fold CV scheme to fit base models, and make out-of-fold predictions
cv = KFold(n_splits=5, shuffle=True, random_state = 42)  
oof_pred = np.zeros((X.shape[0], 4))

for train_idx, test_idx in cv.split(X):
    #print(test_idx)
    X_tr, X_te = X[train_idx], X[test_idx]
    X_le_tr, X_le_te = X_le[train_idx], X_le[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    for idx, mdl in enumerate([lasso, ridge, rf, xgb]):
        mdl_instance = clone(mdl)
        if idx < 2:
            mdl_instance.fit(X_tr,y_tr)
            oof_pred[test_idx, idx] = mdl_instance.predict(X_te)
        else:
            mdl_instance.fit(X_le_tr, y_tr)
            oof_pred[test_idx, idx] = mdl_instance.predict(X_le_te)
        
# train meta-model using out of fold predictions as new features
meta_ols = LinearRegression() 
meta_ols.fit(oof_pred, y)

# load test dataset
with open('test.pickle','rb') as f:
    temp, Xd_test, test_IDs, X_le_test = pickle.load(f)

Xd_test = scaler1.transform(Xd_test)
X_le_test= scaler2.transform(X_le_test)

# train all base models on the entire training data
# use all base models to make predictions on the test data as meta-features
test_meta_features = np.zeros((Xd_test.shape[0], 4))
for idx, mdl in enumerate([lasso, ridge, rf, xgb]):
    if idx < 2:
        mdl.fit(X,y)
        test_meta_features[:, idx] = mdl.predict(Xd_test)
    else:
        mdl.fit(X_le, y)
        test_meta_features[:, idx] = mdl.predict(X_le_test)

# use meta-model trained earlier to make predictions using the meta_features on the test set
y_test_pred = np.expm1(meta_ols.predict(test_meta_features))
y_test_pred = pd.DataFrame(y_test_pred.reshape(-1,1))
submitdata = pd.concat( [test_IDs, y_test_pred], axis = 1 )
submitdata.columns = ['Id','SalePrice']
submitdata.to_csv("yq_submission36_stacking.csv",index = False)
