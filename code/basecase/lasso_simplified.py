# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:17:34 2019

@author: yanqi
"""
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import  KFold, cross_val_score
from sklearn.pipeline import make_pipeline
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pickle
from helper import proj_path, plot_cv_traintestscores, make_prediction

os.chdir(proj_path)

with open('train.pickle', 'rb') as f:
    temp, Xd_train, y_train = pickle.load(f)
X = Xd_train

# loading it as y_train gives a panda series, and causes error in indexing by CV split
# convert to numpy array solves it
y = y_train.values  

alphas = np.logspace(-4, -1, 50)
rs = 1

kfolds = KFold(n_splits=5, shuffle=True, random_state = rs)  
lassocv =  LassoCV(max_iter=1e4, alphas=alphas, random_state=rs, cv=kfolds)
lasso = make_pipeline(StandardScaler(), lassocv)
rmse = np.sqrt(-cross_val_score(lasso, X, y, scoring="neg_mean_squared_error", cv=kfolds))
rmse

ypred1 = lasso.fit(X, y).predict(X)
np.sqrt(mean_squared_error(y, ypred1))
bestmodel = Lasso(max_iter=1e4, alpha = lassocv.alpha_, random_state = rs)
ypred2 = bestmodel.fit(X,y).predict(X)
np.sqrt(mean_squared_error(y, ypred2))

# how to extract the actual lasso model (and best alpha from this)?

