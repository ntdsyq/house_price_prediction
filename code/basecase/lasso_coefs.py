# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 11:41:21 2019

@author: yanqi
"""

from sklearn.linear_model import Lasso
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

# standardize x_train data
scaler = StandardScaler()
X = scaler.fit_transform(X)


lasso = Lasso(max_iter = 10000, alpha =0.004498432668969444)
lasso.fit(X, y)
cols = Xd_train.columns 
coefs = sorted(list(zip(cols,lasso.coef_)),key=lambda t: abs(t[1]), reverse = True)
coefs = pd.DataFrame( coefs, columns = ['Feature', 'Coef'] )
coefs.to_csv("bestlasso_coefs.csv")


