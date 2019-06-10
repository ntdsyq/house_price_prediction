# -*- coding: utf-8 -*-
"""
Created on Thu May  2 15:10:26 2019

@author: yanqi
"""

import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 3 - Machine Learning\\Housing Price Prediction\\house_price_prediction\\code\\change_02'
os.chdir(proj_path)

import pandas as pd
import numpy as np

# simple average of models
y_lasso = pd.read_csv("yq_submission30_lasso.csv",names = ['Id','price_lasso'],header=0,index_col=0)
y_ridge = pd.read_csv("yq_submission31_ridge.csv",names = ['Id','price_ridge'],header=0,index_col=0)
y_xgb = pd.read_csv("yq_submission33_xgb_tuned.csv",names = ['Id','price_xgb'],header=0,index_col=0)
y_rf = pd.read_csv("yq_submission34_RF_tuned.csv", names = ['Id','price_rf'],header=0,index_col=0)
y_df = pd.concat([y_lasso,y_ridge,y_xgb,y_rf], axis = 1)

# check correlation between the predictions
y_df.corr()

# convert price back to log-price, the original model prediction output
logy = y_df.apply(lambda x: np.log1p(x))

# averaging model predictions with two sets of weights
logy_avg1 = logy['price_lasso']*0.4 + logy['price_ridge']*0.3 + logy['price_xgb']*0.2+logy['price_rf']*0.1
logy_avg2 = logy['price_lasso']*0.45 + logy['price_ridge']*0.35 + logy['price_xgb']*0.15 +logy['price_rf']*0.05

# convert log price back to price, for submission
y_avg1 = logy_avg1.apply(lambda x: np.expm1(x)).reset_index()
y_avg1.columns = ['Id','SalePrice']
y_avg2 = logy_avg2.apply(lambda x: np.expm1(x)).reset_index()
y_avg2.columns = ['Id','SalePrice']
y_avg1.to_csv("yq_submission35_ModelAvg_1.csv",index=False)
y_avg2.to_csv("yq_submission35_ModelAvg_2.csv",index=False)  # best performing model overall

