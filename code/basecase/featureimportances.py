# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 10:52:37 2019

@author: yanqi
"""
import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 3 - Machine Learning\\Housing Price Prediction\\house_price_prediction\\code\\basecase'
os.chdir(proj_path)

from sklearn.linear_model import Lasso
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import ensemble
from xgboost.sklearn import XGBRegressor
from helper import load_traindata, permutation_importances, oob_regression_r2_score, oob_regression_mse_score

def FI_reg():
    """
    Add a random column to the feature data, and run the optimized lasso model
    Any feature that ranks lower in importance than the random column has no significance for prediction with this model
    """
    X, y = load_traindata()
    cols = list(X.columns)
    
    lasso = Lasso(max_iter = 10000, alpha =  0.004498433)
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    # add random column 
    rndcol = np.random.randn(X.shape[0])
    X = np.column_stack((X,rndcol))
    cols.append('random')
    lasso.fit(X, y)
    
    coefs = sorted(list(zip(cols,lasso.coef_)),key=lambda t: abs(t[1]), reverse = True)
    coefs = pd.DataFrame( coefs, columns = ['Feature', 'Coef'] )
    rnd_idx = np.argwhere( coefs['Feature'] == 'random' )[0][0]
    print("random column coefficient is : %.4f, ranking %d" %(coefs.iloc[rnd_idx,1], rnd_idx) )
    # random column has zero coefficient and ranks at the bottom correctly. 
    


def FI_RF_sklearn():    
    X, y = load_traindata(encodetype ='le')
    cols = list(X.columns)
#    scaler = RobustScaler()
#    X = scaler.fit_transform(X)
    rndcol = np.random.randn(X.shape[0])
    X = np.column_stack((X,rndcol))
    cols.append('random')
    
    rf = ensemble.RandomForestRegressor(n_estimators = 100, max_depth = 14, max_features = 26, random_state = 42)
    rf.fit(X, y)
    
    imp = sorted(list(zip(cols,rf.feature_importances_)),key=lambda t: abs(t[1]), reverse = True)
    imp = pd.DataFrame( imp, columns = ['Feature', 'Importance'] )
    rnd_idx = np.argwhere( imp['Feature'] == 'random' )[0][0]
    print(imp.iloc[:rnd_idx+1,:])
    return imp

def FI_RF_permuation(metric = oob_regression_r2_score):
    X, y = load_traindata(encodetype ='le')
    rndcol = np.random.randn(X.shape[0])
    X['random'] = rndcol
    
    rf = ensemble.RandomForestRegressor(n_estimators = 100, max_depth = 14, max_features = 26, random_state = 42, oob_score=True)
    rf.fit(X, y)
    
    imp = permutation_importances(rf, X, y, oob_regression_r2_score)
    #imp = permutation_importances(rf, X, y, oob_regression_mse_score)
    imp = sorted(list(zip(X.columns,imp)),key=lambda t: abs(t[1]), reverse = True)
    imp = pd.DataFrame( imp, columns = ['Feature', 'Importance'] )
    rnd_idx = np.argwhere( imp['Feature'] == 'random' )[0][0]
    print(imp.iloc[:rnd_idx+1,:])
    return imp
    
def FI_xgb_sklearn():    
    X, y = load_traindata(encodetype ='le')
    cols = list(X.columns)

    rndcol = np.random.randn(X.shape[0])
    X = np.column_stack((X,rndcol))
    cols.append('random')
    
    xgb1 = XGBRegressor( 
            learning_rate=0.01, n_estimators=3320,
            max_depth=3, min_child_weight=4, 
            colsample_bytree=0.8, subsample=0.8, 
            importance_type='total_gain', objective='reg:linear',
            n_jobs=-1, random_state= 0, 
            seed=27, silent=True
            )

    xgb1.fit(X, y)
    
    imp = sorted(list(zip(cols,xgb1.feature_importances_)),key=lambda t: abs(t[1]), reverse = True)
    imp = pd.DataFrame( imp, columns = ['Feature', 'Importance'] )
    rnd_idx = np.argwhere( imp['Feature'] == 'random' )[0][0]
    print(imp.iloc[:rnd_idx+1,:])
    return imp

def FI_reg_blog():
    """
    Add a random column to the feature data, and run the optimized lasso model
    Any feature that ranks lower in importance than the random column has no significance for prediction with this model
    """
    X, y = load_traindata()
    cols = list(X.columns)
    
    lasso = Lasso(max_iter = 10000, alpha =  0.000308884359647748)
    scaler = RobustScaler()
    X = scaler.fit_transform(X)
    
    lasso.fit(X, y)
    
    coefs = sorted(list(zip(cols,lasso.coef_)),key=lambda t: abs(t[1]), reverse = True)
    coefs = pd.DataFrame( coefs, columns = ['Feature', 'Coef'] )
    return coefs

def FI_RF_permuation_blog():
    X, y = load_traindata(encodetype ='le')
    
    rf = ensemble.RandomForestRegressor(n_estimators = 100, max_depth = 14, max_features = 26, random_state = 42, oob_score=True)
    rf.fit(X, y)
    
    drop_in_mse = permutation_importances(rf, X, y, oob_regression_mse_score)
    imp = [-x for x in drop_in_mse]
    imp = sorted(list(zip(X.columns,imp)),key=lambda t: abs(t[1]), reverse = True)
    imp = pd.DataFrame( imp, columns = ['Feature', 'Importance'] )
    return imp

def FI_xgb_blog():    
    X, y = load_traindata(encodetype ='le')
    cols = list(X.columns)
    
    xgb1 = XGBRegressor( 
            learning_rate=0.01, n_estimators=3320,
            max_depth=3, min_child_weight=4, 
            colsample_bytree=0.8, subsample=0.8, 
            importance_type='total_gain', objective='reg:linear',
            n_jobs=-1, random_state= 0, 
            seed=27, silent=True
            )

    xgb1.fit(X, y)
    
    imp = sorted(list(zip(cols,xgb1.feature_importances_)),key=lambda t: abs(t[1]), reverse = True)
    imp = pd.DataFrame( imp, columns = ['Feature', 'Importance'] )
    return imp


# compare sklearn and permutation FIs
rf_fi1 = FI_RF_sklearn()
rf_fi1 = rf_fi1.reset_index()
rf_fi1.rename(columns={'index':'rank1','Importance':'Imp1'}, inplace=True)

rf_fi2 = FI_RF_permuation()
rf_fi2 = rf_fi2.reset_index()
rf_fi2.rename(columns={'index':'rank2','Importance':'Imp2'}, inplace=True)

rf_fi_compare = pd.merge(rf_fi2,rf_fi1, on='Feature')
rf_fi_compare = rf_fi_compare[['Feature','rank1','rank2','Imp1','Imp2']]
rf_fi_compare.to_csv("RF_imp_sklearn_permutation.csv", index = False)

xgb_fi = FI_xgb_sklearn()
xgb_fi.to_csv("xgb_imp_sklearn.csv", index = False)


# FI data for blog
lasso_FI = FI_reg_blog()
RF_FI = FI_RF_permuation_blog()
xgb_FI = FI_xgb_blog()
FI_out = pd.ExcelWriter('FI_blog.xlsx', engine='xlsxwriter')
lasso_FI.to_excel(FI_out, sheet_name = "lasso")
RF_FI.to_excel(FI_out, sheet_name = "random forest")
xgb_FI.to_excel(FI_out, sheet_name = "xgboost")
FI_out.save()

lasso_LI = lasso_FI.iloc[123:,:]
xgb_LI = xgb_FI.iloc[ 52: ,: ]
RF_LI = RF_FI.iloc[ 58:, : ]
set(xgb_LI.Feature) & set(RF_LI.Feature) & set(lasso_LI.Feature)

# FI plots
topn = 15
#plt.figure().set_size_inches(10, 8)
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
fig = plt.figure(figsize = (20,8))
ax1 = fig.add_subplot(121)
plt.barh(range(topn), lasso_FI.Coef[0:topn][::-1], tick_label=lasso_FI.Feature[0:topn][::-1])
plt.title("Top 15 Features from Best Lasso Model", fontsize = 16)
plt.xlabel("Regression Coefficient")
ax2 = fig.add_subplot(122)
plt.barh(range(topn), xgb_FI.Importance[0:topn][::-1], tick_label=xgb_FI.Feature[0:topn][::-1])
plt.xlabel("Feature Importance")
plt.title("Top 15 Features from Best xgboost Model", fontsize = 16)
plt.tight_layout()
plt.savefig("../../documentation/blog/FI-basecase.pdf")





