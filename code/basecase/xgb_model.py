# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:38:42 2019

@author: yanqi
"""
import pandas as pd
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from helper import load_traindata, plot_cv_traintestscores, plot_cv_testscores, make_prediction_le_noscaling, \
print_grid_scores, root_mean_squared_error, plot_FI_tree

X, y = load_traindata( encodetype ='le')
cols = X.columns 

## standardize x_train data, tree-based models don't need scaling
#scaler = RobustScaler()
#X = scaler.fit_transform(X)
n_folds_o = 5
rs = 1
outer_cv = KFold(n_splits=n_folds_o, shuffle=True, random_state = rs)  

xgtrain = xgb.DMatrix(data=X, label = y)

# default importance_type='gain' (vs. 'total_gain' across all splits using the feature)

def xgbmodelfit(alg, datamatrix, cvfolds ):
    xgb_param = alg.get_xgb_params()
    cvresult = xgb.cv(xgb_param, datamatrix, num_boost_round=alg.get_params()['n_estimators'], folds = cvfolds,
    metrics='rmse', early_stopping_rounds=50)
    alg.set_params(n_estimators=cvresult.shape[0])
    alg.fit(X, y)
    rmse = root_mean_squared_error(alg, X, y)
    n = cvresult.shape[0]
    print("optimal n_estimator is %d" %n)
    print("With optimal n_estimator, mean CV test RMSE is %.4f" %cvresult['test-rmse-mean'][n-1])
    print("With optimal n_estimator, mean CV train RMSE is %.4f" %cvresult['train-rmse-mean'][n-1])
    print("RMSE of xgb entire data is %.4f" %(rmse))
    plot_cv_traintestscores(cvresult['train-rmse-mean'], cvresult['test-rmse-mean'], [i for i in range(n)], 'n_estimators')
    plot_cv_traintestscores(cvresult['train-rmse-mean'][50:150], cvresult['test-rmse-mean'][50:150], [i for i in range(n)][50:150], 'n_estimators')
    #plot_cv_traintestscores(cvresult['train-rmse-mean'][50:], cvresult['test-rmse-mean'][50:], [i for i in range(n)][50:], 'n_estimators')
    plot_cv_testscores(cvresult['test-rmse-mean'][50:], [i for i in range(n)][50:], 'n_estimators')
    
    feat_imp = plot_FI_tree(alg, cols, 20)
    feat_imp[0:20]
    return alg

# step 1: use large learning rate, tune n_estimators
xgb1 = XGBRegressor( 
        learning_rate =0.1, n_estimators=1000, 
        max_depth=5, min_child_weight=1,
        gamma=0, subsample=0.8, colsample_bytree=0.8, 
        objective= 'reg:linear', n_jobs = -1, 
        seed=27, importance_type = 'total_gain')

xgbmodelfit(xgb1, xgtrain, outer_cv)
#submissiondata = make_prediction_le_noscaling(alg)
#submissiondata.to_csv("yq_submission19_xgb.csv",index = False)

# step 2: tune max_depth and min_child_weight
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}

gs1 = GridSearchCV(xgb1, param_grid = param_test1, scoring='neg_mean_squared_error',n_jobs= -1, iid=False, cv= outer_cv)
gs1.fit(X,y)
gs1.grid_scores_, gs1.best_params_, gs1.best_score_
print_grid_scores(gs1)
print('Best parameters: %r' %gs1.best_params_)
print('Best mean test RMSE:' %(np.sqrt(-gs1.best_score_)) )

# step 2.2: fine tune max_depth, min_child_weight
param_test2 = {
 'max_depth':[2,3,4],
 'min_child_weight': range(4,7,1)
}

gs2 = GridSearchCV(xgb1, param_grid = param_test2, scoring='neg_mean_squared_error',n_jobs= -1, iid=False, cv= outer_cv)
gs2.fit(X,y)
print_grid_scores(gs2)
print('Best parameters: %r' %gs2.best_params_)
print('Best mean test RMSE: %.4f' %(np.sqrt(-gs2.best_score_)) )

xgb1.set_params(max_depth = 3, min_child_weight = 4)

# step 3: tune gamma
#param_test3 = {
# 'gamma':[i/10.0 for i in range(0,5)]
#}

#param_test3 = {
# 'gamma':[i/100.0 for i in range(0,10)]
#}

param_test3 = {
 'gamma':range(6)
}

gs3 = GridSearchCV(xgb1, param_grid = param_test3, scoring='neg_mean_squared_error',n_jobs= -1, iid=False, cv= outer_cv)
gs3.fit(X,y)
print_grid_scores(gs3)
print('Best parameters: %r' %gs3.best_params_)
print('Best mean test RMSE: %.4f' %(np.sqrt(-gs3.best_score_)) )

# step 4: re-calibrate n_estimators given the tuned parameters
xgb1.set_params(n_estimators = 1000)
xgb1
xgbmodelfit(xgb1, xgtrain, outer_cv)

# step 5.1: coarse tune subsample and colsample_bytree
param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gs4 = GridSearchCV(xgb1, param_grid = param_test4, scoring='neg_mean_squared_error',n_jobs= -1, iid=False, cv= outer_cv)
gs4.fit(X,y)
print_grid_scores(gs4)
print('Best parameters: %r' %gs4.best_params_)
print('Best mean test RMSE: %.4f' %(np.sqrt(-gs4.best_score_)) )

# # step 5.2: fine tune subsample and colsample_bytree
param_test5 = {
 'subsample':[i/100.0 for i in range(75,90,5)],
 'colsample_bytree':[i/100.0 for i in range(75,90,5)]
}

gs5 = GridSearchCV(xgb1, param_grid = param_test5, scoring='neg_mean_squared_error',n_jobs= -1, iid=False, cv= outer_cv)
gs5.fit(X,y)
print_grid_scores(gs5)
print('Best parameters: %r' %gs5.best_params_)
print('Best mean test RMSE: %.4f' %(np.sqrt(-gs5.best_score_)) )

# step 6: tune L1 regularization parameter alpha
param_test6 = {
 'reg_alpha':[1e-5, 1e-4, 1e-3]
}

gs6 = GridSearchCV(xgb1, param_grid = param_test6, scoring='neg_mean_squared_error',n_jobs= -1, iid=False, cv= outer_cv)
gs6.fit(X,y)
print_grid_scores(gs6)
print('Best parameters: %r' %gs6.best_params_)
print('Best mean test RMSE: %.4f' %(np.sqrt(-gs6.best_score_)) )

# step 7: tune L2 regularization parameter lambda
param_test7 = {
 'reg_lambda': [i/10.0 for i in range(16)]
}

gs7 = GridSearchCV(xgb1, param_grid = param_test7, scoring='neg_mean_squared_error',n_jobs= -1, iid=False, cv= outer_cv)
gs7.fit(X,y)
print_grid_scores(gs7)
print('Best parameters: %r' %gs7.best_params_)
print('Best mean test RMSE: %.4f' %(np.sqrt(-gs7.best_score_)) )

# step 8: reduce learning rate, re-tune n_estimators
xgb1.set_params(learning_rate = 0.01, n_estimators = 5000)
xgbmodelfit(xgb1, xgtrain, outer_cv)

submissiondata = make_prediction_le_noscaling(xgb1)
submissiondata.to_csv("yq_submission20_xgb_tuned.csv",index = False)