# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:32:16 2019

@author: yanqi
"""

import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 3 - Machine Learning\\Housing Price Prediction\\house_price_prediction\\code\\basecase'
os.chdir(proj_path)

from sklearn.model_selection import GridSearchCV, KFold, train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error
from helper import plot_cv_traintestscores, make_prediction_le, load_traindata, plot_FI_tree, display_tree, plot_cv_testscores
from sklearn import tree
from sklearn import ensemble
import time

X, y = load_traindata(encodetype ='le')
cols = X.columns 

# standardize x_train data
scaler = RobustScaler()
X = scaler.fit_transform(X)

n_folds_i = 5
n_folds_o = 5
rs = 1
inner_cv = KFold(n_splits=n_folds_i, shuffle=True, random_state = rs )   
outer_cv = KFold(n_splits=n_folds_o, shuffle=True, random_state = rs) 

def simpleRF(X,y):
    rf_model = ensemble.RandomForestRegressor()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    rf_model.fit(X_train,y_train)
    print("R2 on training data %.2f" %rf_model.score(X_train,y_train))
    print("RMSE on training data %2.3f" %np.sqrt(mean_squared_error(y_train,rf_model.predict(X_train))))
    print("R2 on test data %.2f" %rf_model.score(X_test,y_test))
    print("RMSE on test data %2.3f" %np.sqrt(mean_squared_error(y_test,rf_model.predict(X_test))))
    
    FI = plot_FI_tree(rf_model, cols, topn = 20)
    FI[0:20]
    return rf_model

def tune_Nestimator(X,y):
    """
    Tune only one parameter: the number of estimators in each single tree
    """
    # takes 47 seconds to run
    rf_model = ensemble.RandomForestRegressor(max_depth = 14)
    n_estimators = np.linspace(20, 300, 15, endpoint=True, dtype = int)
    grid_para = [{'n_estimators': n_estimators}]    
    tuneRF = GridSearchCV(rf_model, grid_para, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    t1 = time.time()
    tuneRF.fit(X,y)
    t2 = time.time()
    print(t2 - t1, " seconds\n")
    plot_cv_traintestscores(np.sqrt(-tuneRF.cv_results_['mean_train_score']), np.sqrt(-tuneRF.cv_results_['mean_test_score']), n_estimators, '# of trees')
    plot_cv_testscores(np.sqrt(-tuneRF.cv_results_['mean_test_score']), n_estimators, '# of trees')
    print("best RMSE: ",np.sqrt(-tuneRF.best_score_))
    print("best n_estimator: ",tuneRF.best_params_)
    print(tuneRF.best_estimator_)
    return tuneRF.best_estimator_

def tune_maxdepth(X,y):
    """
    Tune only one parameter: the maximum depth in each single tree
    """
    # takes 23s to run
    rf_model = ensemble.RandomForestRegressor(n_estimators = 60)
    max_depths = np.linspace(2,20,10, endpoint=True, dtype=int)
    grid_para = [{'max_depth': max_depths}]    
    tuneRF = GridSearchCV(rf_model, grid_para, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    t1 = time.time()
    tuneRF.fit(X,y)
    t2 = time.time()
    print(t2 - t1, " seconds\n")
    plot_cv_traintestscores(np.sqrt(-tuneRF.cv_results_['mean_train_score']), np.sqrt(-tuneRF.cv_results_['mean_test_score']), max_depths, 'max tree depth')
    print("best RMSE: ",np.sqrt(-tuneRF.best_score_))
    print("best n_estimator: ",tuneRF.best_params_)
    print(tuneRF.best_estimator_)
    return tuneRF.best_estimator_

def tune_min_samples_leaf(X,y):
    """
    Tune only one parameter: the min_samples_leaf
    """
    # takes 23s to run
    rf_model = ensemble.RandomForestRegressor(n_estimators = 100, max_depth = 14)
    min_samples_leaf = np.linspace(1,10,10, endpoint=True, dtype=int)
    grid_para = [{'min_samples_leaf': min_samples_leaf}]    
    tuneRF = GridSearchCV(rf_model, grid_para, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    t1 = time.time()
    tuneRF.fit(X,y)
    t2 = time.time()
    print(t2 - t1, " seconds\n")
    plot_cv_traintestscores(np.sqrt(-tuneRF.cv_results_['mean_train_score']), np.sqrt(-tuneRF.cv_results_['mean_test_score']), min_samples_leaf, 'min_samples_leaf')
    print("best RMSE: ",np.sqrt(-tuneRF.best_score_))
    print("best n_estimator: ",tuneRF.best_params_)
    print(tuneRF.best_estimator_)
    return tuneRF.best_estimator_

def tune_min_samples_split(X,y):
    """
    Tune only one parameter: the min_samples_split in individual trees
    """
    # takes 23s to run
    rf_model = ensemble.RandomForestRegressor(n_estimators = 100, max_depth = 14, min_samples_leaf = 2)
    min_samples_split = np.linspace(2,5,4, endpoint=True, dtype=int)
    grid_para = [{'min_samples_split': min_samples_split}]    
    tuneRF = GridSearchCV(rf_model, grid_para, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    t1 = time.time()
    tuneRF.fit(X,y)
    t2 = time.time()
    print(t2 - t1, " seconds\n")
    plot_cv_traintestscores(np.sqrt(-tuneRF.cv_results_['mean_train_score']), np.sqrt(-tuneRF.cv_results_['mean_test_score']), min_samples_split, 'min_samples_split')
    print("best RMSE: ",np.sqrt(-tuneRF.best_score_))
    print("best n_estimator: ",tuneRF.best_params_)
    print(tuneRF.best_estimator_)
    return tuneRF.best_estimator_

def tune_max_features(X,y):
    """
    Tune only one parameter: the max_features at each split
    """
    # takes 23s to run
    rf_model = ensemble.RandomForestRegressor(n_estimators = 100, max_depth = 14, min_samples_leaf = 1, min_samples_split = 2)
    max_features = np.linspace(24, 44, 11, endpoint=True, dtype = int)
    print(max_features)
    grid_para = [{'max_features': max_features}]    
    tuneRF = GridSearchCV(rf_model, grid_para, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    t1 = time.time()
    tuneRF.fit(X,y)
    t2 = time.time()
    print(t2 - t1, " seconds\n")
    plot_cv_traintestscores(np.sqrt(-tuneRF.cv_results_['mean_train_score']), np.sqrt(-tuneRF.cv_results_['mean_test_score']), max_features, 'max_features')
    plot_cv_testscores(np.sqrt(-tuneRF.cv_results_['mean_test_score']), max_features, 'max_features')
    print("best RMSE: ",np.sqrt(-tuneRF.best_score_))
    print("best n_estimator: ",tuneRF.best_params_)
    print(tuneRF.best_estimator_)
    return tuneRF.best_estimator_

def tune_Nestimators_maxdepth(X,y):
    """
    Tune n_estimators and max_depth in a gridsearch
    """
    # 415s runtime
    rf_model = ensemble.RandomForestRegressor()
    n_estimators = np.linspace(10, 100, 10, endpoint=True, dtype = int)
    max_depths = np.linspace(2,20,10, endpoint=True, dtype=int)
    grid_para = [{'n_estimators': n_estimators, 'max_depth': max_depths}]    
    tuneRF = GridSearchCV(rf_model, grid_para, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    t1 = time.time()
    tuneRF.fit(X,y)
    t2 = time.time()
    print(t2 - t1, " seconds\n")
    #plot_cv_traintestscores(np.sqrt(-tuneRF.cv_results_['mean_train_score']), np.sqrt(-tuneRF.cv_results_['mean_test_score']), n_estimators, '# of trees')
    print("best RMSE: ",np.sqrt(-tuneRF.best_score_))
    print("best n_estimator: ",tuneRF.best_params_)
    print(tuneRF.best_estimator_)
    
    # plotting test RMSE as a function of max_depth and n_estimator
    test_rmse = np.sqrt(-tuneRF.cv_results_['mean_test_score']).reshape((10,10))
    plt.figure().set_size_inches(8, 6)
    for ridx, depth in enumerate(max_depths):
        if ridx >= 3:                   
            plt.plot(n_estimators, test_rmse[ridx,:], label = "max_depth: " + str(depth))
    plt.xlabel('# of trees')
    plt.ylabel('Mean RMSE on test set (5-fold CV)')
    plt.legend(loc = 'upper left')
    
    return tuneRF


#RF0 = simpleRF(X,y)
RF1 = tune_Nestimator(X,y)
FI = plot_FI_tree(RF1, cols, topn = 20)
FI[0:20]

RF2 = tune_maxdepth(X,y)
FI = plot_FI_tree(RF2, cols, topn = 20)
FI[0:20]

tuneRF3 = tune_Nestimators_maxdepth(X,y)
FI = plot_FI_tree(tuneRF3.best_estimator_, cols, topn = 20)
FI[0:20]

RF3 = tune_min_samples_leaf(X,y)
RF4 = tune_min_samples_split(X,y)
RF5 = tune_max_features(X,y)

# make predictions with best model, most complex model, simplest model
best_RF1 = ensemble.RandomForestRegressor(n_estimators = 100, max_depth = 14, max_features = 26)
best_RF1.fit(X,y)
outdata1 = make_prediction_le(best_RF1, scaler)
outdata1.to_csv("yq_submission16_RF.csv",index = False)

best_RF2 = ensemble.RandomForestRegressor(n_estimators = 160, max_depth = 18)
best_RF2.fit(X,y)
outdata2 = make_prediction_le(best_RF2, scaler)
outdata2.to_csv("yq_submission17_RF.csv",index = False)

best_RF3 = ensemble.RandomForestRegressor(n_estimators = 60, max_depth = 12)
best_RF3.fit(X,y)
outdata3 = make_prediction_le(best_RF3, scaler)
outdata3.to_csv("yq_submission18_RF.csv",index = False)
              



