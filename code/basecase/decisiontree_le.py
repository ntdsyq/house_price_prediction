# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:14:36 2019

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
from helper import plot_cv_traintestscores, make_prediction_dummy, load_traindata
from sklearn import tree
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

def simpleDT(X,y):
    tree_model = tree.DecisionTreeRegressor()
    tree_model.fit(X,y)
    R2 = tree_model.score(X,y)
    ypred = tree_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y,ypred))
    print("R^2 is %.2f and rmse is %2.3f" %(R2,rmse))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    tree_model.fit(X_train,y_train)
    print("R2 on training data %.2f" %tree_model.score(X_train,y_train))
    print("RMSE on training data %2.3f" %np.sqrt(mean_squared_error(y_train,tree_model.predict(X_train))))
    print("R2 on test data %.2f" %tree_model.score(X_test,y_test))
    print("RMSE on test data %2.3f" %np.sqrt(mean_squared_error(y_test,tree_model.predict(X_test))))
    
    FI = list(zip(cols,tree_model.feature_importances_))
    FI = sorted(FI, key = lambda t: t[1], reverse = True)
    print(FI[0:20])
    print("max_depth: ", tree_model.tree_.max_depth)
    
    # how to unveil decision tree structure
    # https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html#sphx-glr-auto-examples-tree-plot-unveil-tree-structure-py
    
    return tree_model

def tune_max_depth(X,y):
    tree_model = tree.DecisionTreeRegressor()
    max_depths = np.linspace(1, 25, 25, endpoint=True)
    grid_para = [{'max_depth': max_depths}]    
    tuneDT1 = GridSearchCV(tree_model, grid_para, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    t1 = time.time()
    tuneDT1.fit(X,y)
    t2 = time.time()
    print(t2 - t1, " seconds\n")
    plot_cv_traintestscores(np.sqrt(-tuneDT1.cv_results_['mean_train_score']), np.sqrt(-tuneDT1.cv_results_['mean_test_score']), max_depths, 'max_depth')
    print("best RMSE: ",np.sqrt(-tuneDT1.best_score_))
    print("best max_depth: ",tuneDT1.best_params_)
    print(tuneDT1.best_estimator_)
    return tuneDT1.best_estimator_

def tune_leaf(X,y):
    tree_model = tree.DecisionTreeRegressor()
    min_samples_leaf = np.linspace(1, 20, 20, endpoint=True, dtype = int )
    grid_para = [{'min_samples_leaf': min_samples_leaf}]    
    tuneDT2 = GridSearchCV(tree_model, grid_para, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    t1 = time.time()
    tuneDT2.fit(X,y)
    t2 = time.time()
    print(t2 - t1, " seconds\n")
    plot_cv_traintestscores(np.sqrt(-tuneDT2.cv_results_['mean_train_score']), np.sqrt(-tuneDT2.cv_results_['mean_test_score']), min_samples_leaf, 'min_samples_leaf')
    print("best RMSE: ", np.sqrt(-tuneDT2.best_score_))
    print("best min_samples_leaf: ", tuneDT2.best_params_)
    print("max_depth in best tree: ", tuneDT2.best_estimator_.tree_.max_depth)
    print(tuneDT2.best_estimator_)
    return tuneDT2.best_estimator_

def tune_split(X,y):
    tree_model = tree.DecisionTreeRegressor()
    min_samples_split = np.linspace(2, 60, 30, endpoint=True, dtype = int )
    grid_para = [{'min_samples_split': min_samples_split}]    
    tuneDT2 = GridSearchCV(tree_model, grid_para, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    t1 = time.time()
    tuneDT2.fit(X,y)
    t2 = time.time()
    print(t2 - t1, " seconds\n")
    plot_cv_traintestscores(np.sqrt(-tuneDT2.cv_results_['mean_train_score']), np.sqrt(-tuneDT2.cv_results_['mean_test_score']), min_samples_split, 'min_samples_split')
    print("best RMSE: ", np.sqrt(-tuneDT2.best_score_))
    print("best min_samples_split: ", tuneDT2.best_params_)
    print("max_depth in best tree: ", tuneDT2.best_estimator_.tree_.max_depth)
    print(tuneDT2.best_estimator_)
    return tuneDT2.best_estimator_

# gridsearch all parameters together 
def nonNested_cv_DT(X,y):   
    tree_model = tree.DecisionTreeRegressor()
    max_depths = np.linspace(3, 24, 8, endpoint=True)
    min_samples_split = np.linspace(10, 50, 5, endpoint=True, dtype = int )
    min_samples_leaf = np.linspace(4, 20, 5, endpoint=True, dtype = int )
    grid_para = [{
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_depth": max_depths
    }]
    
    tuneDT3 = GridSearchCV(tree_model, grid_para, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
    t1 = time.time()
    tuneDT3.fit(X,y)
    t2 = time.time()
    print(t2 - t1, " seconds\n")
    print("best RMSE: ", np.sqrt(-tuneDT3.best_score_))
    print("best min_samples_split: ", tuneDT3.best_params_)
    print("max_depth in best tree: ", tuneDT3.best_estimator_.tree_.max_depth)
    print(tuneDT3.best_estimator_)
    return tuneDT3.best_estimator_

def nested_CV_DT(X,y):
    # nestedCV
    best_params = []
    # scores with the best model in each outer CV
    best_train_score = []
    best_test_score = []
    best_val_score = []
    trainscores = []
    testscores = []
    i = 0
    
    tree_model = tree.DecisionTreeRegressor()
    max_depths = np.linspace(3, 24, 8, endpoint=True)
    min_samples_split = np.linspace(10, 50, 5, endpoint=True, dtype = int )
    min_samples_leaf = np.linspace(4, 20, 5, endpoint=True, dtype = int )
    grid_para = [{
        "min_samples_leaf": min_samples_leaf,
        "min_samples_split": min_samples_split,
        "max_depth": max_depths
    }]
    
    for train_idx, test_idx in outer_cv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        
        tuneDT4 = GridSearchCV(tree_model, grid_para, cv=inner_cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
        tuneDT4.fit(X_tr,y_tr)
        best_params.append(tuneDT4.best_params_)
        best_model = tuneDT4.best_estimator_
        best_val_score.append( np.sqrt(mean_squared_error(y_te, best_model.predict(X_te))) )
        best_train_score.append( np.sqrt( - tuneDT4.cv_results_['mean_train_score'][tuneDT4.best_index_]) )
        best_test_score.append( np.sqrt( - tuneDT4.cv_results_['mean_test_score'][tuneDT4.best_index_]) )
        trainscores.append(  np.sqrt(- tuneDT4.cv_results_['mean_train_score']) )
        testscores.append(  np.sqrt(- tuneDT4.cv_results_['mean_test_score']) )
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

tree1 = simpleDT(X,y)
tree2 = tune_max_depth(X,y)
tree3 = tune_leaf(X,y)
tree4 = tune_split(X,y)
tree5 = nonNested_cv_DT(X,y)
tree6 = nested_CV_DT(X,y)
