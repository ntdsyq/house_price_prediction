# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 17:59:38 2019

@author: yanqi
"""

# try the automatic LassoCV method
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_cv_testscores(model):
    scores = model.cv_results_['mean_test_score']
    scores_std = model.cv_results_['std_test_score']
    plt.figure().set_size_inches(8, 6)
    plt.semilogx(alphas, scores)
    
    std_error = scores_std / np.sqrt(n_folds)
    
    plt.semilogx(alphas, scores + std_error, 'b--')
    plt.semilogx(alphas, scores - std_error, 'b--')
    
    # alpha=0.2 controls the translucency of the fill color
    plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)
    
    plt.ylabel('CV score +/- std error')
    plt.xlabel('alpha')
    plt.axhline(np.max(scores), linestyle='--', color='.5')
    plt.xlim([alphas[0], alphas[-1]])
    #plt.ylim(0.8,1)
    
def plot_cv_traintestscores(model):
    testscores = model.cv_results_['mean_test_score']
    trainscores = model.cv_results_['mean_train_score']
    plt.figure().set_size_inches(8, 6)
    plt.semilogx(alphas, testscores)
    plt.semilogx(alphas, trainscores, 'b--')
    #plt.ylim(0.8,1)
    
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Python Machine Learning\\Housing Price Prediction\\house_price_prediction\\code'
os.chdir(proj_path)
infname = "../results/s2_clean_dummified.csv"
df = pd.read_csv(infname)

lasso = Lasso(normalize = True, max_iter = 10000)
alphas = np.logspace(-7, -2, 50)
tuned_parameters = [{'alpha': alphas}]
n_folds = 5

X = df.copy()
X.drop(['LogSalePrice','Id'], axis = 1, inplace = True)
Y = df['LogSalePrice']

cv = KFold(n_splits=n_folds, shuffle=True)
tune_lasso = GridSearchCV(lasso, tuned_parameters, cv=cv, refit=True, return_train_score = True, scoring = 'neg_mean_squared_error')
tune_lasso.fit(X, Y)

plot_cv_testscores(tune_lasso)
plot_cv_traintestscores(tune_lasso)
print(tune_lasso.best_params_)
print(np.max(tune_lasso.cv_results_['mean_test_score']))
print(np.max(tune_lasso.cv_results_['mean_train_score']))