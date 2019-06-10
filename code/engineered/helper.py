# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:44:13 2019

@author: yanqi
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble.forest import _generate_unsampled_indices
import warnings
from sklearn.metrics import r2_score, mean_squared_error

train_file = '../../data/train.csv'
test_file = '../../data/test.csv'

def chk_mv(df):
    mv_bycol = pd.DataFrame( df.isnull().sum(axis=0), columns = ['num_mv'])
    mv_bycol['pct_mv'] = mv_bycol['num_mv']/df.shape[0]
    mv_bycol = mv_bycol.sort_values('num_mv', ascending=False)
    mv_by_col = mv_bycol[mv_bycol['num_mv'] > 0]
    print(mv_by_col)
    
def plot_cv_traintestscores_log(trainscores, testscores,  alphas, para_name = 'alpha'):
    plt.figure().set_size_inches(8, 6)
    plt.semilogx(alphas, testscores, label = 'test')
    plt.semilogx(alphas, trainscores, 'b--', label = 'train')
    plt.ylabel('RMSE on test and trainscores')
    plt.xlabel(para_name)
    plt.legend(loc = 'upper left')
    #plt.ylim(0.8,1)

def plot_cv_traintestscores(trainscores, testscores,  alphas, para_name = 'alpha'):
    plt.figure().set_size_inches(8, 6)
    plt.plot(alphas, testscores, label = 'test')
    plt.plot(alphas, trainscores, 'b--', label = 'train')
    plt.ylabel('RMSE on test and trainscores')
    plt.xlabel(para_name)
    plt.legend(loc = 'upper left')
    
def plot_cv_testscores(testscores,  alphas, para_name = 'alpha'):
    plt.figure().set_size_inches(8, 6)
    plt.plot(alphas, testscores, label = 'test')
    plt.ylabel('RMSE on test scores')
    plt.xlabel(para_name)
    
def make_prediction_dummy(model, scaler):
    """
    - function to make prediction on the kaggle test set where categorical features
    were dummified
    - Should be used with regression models
    """
    with open('test.pickle','rb') as f:
        temp, Xd_test, test_IDs, temp1 = pickle.load(f)
    
    # add the columns that are not in test set
    Xd_test = scaler.transform(Xd_test)
    y_test_pred = np.expm1(model.predict(Xd_test))
    
    y_test_pred = pd.DataFrame(y_test_pred.reshape(-1,1))
    submitdata = pd.concat( [test_IDs, y_test_pred], axis = 1 )
    submitdata.columns = ['Id','SalePrice']
    return submitdata

def make_prediction_le(model, scaler):
    """
    - function to make prediction on the kaggle test set with where categorical features
    were label encoded.
    - Should be used with tree-based models 
    """
    with open('test.pickle','rb') as f:
        temp, temp1, test_IDs, X_le_test = pickle.load(f)
    
    X_le_test = scaler.transform(X_le_test)
    y_test_pred = np.expm1(model.predict(X_le_test))
    
    y_test_pred = pd.DataFrame(y_test_pred.reshape(-1,1))
    submitdata = pd.concat( [test_IDs, y_test_pred], axis = 1 )
    submitdata.columns = ['Id','SalePrice']
    return submitdata

def make_prediction_le_noscaling(model):
    """
    - function to make prediction on the kaggle test set with where categorical features
    were label encoded.
    - Should be used with tree-based models 
    """
    with open('test.pickle','rb') as f:
        temp, temp1, test_IDs, X_le_test = pickle.load(f)

    y_test_pred = np.expm1(model.predict(X_le_test))
    
    y_test_pred = pd.DataFrame(y_test_pred.reshape(-1,1))
    submitdata = pd.concat( [test_IDs, y_test_pred], axis = 1 )
    submitdata.columns = ['Id','SalePrice']
    return submitdata

def labelencode_feature(X_train, X_test):
    # find categorical columns
    num_cols = sorted(X_train._get_numeric_data().columns)
    cat_cols = sorted(list(set(X_train.columns) - set(num_cols)))
    
    # add train/test label for splitting the data later
    X_train['dataset'] = 'train'
    X_test['dataset'] = 'test'
    
    # combine training and testing data for consistent labelencoding on the two datasets
    X = pd.concat([X_train, X_test], ignore_index = True)
    X_le = X.copy()
    
    # labelencode the categorical features
    for col in cat_cols:
        X_le[col] = LabelEncoder().fit_transform(X_le[col])
        #print("%s data type is %s" %(col, X_le[col].dtype))
        
    # split out train and test 
    X_le_train = X_le[ X_le['dataset'] == 'train' ].copy()
    X_le_test = X_le[ X_le['dataset'] == 'test' ].copy()
    X_le_train.drop('dataset', axis = 1, inplace = True)
    X_le_test.drop('dataset', axis = 1, inplace = True)
    X_train.drop('dataset', axis = 1, inplace = True)
    X_test.drop('dataset', axis = 1, inplace = True)
    print(X_le_train.shape)
    print(X_le_test.shape)
    return X_le_train, X_le_test

def load_traindata(encodetype = 'dummy'):
    with open('train.pickle', 'rb') as f:
        if encodetype == 'dummy':
            temp, X, y_train, temp1 = pickle.load(f)
        elif encodetype == 'le':
            temp, temp1, y_train, X = pickle.load(f)
        else:
            print("returning un-encoded features")
            X, temp, y_train, temp1 = pickle.load(f)
            
    # loading it as y_train gives a panda series, and causes error in indexing by CV split
    # convert to numpy array solves it
    y = y_train.values 
    return X, y

def plot_FI_tree(model, cols, topn):
    """
    Plot feature importance from tree-based models
    model: a tree-based model object
    cols: feature names
    """
    feature_importance = list(zip(cols, model.feature_importances_))
    dtype = [('feature','S10'),('importance','float')]
    feature_importance = np.array(feature_importance, dtype = dtype)
    
    # 1D np array with len = n_feature, each element is a tuple (feature, imp)
    feature_sort = np.sort(feature_importance, order = 'importance')[::-1]
    
    featureNames, featureScores = zip(*list(feature_sort[0:topn][::-1]))
    plt.figure().set_size_inches(8, 6)
    plt.barh(range(len(featureScores)), featureScores, tick_label=featureNames)   
   
    feature, imp = zip(*list(feature_sort))
    feature_imp_df = pd.DataFrame({'feature':feature, 'imp':imp})
    
    return feature_imp_df

def display_tree(tree_model, cols):
    """
    tree_model: a decision tree model (e.g. an estimator from random forest model)
    """
    from graphviz import Source
    from IPython.display import SVG
    from sklearn import tree
    graph = Source(tree.export_graphviz(tree_model, out_file=None, feature_names= cols))
    SVG(graph.pipe(format='svg'))
    
def root_mean_squared_error(model, X, y):
    return np.sqrt(mean_squared_error(y, model.predict(X)))

def print_grid_scores(gs):
    """
    gs is a gridsearchCV object that has been fit
    this is equivalent to gs.grid_scores_
    """
    means = np.sqrt( - gs.cv_results_['mean_test_score'] )
    stds = gs.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        print("mean of RMSE: %.4f, std of MSE: %0.4f for %r" %(mean, std * 2, params))
        
def permutation_importances(rf, X_train, y_train):
    """
    Return array of importances from pre-fit rf; metric is function
    that measures accuracy or R^2 or similar. This function
    works for regressors and classifiers.
    rf is a randomforest model that has been fit already with oob_score turned on
    """

    baseline = oob_regression_mse_score(rf, X_train, y_train)
    imp = []
    for col in X_train.columns:
        save = X_train[col].copy()
        X_train[col] = np.random.permutation(X_train[col])
        m = oob_regression_mse_score(rf, X_train, y_train)
        X_train[col] = save
        change_in_metric = m - baseline
        imp.append(change_in_metric)
    return imp

def oob_regression_r2_score(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) R^2 for a scikit-learn random forest
    regressor. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
    """
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train

    n_samples = len(X)
    predictions = np.zeros(n_samples)
    n_predictions = np.zeros(n_samples)
    for tree in rf.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
        n_predictions[unsampled_indices] += 1

    if (n_predictions == 0).any():
        warnings.warn("Too few trees; some variables do not have OOB scores.")
        n_predictions[n_predictions == 0] = 1

    predictions /= n_predictions

    oob_score = r2_score(y, predictions)
    return oob_score

def oob_regression_mse_score(rf, X_train, y_train):
    """
    Compute out-of-bag (OOB) MSE for a scikit-learn random forest
    regressor. We learned the guts of scikit's RF from the BSD licensed
    code:
    https://github.com/scikit-learn/scikit-learn/blob/a24c8b46/sklearn/ensemble/forest.py#L702
    """
    X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    y = y_train.values if isinstance(y_train, pd.Series) else y_train

    n_samples = len(X)
    predictions = np.zeros(n_samples)
    n_predictions = np.zeros(n_samples)
    for tree in rf.estimators_:
        unsampled_indices = _generate_unsampled_indices(tree.random_state, n_samples)
        tree_preds = tree.predict(X[unsampled_indices, :])
        predictions[unsampled_indices] += tree_preds
        n_predictions[unsampled_indices] += 1

    if (n_predictions == 0).any():
        warnings.warn("Too few trees; some variables do not have OOB scores.")
        n_predictions[n_predictions == 0] = 1

    predictions /= n_predictions

    oob_score = mean_squared_error(y, predictions)
    return oob_score
    

            
    
    



    
