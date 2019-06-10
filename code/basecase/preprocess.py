# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:55:54 2019

@author: yanqi
"""

import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 3 - Machine Learning\\Housing Price Prediction\\house_price_prediction\\code\\basecase'
os.chdir(proj_path)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy  as np
import pandas as pd
from sklearn import linear_model
plt.style.use('ggplot')
from helper import chk_mv, train_file, test_file, labelencode_feature
    
def chk_LotFrontage_nhood(df1): 
    sns.lmplot(y = 'LotFrontage', x = 'LotArea', data = df1[df1.LotArea < 20000], col = 'Neighborhood', \
           sharey = False, sharex = False, height = 3, col_wrap = 3, scatter_kws={'s':8, 'alpha':0.5, 'edgecolor':"black"})   # 

def chk_porch(df1):
    df = df1.copy()
    df['Porch'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']
    df['Porch2'] = df['OpenPorchSF'] + df['ScreenPorch']
    df['PorDeck'] = df['OpenPorchSF'] + df['WoodDeckSF']
    df['PorDeck2'] = df['OpenPorchSF'] + df['ScreenPorch'] + df['WoodDeckSF']
    df['PorDeck3'] = df['Porch'] + df['WoodDeckSF']
    
    r = df[ ['WoodDeckSF','PorDeck','PorDeck2','PorDeck3','Porch','Porch2',\
         'OpenPorchSF','EnclosedPorch', '3SsnPorch','ScreenPorch','SalePrice'] ].corr() 

    r1 = r[['SalePrice']]

    print(r)
    print(r1)
    
def fill_mv_LotFrontage_train(df,r0):
    ols = linear_model.LinearRegression()
    nhoods = df.Neighborhood.unique()

    for nhood in nhoods:
        df_n = df[ df['Neighborhood'] == nhood ]
        mv_idx = (df['Neighborhood'] == nhood) & (df['LotFrontage'].isnull())
        
        # if there are mv in this neighborhood
        if np.sum(mv_idx) > 0:
            X = np.array(df_n.loc[ df_n['LotFrontage'].notnull(),['LotArea']]).reshape(-1,1)
            Y = np.array(df_n.loc[ df_n['LotFrontage'].notnull(), ['LotFrontage']])
            ols.fit(X,Y)
            R2 = ols.score(X,Y)
            #print(nhood, "R^2: %.2f" %R2, "beta_1: %.3f" %ols.coef_, "beta_0: %.3f" %ols.intercept_)
        
            # if neighborhood based regression on LotArea has decent R^2
            if R2 > r0:
                df.loc[ mv_idx , ['LotFrontage'] ] = ols.predict( np.array(df.loc[mv_idx, 'LotArea' ]).reshape(-1,1) )
                #print("imputed with regression \n", df.loc[ mv_idx , ['LotFrontage'] ],"\n" )
            else:
                df.loc[ mv_idx , ['LotFrontage'] ] = np.median(Y)
                #print("imputed with neighborhood median \n",  df.loc[ mv_idx , ['LotFrontage'] ],"\n" )
    return df
    
def fill_mv_train(df, NA2Nonecols, r0):
    """
    This function fills in missing value 
    (1) NA2None: fill MV with string 'None' for the columns where 'NA' meant 'None'
    (2) special values such as 0 or mode
    (3) LotFrontage based on within neighborhood regression or median of LotArea
    """
    for name in NA2Nonecols:
        df.loc[ df[name].isnull(), [name] ] = 'None'
        
    # MasVnrArea: missing means no veneer
    df.loc[ df['MasVnrArea'].isnull(), ['MasVnrArea'] ] = 0
    
    # Electrical
    mv_year = list(df.loc[ df['Electrical'].isnull(), 'YearBuilt' ])[0]
    print(mv_year)
    print(df.loc[ df['YearBuilt'] == mv_year ].Electrical.value_counts())
    df.loc[ df['Electrical'].isnull(), ['Electrical'] ] = 'SBrkr'  # all houses built in 2006 has SBrkr    
  
    # LotFrontage      
    df = fill_mv_LotFrontage_train(df,0.5)  
    return df

def mod_vars_train(df):
    """
    This function modifies the values of existing variables
    - combine categories that are too granular
    - change data type 
    """
    # Functional: change to two categories (Typ or NonTyp)
    print(df['Functional'].value_counts())
    df.loc[ df['Functional'] != 'Typ', ['Functional'] ] = 'NonTyp'
    print(df['Functional'].value_counts())
    
    # PavedDrive: combine P and N
    print(df['PavedDrive'].value_counts())
    df.loc[ df['PavedDrive'] == 'P', ['PavedDrive'] ] = 'N'
    print(df['PavedDrive'].value_counts())
    
    # PoolQC: convert to binary 0 = No pool, 1 = has pool
    print(df['PoolQC'].value_counts())
    idx_none = (df['PoolQC'] == 'None')
    idx_other = (df['PoolQC'] != 'None')
    df.loc[ idx_none , ['PoolQC'] ] = 0
    df.loc[ idx_other, ['PoolQC'] ] = 1
    print(df['PoolQC'].value_counts())
    
    # MoSold: convert to categorical 
    df['MoSold'] = df['MoSold'].apply(str)
    
    return df

def new_features_train(df):
    """
    This function generates new features from existing features
    """
    # Create 2 new house age Variables from YearBuilt, YearRemodAdd, YrSold
    df['Age'] = df['YrSold'] - df['YearBuilt']
    df['Re_Age'] = df['YrSold'] - df['YearRemodAdd']
    
    # Create new variables from Condition1 and Condition2 -- consider collapse
    #print(df['Condition1'].value_counts())
    #print(df['Condition2'].value_counts())
    df.groupby(['Condition1','Condition2'])['Condition1'].agg(['count'])
    allcon = np.union1d(df['Condition1'].unique(),df['Condition2'].unique())
    
    for con in allcon:
        df[con] = 0
        df.loc[ (df['Condition1'] == con) | (df['Condition2'] == con) , [con] ] = 1
        #print(df[con].value_counts(),"\n")
    
    return df

def dummify_features(df):
    num_cols = df._get_numeric_data().columns  
    cat_cols = list(set(df.columns) - set(num_cols))
    df_dummy = pd.get_dummies(df[cat_cols], drop_first= True)
    df.drop(cat_cols,axis=1,inplace=True)
    df = pd.concat([df,df_dummy],axis=1)
    return df

def fill_mv_test(df, NA2Nonecols, NA2ZERO, r0):
    """
    This function fills in missing value 
    (1) NA2None: fill MV with string 'None' for the columns where 'NA' meant 'None'
    (2) NA2ZERO: where NA is more approriately interpreted as 0
    (2) special values such as 0 or mode
    (3) LotFrontage based on within neighborhood regression or median of LotArea
    """
    
    for name in NA2Nonecols:
        df.loc[ df[name].isnull(), [name] ] = 'None'
        
    for var in NA2ZERO:
        df.loc[ df[var].isnull(), [var] ] = 0
     
    # Missing value - special cases, fill with values based on manual inspection of cases
    df.loc[ df['KitchenQual'].isnull(), ['KitchenQual'] ] = "TA"
    df.loc[ df['Utilities'].isnull(), ['Utilities'] ] = "AllPub" 
    df.loc[ df['Functional'].isnull(), ['Functional'] ] = "Typ"
    df.loc[ df['Exterior1st'].isnull(), ['Exterior1st'] ] = df.Exterior1st.mode()[0]
    df.loc[ df['Exterior2nd'].isnull(), ['Exterior2nd'] ] = df.Exterior2nd.mode()[0]
    df.loc[ df['SaleType'].isnull(), ['SaleType'] ] = "WD"
 
    # MSZoning: 4 MVs, use the most prevalent MSZoning values in the neighborhood to impute
    df.loc[ (df['MSZoning'].isnull()) & (df['Neighborhood'] == "IDOTRR"), ['MSZoning'] ] = "RM"
    df.loc[ (df['MSZoning'].isnull()) & (df['Neighborhood'] == "Mitchel"), ['MSZoning'] ] = "RL"
    
    # LotFrontage      
    df = fill_mv_LotFrontage_test(df, r0)  
    return df
    
def fill_mv_LotFrontage_test(df_test,r0):
    df_train = pd.read_csv(train_file)
    ols = linear_model.LinearRegression()
    nhoods = df_test.Neighborhood.unique()

    for nhood in nhoods:
        df_n_train = df_train[ df_train['Neighborhood'] == nhood ]
        mv_idx = (df_test['Neighborhood'] == nhood) & (df_test['LotFrontage'].isnull())
        
        # if there are mv in this neighborhood
        if np.sum(mv_idx) > 0:
            X_train = np.array(df_n_train.loc[ df_n_train['LotFrontage'].notnull(),['LotArea']]).reshape(-1,1)
            Y_train = np.array(df_n_train.loc[ df_n_train['LotFrontage'].notnull(), ['LotFrontage']])
            ols.fit(X_train,Y_train)
            R2 = ols.score(X_train,Y_train)
            #print(nhood, "R^2: %.2f" %R2, "beta_1: %.3f" %ols.coef_, "beta_0: %.3f" %ols.intercept_)
        
            # if neighborhood based regression on LotArea has decent R^2
            if R2 > r0:
                df_test.loc[ mv_idx , ['LotFrontage'] ] = ols.predict( np.array(df_test.loc[mv_idx, 'LotArea' ]).reshape(-1,1) )
                #print("imputed with regression on LotArea, based on training data \n", df_test.loc[ mv_idx , ['LotFrontage'] ],"\n" )
            else:
                df_test.loc[ mv_idx , ['LotFrontage'] ] = np.median(Y_train)
                #print("imputed with neighborhood median from training data \n",  df_test.loc[ mv_idx , ['LotFrontage'] ],"\n" )
    return df_test

def make_train_data():
    raw = pd.read_csv(train_file)
    #raw = raw.drop('Id',axis=1)
    raw.shape
    pd.set_option('display.max_columns', 90)
    raw.head()
    chk_mv(raw)
    df1 = raw.copy()
    
    # missing value imputation
    NA2None = ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1', \
                   'BsmtFinType2','FireplaceQu','GarageType','GarageYrBlt','GarageFinish', \
                   'GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
    df1 = fill_mv_train(df1,NA2None, 0.5)
    chk_mv(df1)
    df1 = mod_vars_train(df1)
    df1 = new_features_train(df1)
    
    # log transform response variable to make it normally distributed
    df1['LogSalePrice'] = np.log(df1['SalePrice'])
    
    # remove 2 outliers, and reset index 
    outliers = df1[ (df1['GrLivArea'] > 4000) & (df1['LogSalePrice'] < 13) ].index
    df1.drop(outliers,axis=0, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    y = df1['LogSalePrice']
    
    # drop minimum number of columns
    df1.drop(['MSSubClass','YrSold','YearBuilt','YearRemodAdd','Norm','Condition1', \
              'Condition2', 'GarageYrBlt','Id','SalePrice','LogSalePrice'], axis = 1,inplace=True)
    
    # features without dummification, with dummification, with labelencoding
    X = df1.copy()
    Xd = dummify_features(df1)
    return X, Xd, y

def make_test_data():
    # clean test data
    raw = pd.read_csv(test_file)
    raw.shape
    pd.set_option('display.max_columns', 90)
    raw.head()
    
    chk_mv(raw)
    
    # Missing Value - Most common case: NA means None, e.g. NA for Alley means "No access to Alley"
    df1 = raw.copy()
    NA2None = ['Alley','MasVnrType','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1', \
               'BsmtFinType2','FireplaceQu','GarageType','GarageYrBlt','GarageFinish', \
               'GarageQual','GarageCond','PoolQC','Fence','MiscFeature']
    
    NA2ZERO = ['MasVnrArea', 'BsmtHalfBath', 'BsmtFullBath', 'BsmtFinSF1', 'BsmtFinSF2', \
                   'BsmtUnfSF', 'TotalBsmtSF', 'GarageCars', 'GarageArea']
    
    df1 = fill_mv_test(df1, NA2None, NA2ZERO, 0.5)
    df1 = mod_vars_train(df1)
    df1 = new_features_train(df1)
    
    # drop minimum number of columns
    Id = df1['Id']
    df1.drop(['MSSubClass','YrSold','YearBuilt','YearRemodAdd','Norm','Condition1', \
              'Condition2', 'GarageYrBlt','Id'], axis = 1,inplace=True)
    
    # features without dummification, with dummification, with labelencoding
    X = df1.copy()
    Xd = dummify_features(df1)
    
    return X, Xd, Id

# get rid of dummy columns that only appear in train or test but not the other
X_train, Xd_train, y_train = make_train_data()
X_test, Xd_test, Id_test = make_test_data()
col1 = set(Xd_train.columns) - set(Xd_test.columns)
col2 = set(Xd_test.columns) - set(Xd_train.columns)
Xd_train.drop( col1, axis=1, inplace = True )
Xd_test.drop( col2, axis=1, inplace = True)

# labelencode features for use with tree-based models
X_le_train, X_le_test = labelencode_feature(X_train, X_test)

# save training and test data to pickle file
import pickle
with open('train.pickle', 'wb') as f:
    pickle.dump([X_train, Xd_train, y_train, X_le_train], f)
    
with open('test.pickle','wb') as f:
    pickle.dump([X_test, Xd_test, Id_test, X_le_test], f)
    
#Xd_train.to_csv("Xd_train.csv", index = False, header = True)
#y_train.to_csv("y_train.csv", index = False, header = True)
#Xd_test.to_csv("Xd_test.csv", index = False, header = True)










