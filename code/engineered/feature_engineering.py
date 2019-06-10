# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:05:45 2019

@author: yanqi
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:07:02 2019

@author: yanqi
"""
import os
proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Project 3 - Machine Learning\\Housing Price Prediction\\house_price_prediction\\code\\change_02'
os.chdir(proj_path)

import numpy as np
import pandas as pd
from scipy.special import boxcox1p
from scipy.stats import skew, boxcox_normmax
from helper import labelencode_feature

# start with 83 features
import pickle
with open('train_prep.pickle', 'rb') as f:
    X_tr, tmp1, y_train, tmp2 = pickle.load(f)
    
with open('test_prep.pickle', 'rb') as f:
    X_te, tmp1, Id_test, tmp2 = pickle.load(f)
    
def FE(Xin):
    X = Xin.copy()
# drop 4: drop features that have very little variation across samples, these also have small feature importance across all models
    drop_vars = ['Utilities','Street','PoolQC','PoolArea','Heating']
    X.drop(drop_vars, axis = 1, inplace = True)
    
    # create a few new variables
    X['TotalSF'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF']  # total house square foot
    X['Has2ndFlr'] = 0
    X.loc[ X['2ndFlrSF'] > 0, 'Has2ndFlr' ] = 1
    X.drop(['2ndFlrSF'], axis = 1, inplace = True) 
    X['TotalBath'] = X['BsmtFullBath'] + X['FullBath'] + X['BsmtHalfBath']*0.5 + X['HalfBath']*0.5
    X['BsmtHasBath'] = (X['BsmtFullBath'] + X['BsmtHalfBath'])*1
    X.drop(['BsmtFullBath','HalfBath','FullBath','BsmtHalfBath'], axis = 1, inplace = True)
    X['OverallQualCon'] = X['OverallQual']*X['OverallCond']  # material finish coupled with how well kept the house is
    #X['HouseQualCon'] = X['TotalSF']*X['OverallQualCon']
    #X['HouseQual'] = X['TotalSF']*X['OverallQual']
    X['HouseCond'] = X['TotalSF']*X['OverallCond']
    #X['GrLivingQualCon'] = X['GrLivArea']*X['OverallQualCon']
    #X['GrLivingQual'] = X['GrLivArea']*X['OverallQual']
    
#    # SaleCondition ranks higher than SaleType in basecase models, SaleType new fully captured by SaleCondition, drop SaleType 
#    X.drop(['SaleType'], axis = 1, inplace = True)
#    
#    # Porch variables: EnclosedPorch, 3SsnPorch did not appear significant in any models in the basecase, drop
#    X.drop(['EnclosedPorch', '3SsnPorch'], axis = 1, inplace = True)
#    
#    # MSZoning: combine 'RM','RH' to 'RMH'
#    X['MSZoning'].value_counts()
#    X.loc[ X['MSZoning'] == 'RM', 'MSZoning' ]  = 'RMH'
#    X.loc[ X['MSZoning'] == 'RH', 'MSZoning' ] = 'RMH'
#    X['MSZoning'].value_counts()
#    
#    # MoSold: group into Spring, Summer, Fall, Winter
#    X['SeasonSold'] = 'None'
#    X.loc[X['MoSold'].isin(['1', '2', '12']), 'SeasonSold' ] = 'Winter' 
#    X.loc[X['MoSold'].isin(['3', '4', '5']), 'SeasonSold' ] ='Spring'
#    X.loc[X['MoSold'].isin(['6', '7', '8']), 'SeasonSold' ] = 'Summer'
#    X.loc[X['MoSold'].isin(['9', '10', '11']), 'SeasonSold' ] = 'Fall'
#    X.drop(['MoSold'], axis = 1, inplace = True)
    
    # process the garage variables
    # GarageQual: combine Po and None, Gd and Ex: new categories: 'Po_None', 'Fa', 'TA', 'Gd_Ex'
    print(X['GarageQual'].value_counts())
    X.loc[ X['GarageQual'].isin(['Po','Fa']), 'GarageQual' ] = 'Po_Fa'
    X.loc[ X['GarageQual'].isin(['Gd','Ex']), 'GarageQual' ] = 'Gd_Ex'
    print(X['GarageQual'].value_counts())
    # GarageCond: drop, highly correlated with GarageQual, similar FI ranking in tree models, GarageQual vs. y better ordinality
    # X.drop(['GarageCond'], axis = 1, inplace = True)
    # keep GarageFnish, as it presents different info than quality or condition
    # GarageType: keep. ranked top 15 in tree-models, also multiple non-zero coefs in lasso
    # Create new indicator variables, later need to delete the None columns from all Garage Variables
    X['HasGarage'] = 1
    X.loc[ X['GarageType'] == 'None', 'HasGarage'] = 0
    print(X.HasGarage.value_counts())
    X['HasBsmt'] = X['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    X['HasFireplace'] = X['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
    X.drop(['Fireplaces'], axis = 1, inplace = True)
    
    # Process basement variables
    # add 1, drop 4: create BsmtGLQFinSF from BsmtFinSF1, BsmtFinType1, BsmtFinSF2, BsmtFinType2
    X['BsmtGLQFinSF'] = (X['BsmtFinType1'] == 'GLQ')*X['BsmtFinSF1'] + (X['BsmtFinType2'] == 'GLQ')*X['BsmtFinSF2']
    X.drop(['BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2'], axis = 1,inplace=True)
    
    # FI order: 'BsmtQual' > 'BsmtExposure' > 'BsmtCond' in RF
    # FI order:  'BsmtExposure' far more important than the other two (got zero coefs)
#    X.drop(['BsmtCond'], axis = 1, inplace = True)
    
#    # This set of columns may be collapsed to fewer
#    old_cond_vars = ['Artery', 'Feedr', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe']
#    print(X[old_cond_vars].apply(lambda x: x.value_counts()))
#    X['NearRd'] = 0
#    X.loc[ (X['Artery'] + X['Feedr']) > 0, 'NearRd'  ] = 1
#    X['NearRail'] = 0
#    X.loc[ (X['RRNn'] + X['RRAn'] + X['RRNe'] + X['RRAe']) > 0 , 'NearRail'  ] = 1
#    X['PosCond'] = 0
#    X.loc[ (X['PosN'] + X['PosA']) > 0, 'PosCond' ] = 1
#    new_cond_vars = ['NearRd', 'NearRail','PosCond']
#    print(X[new_cond_vars].apply(lambda x: x.value_counts()))
#    X.drop(old_cond_vars, axis = 1, inplace = True)
    X['TotPorchDeckSF'] = (X['OpenPorchSF'] + X['3SsnPorch'] +
                              X['EnclosedPorch'] + X['ScreenPorch'] +
                              X['WoodDeckSF'])
    X.drop(['OpenPorchSF','3SsnPorch','EnclosedPorch','ScreenPorch','WoodDeckSF'],axis=1, inplace=True)
    X.drop(['GarageArea','TotRmsAbvGrd','TotalBsmtSF','MiscVal','MiscFeature'], axis=1, inplace=True)
    return X
   
def create_Xdummy(df):
    """
    Dummify categorical variables in feature matrix
    Type1: for some Garage & Bsmt variables, manually delete "None" columns (cat_cols1)
    Type2: auto delete the first dummified column for the rest of categorical variables
    """
    num_cols = df._get_numeric_data().columns  
    cat_cols = list(set(df.columns) - set(num_cols))
    cat_cols1 = ['GarageFinish', 'GarageQual', 'GarageType','GarageCond', 'BsmtQual','BsmtCond', 'BsmtExposure']
    df_dummy1 = pd.get_dummies(df[cat_cols1])
    for var in cat_cols1:
        df_dummy1.drop(var+'_None',axis=1,inplace=True)
    #print(df_dummy1.shape)
    
    cat_cols2 = list(set(cat_cols) - set(cat_cols1))
    df_dummy2 = pd.get_dummies(df[cat_cols2], drop_first= True)
    #print(df_dummy2.shape)
    df_all = pd.concat([df[num_cols],df_dummy1, df_dummy2],axis=1)  
    #print(df_all.shape)  
    
    return df_all

# Apply feature engineering to training and test data
X_train = FE(X_tr)
print("shape and columns of X_train after initial processing")
print(X_train.shape)
print(X_train.columns)
X_test = FE(X_te)
print("shape and columns of X_test after initial processing")
print(X_test.shape)
print(X_test.columns)

 # fix skew in some variables
fix_var = ['LotFrontage', 'LotArea', 'BsmtUnfSF', '1stFlrSF', 'GrLivArea','TotalSF']
for var in fix_var:
    lam = boxcox_normmax(X_train[var] + 1)
    X_train[var] = boxcox1p(X_train[var], lam )
    X_test[var] = boxcox1p(X_test[var],lam )

# Dummify X_train & X_test
Xd_train = create_Xdummy(X_train)
Xd_test = create_Xdummy(X_test)

# get rid of columns that only appear in 1 dataset but not the other
col1 = set(Xd_train.columns) - set(Xd_test.columns)
col2 = set(Xd_test.columns) - set(Xd_train.columns)
Xd_train.drop( col1, axis=1, inplace = True )
Xd_test.drop( col2, axis=1, inplace = True)

# remove dummy columns that have very very low number of 1s
col2del = []
for col in Xd_train.columns: 
    vc = Xd_train[col].value_counts()
    if vc.iloc[0]/Xd_train.shape[0] > 0.995:
        col2del.append(col)
print(col2del)
Xd_train.drop( col2del, axis=1, inplace = True )
Xd_test.drop( col2del, axis=1, inplace = True)

# Create label encoded data
X_le_train, X_le_test = labelencode_feature(X_train,X_test)

# Save processed training and testing data for modeling
with open('train.pickle', 'wb') as f:
    pickle.dump([X_train, Xd_train, y_train, X_le_train], f)
    
with open('test.pickle','wb') as f:
    pickle.dump([X_test, Xd_test, Id_test, X_le_test], f)

    
    