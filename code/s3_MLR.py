# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 18:05:11 2019

@author: yanqi
"""
import os
import numpy  as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sklearn
assert(sklearn.__version__ > '0.18' and sklearn.__version__ < '0.20')
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm 
import sklearn.model_selection as ms

# so that we can see more content on the same row
pd.set_option('display.width',1000)

proj_path = 'C:\\Users\\yanqi\\Documents\\NYCDSA\\Python Machine Learning\\Housing Price Prediction\\house_price_prediction\\code'
os.chdir(proj_path)
infname = "../results/s2_clean_dummified.csv"
df = pd.read_csv(infname)


# Multiple linear regression with all features, without normalization or standardization
ols = LinearRegression()
X = df.copy()
X.drop(['LogSalePrice','Id'], axis = 1, inplace = True)
Y = df['LogSalePrice']
ols.fit(X,Y)
print("R^2 of MLR model on all data: %f" %ols.score(X,Y))

feature_type = {}
# figure out which ones are continuous features and which ones are dummy variables
for name in X.columns:
    vals =  np.sort(X[name].unique())
    if (len(vals) == 2) & (vals[0] == 0) & (vals[1] == 1):
        feature_type[name] = 'dummy'
    else:
        feature_type[name] = 'cont'
        
cont_vars = [v for v in feature_type if feature_type[v] == 'cont']
dummy_vars = [v for v in feature_type if feature_type[v] == 'dummy']

# Run MLR on each variable regressed on the rest of the features
# This shows high multi-colinearity problem 
scores = {}
ols2 = LinearRegression()

for v in cont_vars:
    X2 = X[cont_vars].copy()
    Y2 = X2[v].copy()
    X2.drop(v, axis=1, inplace=True)
    ols2.fit(X2, Y2)
    scores[v] = ols2.score(X2, Y2)

ovr_scores = pd.DataFrame(scores, index = ['R2'])
ovr_scores = ovr_scores.T.reset_index()
ovr_scores.rename(columns={'index':'FeatureAsResponse'}, inplace=True)
print(ovr_scores.sort_values(by='R2',ascending=False))
#sns.barplot(x = 'FeatureAsResponse', y = 'R2', data = ovr_scores)
#plt.title('R2 of a continuous feature against the other features')

# take a look at coefficients, all of them very small
coefs = pd.DataFrame(list(zip(X.columns,ols.coef_)))
coefs.columns = ['feature','coef']
coefs.set_index('feature')
coefs['coef'].hist(rwidth = 0.8)
plt.xlabel('Coefficients fitted from MLS on all data')
plt.ylabel('Number of features')
plt.title('Histogram of MLR coefficients (un-normalized features) ',fontsize = 12)

# Use statsmodel to look at overall model fit
ind_std = True
Xsm = X.copy()
if ind_std == True:
    scaler = StandardScaler()
    scaler.fit(Xsm)
    Xsm = scaler.transform(Xsm)

Xsm = pd.DataFrame(Xsm, columns = X.columns)

X_add_const = sm.add_constant(Xsm)
ols = sm.OLS(Y, X_add_const)
ans = ols.fit()
print(ans.summary())

table = pd.DataFrame(ans.summary().tables[1].data[1:])
table.columns = ['name','coef','std err','t value','p value','2.5% confidence','97.5% confidence']
table = table.astype({'name':str,'coef':float,'std err':float, 't value':float, 'p value':float,'2.5% confidence':float, '97.5% confidence':float})
subtable = table[table['p value']<0.05][['name','coef','std err']]
subtable = subtable.sort_values(by='coef',ascending=False)

# take a look at 