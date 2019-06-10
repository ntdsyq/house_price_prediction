Predict house price in Ames, Iowa sold 2006 to 2010 using machine learning techniques
Data source: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

folder /data contains the training data, test data and data description from the Kaggle competition. 
folder /code has two subfolders /basecase and /engineered. These two folders contains the code, analysis, results and submission files for two datasets. 
1) We obtained the basecase data by applying fundational pre-processing to the training and testing data, including missing value imputation, outlier removal, and a few basic variable transformations. Although extensive EDA accompanied the making of this dataset, we did little feature selection/engineering at this stage, keeping almost all features from original data. This dataset allowed us to set up a process for tuning our models, gauge baseline prediction performance, and obtain insights on feature importance based on the model outputs. This folder includes code for pre-processing, training/testing with lasso, ridge, elasticnet, decision tree, random forest, and xgboost models, and for analyzing feature importance. The name of the scripts should be fairly self explanatory. The helper.py file contains convenience functions we defined for this project.

2) Using the basecase data as a starting point, we experimented with various feature selection and feature engineering techniques to arrive at the engineered dataset. Our FS and FE efforts were guided by the feature importance from the baseline models and more EDA. As lasso performed the best on the baseline data among all models, we focused on engineering features that improves the performance of the lasso model. In addition to individual models, we also tested stacking models on this dataset. 

We were able to achieve RMSE of 0.11664 (top 15%) with the stacking model. Please see details in this blogpost [insert link]. 


