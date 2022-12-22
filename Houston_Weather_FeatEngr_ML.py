#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created: 20Dec2022

Author: Lynn Menchaca

Project: Houston Weather Prediction

"""
"""



Feature Engineering and ML Models:
    - Feature Selection
    - Train Test Split
    - Handel Imbalance Data Set 
    - Scaling/Tranformation and Fit
    - Machine Learning (ML) Models

Resources:
Kaggle Data Set: Houston Weather Data
    https://www.kaggle.com/datasets/alejandrochapa/houston-weather-data

7 Techniques to Handle Imbalanced Data by Ye Wu & Rick Radewagen
    https://www.kdnuggets.com/2017/06/7-techniques-handle-imbalanced-data.html

How to balance a dataset in Python by Angelica Lo Duca
    https://towardsdatascience.com/how-to-balance-a-dataset-in-python-36dff9d12704

How to improve the performance of a (Supervised) Machine Learning Algorithm by Angelica Lo Duca
How to Deal with Imbalanced Multiclass Datasets in Python by Angelica Lo Duca

"""


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#feat select
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier

#data split
from sklearn.model_selection import train_test_split
from collections import Counter

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

data_file_path = '/Users/lynnpowell/Documents/DS_Projects/Data_Files/Houston_Weather_Data/'


df = pd.read_csv(data_file_path + 'weather_data_clean.csv')

df.info()

#convert data feature in to a time data type
df['date'] = pd.to_datetime(df['date'])
df.dtypes

#Output (dependent) features are: rainfall, rain_today, rain_tomorrow

#predicting rain tomorrow
#droping columuns that indicate raining

rain_feat = ['rainfall', 'rain_9am', 'rain_3pm', 'rain_wind_9am', 'rain_wind_3pm', 'rain_today']
df.drop(rain_feat, axis=1, inplace=True)

#dropping date feature - dose not work with feature selection methods
df.drop(['date', 'index'], axis=1, inplace=True)

#convert yes/no to 1/0 for rain_tomorrow
df['rain_tomorrow'] = df['rain_tomorrow'].map({'Yes':1, 'No':0})

#####   Feature Selection      #####

#### Pearson Correlation ####
df_corr = df.corr()
target_corr = df_corr['rain_tomorrow'].abs().sort_values(ascending=False)
target_bin_corr = target_corr.drop(labels=(['rain_tomorrow']))
#corr_feat = target_bin_corr[target_bin_corr > 0.1].index.values.tolist()
#corr_feat_df = pd.DataFrame(data=corr_feat, columns=['Features'])
#corr_order_feat = target_bin_corr.index.values.tolist()
#corr_feat_full_df = pd.DataFrame(data=corr_order_feat, columns=['Features']).reset_index(drop=True)


## Univariate Selection: ##

X = df.drop('rain_tomorrow', axis=1)
y = df['rain_tomorrow']


X_col = X.shape[1]

### Apply SelectKBest Algorithm
### Also refered to as information gain?

ordered_feature = SelectKBest(score_func=chi2, k=X_col).fit(X,y)
#ordered_feature = SelectKBest(score_func=f_classif, k=X_col).fit(X,y)
#ordered_feature = SelectKBest(score_func=mutual_info_classif, k=X_col).fit(X,y)

univar_score = pd.DataFrame(ordered_feature.scores_, columns=['Score'])
univar_col = pd.DataFrame(X.columns)

univar_df = pd.concat([univar_col, univar_score], axis=1)
univar_df.columns=['Features','Score']

# For SelectKBest Algorithm the higher the score the higher the feature importance
univar_df['Score'].sort_values(ascending=False)
univar_df = univar_df.nlargest(50, 'Score').reset_index(drop=True)


#### Information Gain ####
#Looking to see what highly correlated features are important to the final answer

mutual_info_values = mutual_info_classif(X,y)
mutual_info = pd.Series(mutual_info_values, index=X.columns)
mutual_info.sort_values(ascending=False)
mutual_info_df = mutual_info.sort_values(ascending=False).to_frame().reset_index()
mutual_info_df.columns=['Features','Mutual Info']


#Dropping Features
feat_drop = ['day', 'year']
df.drop(feat_drop, axis=1, inplace=True)


# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
print(f"Training target statistics: {Counter(y_train)}")
print(f"Testing target statistics: {Counter(y_test)}")






