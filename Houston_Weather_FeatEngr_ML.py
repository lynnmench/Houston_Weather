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

Hyperparameter Grid Search with XGBoost
    https://www.kaggle.com/code/tilii7/hyperparameter-grid-search-with-xgboosthttps://www.kaggle.com/code/tilii7/hyperparameter-grid-search-with-xgboost

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

#model accuaracy
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.metrics import precision_score, recall_score, f1_score


#Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.svm import SVC
from imblearn.ensemble import EasyEnsembleClassifier


#Balancing Methods
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

#Hyperparameter Turning
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


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
df_eval = df.drop(feat_drop, axis=1)

#To keep track of model scores
ml_scores = []
hyper_scores = []
best_estimators = {}

#function to evaluate models:
def model_test_noparm(ml_dic, X_tr, X_te, y_tr, y_te):
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
    print(f"Training target statistics: {Counter(y_train)}")
    print(f"Testing target statistics: {Counter(y_test)}")
    
    for name, ml in ml_dic.items():
        model = ml['model']
        
        #No hyperparameter turning
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        
        #with hyperparameter tuning
        #clf=GridSearchCV(log_class,grid,cv=cv,n_jobs=-1,scoring='f1_macro')
        #clf.fit(X_train,y_train)
        #y_pred=clf.predict(X_test)
        
        print(name)
        print(confusion_matrix(y_te,y_pred))
        print(classification_report(y_te,y_pred))
        print(accuracy_score(y_te,y_pred))
        print('Recall Score 1: ', recall_score(y_te, y_pred))
        precision = round(precision_score(y_test, y_pred)*100, 2)
        recall = round(recall_score(y_te, y_pred)*100, 2)
        f1 = round(f1_score(y_te, y_pred)*100, 2)
        
        ml_scores.append({
            'model': name,
            'precision_score':precision,
            'recall_score': recall,
            'f1_score': f1
            })


# Train Test Split
X = df_eval.drop(['rain_tomorrow'], axis=1)
y = df_eval['rain_tomorrow']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)
print(f"Training target statistics: {Counter(y_train)}")
print(f"Testing target statistics: {Counter(y_test)}")


#Dictionary of models to test
#Testing classifiers on unbalanced data and comparing to log regression
model_eval_unbal = {'log_reg_unbal':{'model':LogisticRegression()},
                    'rfc_unbal':{'model':RandomForestClassifier()},
                    'xgb_unbal':{'model':xgb.XGBClassifier()},
                    'knn_unbal':{'model':KNeighborsClassifier()},
                    'svm_c_unbal':{'model':SVC()}
                    }


#looking at the models with unbalanced data
#just to get a base line understanding of the data
model_test_noparm(model_eval_unbal, X_train, X_test, y_train, y_test)

#Before balancing the data and using hyperparameters 
#Best Scores: XGBoost (18.82%) and KNN (16.675)


####    Class Weight   ####

#using class weight - training data is about 22% so using 21 to 1 ratio for pos and neg

ratio = Counter(y_train)[0]/Counter(y_train)[1]
print(ratio)

model_eval_unbal = {'log_reg_weight':{'model':LogisticRegression(class_weight={0:1, 1:ratio})},
                    'rfc_weight':{'model':RandomForestClassifier(class_weight={0:1, 1:ratio})},
                    'xgb_weight':{'model':xgb.XGBClassifier(scale_pos_weight=ratio)},
                    'knn_weight':{'model':KNeighborsClassifier()},
                    'svm_c_weight':{'model':SVC(class_weight={0:1, 1:ratio})}
                    }
model_test_noparm(model_eval_unbal, X_train, X_test, y_train, y_test)

#Using weights was a huge improvment
#best score was XGBoost with a recall of 40% and f1-score of 41%


####    Under Sample Methods    ####

### Random under sample

under_sampler = RandomUnderSampler(random_state=11)
X_train_res, y_train_res = under_sampler.fit_resample(X_train, y_train)
print(f"Training target statistics: {Counter(y_train_res)}")
print(f"Testing target statistics: {Counter(y_test)}")

#evaluating on base of models
model_eval_unbal = {'log_reg_under_ran':{'model':LogisticRegression()},
                    'rfc_under_ran':{'model':RandomForestClassifier()},
                    'xgb_under_ran':{'model':xgb.XGBClassifier()},
                    'knn_under_ran':{'model':KNeighborsClassifier()},
                    'svm_c_under_ran':{'model':SVC()}
                    }
model_test_noparm(model_eval_unbal, X_train_res, X_test, y_train_res, y_test)

#The recall score improved but the preceision score dropped significantly

### Near Miss

under_sampler = NearMiss()
X_train_res, y_train_res = under_sampler.fit_resample(X_train, y_train)
print(f"Training target statistics: {Counter(y_train_res)}")
print(f"Testing target statistics: {Counter(y_test)}")
model_test_noparm(model_eval_unbal, X_train_res, X_test, y_train_res, y_test)
model_eval_unbal = {'log_reg_under_nm':{'model':LogisticRegression()},
                    'rfc_under_nm':{'model':RandomForestClassifier()},
                    'xgb_under_nm':{'model':xgb.XGBClassifier()},
                    'knn_under_nm':{'model':KNeighborsClassifier()},
                    'svm_c_under_nm':{'model':SVC()}
                    }
model_test_noparm(model_eval_unbal, X_train_res, X_test, y_train_res, y_test)

#The recall score improved but the preceision score dropped


####    Over Sample Methods    ####

### Random Over Sample
over_sampler = RandomOverSampler(random_state=11)
X_train_res, y_train_res = over_sampler.fit_resample(X_train, y_train)
print(f"Training target statistics: {Counter(y_train_res)}")
print(f"Testing target statistics: {Counter(y_test)}")
model_test_noparm(model_eval_unbal, X_train_res, X_test, y_train_res, y_test)
model_eval_unbal = {'log_reg_over_ran':{'model':LogisticRegression()},
                    'rfc_over_ran':{'model':RandomForestClassifier()},
                    'xgb_over_ran':{'model':xgb.XGBClassifier()},
                    'knn_over_ran':{'model':KNeighborsClassifier()},
                    'svm_c_over_ran':{'model':SVC()}
                    }
model_test_noparm(model_eval_unbal, X_train_res, X_test, y_train_res, y_test)


### SMOTE
#over_sampler = SMOTE(k_neighbors=4)
over_sampler = SMOTE(random_state=11)
X_train_res, y_train_res = over_sampler.fit_resample(X_train, y_train)
print(f"Training target statistics: {Counter(y_train_res)}")
print(f"Testing target statistics: {Counter(y_test)}")
model_test_noparm(model_eval_unbal, X_train_res, X_test, y_train_res, y_test)
model_eval_unbal = {'log_reg_over_smote':{'model':LogisticRegression()},
                    'rfc_over_smote':{'model':RandomForestClassifier()},
                    'xgb_over_smote':{'model':xgb.XGBClassifier()},
                    'knn_over_smote':{'model':KNeighborsClassifier()},
                    'svm_c_over_smote':{'model':SVC()}
                    }
model_test_noparm(model_eval_unbal, X_train_res, X_test, y_train_res, y_test)

# Both have more balanced results than undersampling but do not perform as well as class weight



#### Combin Over and Under -> SMOTE Tomek #####

smote_tomek = SMOTETomek(random_state=11)
X_train_res, y_train_res=smote_tomek.fit_resample(X_train,y_train)
print(f"Training target statistics: {Counter(y_train_res)}")
print(f"Testing target statistics: {Counter(y_test)}")
model_test_noparm(model_eval_unbal, X_train_res, X_test, y_train_res, y_test)
model_eval_unbal = {'log_reg_smoteTomek':{'model':LogisticRegression()},
                    'rfc_smoteTomek':{'model':RandomForestClassifier()},
                    'xgb_smoteTomek':{'model':xgb.XGBClassifier()},
                    'knn_smoteTomek':{'model':KNeighborsClassifier()},
                    'svm_c_smoteTomek':{'model':SVC()}
                    }
model_test_noparm(model_eval_unbal, X_train_res, X_test, y_train_res, y_test)


#### Easy Ensemble Classifier

easy=EasyEnsembleClassifier()
easy.fit(X_train,y_train)
y_pred=easy.predict(X_test)
print("Easy Ensemble Classifier")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print('Recall Score 1: ', recall_score(y_test, y_pred))
precision = round(precision_score(y_test, y_pred)*100, 2)
recall = round(recall_score(y_test, y_pred)*100, 2)
f1 = round(f1_score(y_test, y_pred)*100, 2)

ml_scores.append({
    'model': 'easy_ensemble',
    'precision_score':precision,
    'recall_score': recall,
    'f1_score': f1
    })

# Precision score is very low compared to the recall

#turn scores array in to a data frame
base_scores_df = pd.DataFrame(ml_scores,columns=['model','precision_score','recall_score','f1_score'])
base_scores_df = base_scores_df.sort_values(by=['f1_score'], ascending=False)


#based off of the scores so far the best models are:
# XGBoost with class weight, oversample radom
# Random Forest Classifier oversample random and smote tomek


# with hyperparameter tuning
## XGBoost Class Weight
xgb_model = xgb.XGBClassifier(scale_pos_weight=ratio)
grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
cv = KFold(n_splits=5,random_state=None,shuffle=False)
clf=GridSearchCV(xgb_model, grid, cv=cv, n_jobs=-1,scoring='f1_macro')
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("XGBoost Class Weight")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print('Recall Score 1: ', recall_score(y_test, y_pred))
precision = round(precision_score(y_test, y_pred)*100, 2)
recall = round(recall_score(y_test, y_pred)*100, 2)
f1 = round(f1_score(y_test, y_pred)*100, 2)

hyper_scores.append({
    'model': 'Hyper_XGB_Weight',
    'precision_score':precision,
    'recall_score': recall,
    'f1_score': f1,
    'best_params': clf.best_params_
    })
best_estimators['Hyper_XGB_Weight'] = clf.best_estimator_

## XGBoost Over Sample Radom
xgb_model = xgb.XGBClassifier()
grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }
cv = KFold(n_splits=5,random_state=None,shuffle=False)
clf=GridSearchCV(xgb_model, grid, cv=cv, n_jobs=-1,scoring='f1_macro')
over_sampler = RandomOverSampler(random_state=11)
X_train_res, y_train_res = over_sampler.fit_resample(X_train, y_train)
clf.fit(X_train_res, y_train_res)
y_pred=clf.predict(X_test)

print("XGBoost Over Sample Random")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print('Recall Score 1: ', recall_score(y_test, y_pred))
precision = round(precision_score(y_test, y_pred)*100, 2)
recall = round(recall_score(y_test, y_pred)*100, 2)
f1 = round(f1_score(y_test, y_pred)*100, 2)

hyper_scores.append({
    'model': 'Hyper_XGB_Over_Ran',
    'precision_score':precision,
    'recall_score': recall,
    'f1_score': f1,
    'best_params': clf.best_params_
    })
best_estimators['Hyper_XGB_Over_Ran'] = clf.best_estimator_


## Random Forest Classifier Over Sample Radom
rfc_model = RandomForestClassifier()
grid = {
    'criterion' : ['entropy', 'gini'],
    'max_depth' : [5, 10],
    'max_features' : ['log2', 'sqrt'],
    'min_samples_leaf' : [1,5],
    'min_samples_split' : [3,5],
    'n_estimators' : [6,9]
}
cv = KFold(n_splits=5,random_state=None,shuffle=False)
clf=GridSearchCV(rfc_model, grid, cv=cv, n_jobs=-1,scoring='f1_macro')
over_sampler = RandomOverSampler(random_state=11)
X_train_res, y_train_res = over_sampler.fit_resample(X_train, y_train)
clf.fit(X_train_res, y_train_res)
y_pred=clf.predict(X_test)

print("RFC Over Sample Random")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print('Recall Score 1: ', recall_score(y_test, y_pred))
precision = round(precision_score(y_test, y_pred)*100, 2)
recall = round(recall_score(y_test, y_pred)*100, 2)
f1 = round(f1_score(y_test, y_pred)*100, 2)

hyper_scores.append({
    'model': 'Hyper_RFC_Over_Ran',
    'precision_score':precision,
    'recall_score': recall,
    'f1_score': f1,
    'best_params': clf.best_params_
    })
best_estimators['Hyper_RFC_Over_Ran'] = clf.best_estimator_


## Random Forest Classifier SMOTE Tomek
rfc_model = RandomForestClassifier()
grid = {
    'criterion' : ['entropy', 'gini'],
    'max_depth' : [5, 10],
    'max_features' : ['log2', 'sqrt'],
    'min_samples_leaf' : [1,5],
    'min_samples_split' : [3,5],
    'n_estimators' : [6,9]
}
cv = KFold(n_splits=5,random_state=None,shuffle=False)
clf=GridSearchCV(rfc_model, grid, cv=cv, n_jobs=-1,scoring='f1_macro')
smote_tomek = SMOTETomek(random_state=11)
X_train_res, y_train_res=smote_tomek.fit_resample(X_train,y_train)
clf.fit(X_train_res, y_train_res)
y_pred=clf.predict(X_test)

print("RFC Smote Tomek")
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print('Recall Score 1: ', recall_score(y_test, y_pred))
precision = round(precision_score(y_test, y_pred)*100, 2)
recall = round(recall_score(y_test, y_pred)*100, 2)
f1 = round(f1_score(y_test, y_pred)*100, 2)

hyper_scores.append({
    'model': 'Hyper_RFC_SomteTomek',
    'precision_score':precision,
    'recall_score': recall,
    'f1_score': f1,
    'best_params': clf.best_params_
    })
best_estimators['Hyper_RFC_Over_Ran'] = clf.best_estimator_


hyper_df = pd.DataFrame(hyper_scores,columns=['model','precision_score','recall_score','f1_score', 'best_params'])
hyper_df = hyper_df.sort_values(by=['f1_score'], ascending=False)

scores_df = hyper_df.append(base_scores_df, ignore_index=True)

#file_path = '/Users/lynnpowell/Documents/DS_Projects/Houston_Weather/'
#scores_df.to_csv(data_file_path+'Houston_Weather_ML_Scores.csv',index=False)

