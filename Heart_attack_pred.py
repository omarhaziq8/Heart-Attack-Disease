# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:23:01 2022

@author: pc
"""

import os 
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

import scipy.stats as ss 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

#%% Functions

def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

#%% Statics

CSV_PATH = os.path.join(os.getcwd(),'heart.csv')
BEST_MODEL_PATH = os.path.join(os.getcwd(),'Best_Model.pkl')

#%% Data Loading

df = pd.read_csv(CSV_PATH)

df.columns
# Index(['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
      #  'exng', 'oldpeak', 'slp', 'caa', 'thall', 'output'],
      # dtype='object')

#%% Data Inspection

df.info()
# To check dtype all in float64 and int64,no NaN values
df.isnull().sum()
# To confirm there is NO NaN values inside each column
df.duplicated().sum()
# To check duplicated values in dataframe :1 duplicated


# To visualise continous features: use distribution graph
con_columns = ['age', 'trtbps', 'chol','thalachh', 'oldpeak']
for con in con_columns:
    plt.figure()
    sns.distplot(df[con])
    plt.show()

# Tp visualise categorical features: use count plot graph
cater_columns = ['sex', 'cp', 'fbs', 'restecg', 'exng',
                'slp','caa','thall','output']
for cat in cater_columns:
    plt.figure()
    sns.countplot(df[cat])
    plt.show()

# To check specifically inside continous features for the outliers
for con in con_columns:
    plt.figure()
    sns.boxplot(df[con])
    plt.show()
# From boxplot: trtbps,chol and oldpeak shows many outliers

# To illustrate outliers for overall data features
df.boxplot()

#%% Data Cleaning

# No NaNs value to be remove
# This dataset has duplicated values, therefore we need to remove to avoid 
# ruining in train split validation
# use drop duplicate for entire df
df = df.drop_duplicates()
print(df.duplicated().sum())
print(df[df.duplicated()])
# To examine no duplicated values


#%% Features Selection

# Logistic Regression
# categorical vs continous
for con in con_columns:
    print(con)
    lr = LogisticRegression(solver='liblinear')
    lr.fit(np.expand_dims(df[con], axis=-1),df['output'])
    print(lr.score(np.expand_dims(df[con], axis=-1),df['output']))
# Results:
# age
# 0.5894039735099338
# trtbps
# 0.5662251655629139
# chol
# 0.5496688741721855
# thalachh
# 0.6721854304635762
# oldpeak
# 0.6854304635761589

# From logistic regression, we able to relate the thalachh and oldpeak features
# have 0.67 and 0.69 accuracy respectively to the output
# age has 0.59 acc which closest to 60%, might into consideration 

# Cramer's V
# categorical vs categorical
for cat in cater_columns:
    print(cat)
    confusion_mat = pd.crosstab(df[cat],df['output']).to_numpy()
    print(cramers_corrected_stat(confusion_mat))
# Results: 
# sex
# 0.2708470833804965
# cp
# 0.508955204032273
# fbs
# 0.0 # cause the data only contains true and false ; 1:0
# restecg
# 0.1601818218998346
# exng
# 0.42533348943620414
# slp
# 0.38615287747239485
# caa
# 0.48113005767539546
# thall
# 0.5206731262866439

# From Cramers V, the results indicate insignificat features since they
# have low accuracy below than 50%, only thall 0.52 acc
# To conclude, we select thalachh,oldpeak,age and thall

#%% Preprocessing

x = df.loc[:,['age','thalachh','oldpeak','thall']] # Features
y = df.loc[:,'output'] # Target


x_train,x_test,y_train,y_test = train_test_split(x,y,
                                                  test_size=0.3,
                                                  random_state=123)

#%% Machine Learning Development (Pipeline)

# Random Forest pipeline
step_ss_rf = Pipeline([('Standard Scaler', StandardScaler()),
                       ('Random Forest Classifier', RandomForestClassifier())])

step_mms_rf = Pipeline([('Min Max Scaler', MinMaxScaler()),
                        ('Random Forest Classifier', RandomForestClassifier())])

# Decision Tree pipeline
step_ss_dt = Pipeline([('Standard Scaler', StandardScaler()),
                       ('Decision Tree Classifier', DecisionTreeClassifier())])


step_mms_dt = Pipeline([('Min Max Scaler', MinMaxScaler()),
                        ('Decision Tree Classifier', DecisionTreeClassifier())])

# To make pipeline

pipelines = [ step_ss_rf, step_mms_rf,step_ss_dt, step_mms_dt]


# fitting of data
for pipe in pipelines:
    pipe.fit(x_train,y_train)


pipe_dict = {0:'SS+RandomForest', 1:'MMS+RandomForest', 2:'SS+DecisionTree', 
             3:'MMS+DecisionTree'}
best_accuracy = 0

#model evaluation
for i, model in enumerate(pipelines):
    print(pipe_dict[i])
    print(model.score(x_test,y_test))
    if model.score(x_test,y_test) > best_accuracy:
        best_accuracy = model.score(x_test, y_test)
        best_pipeline = model
        best_scaler = pipe_dict[i]


print('The best scaling approach for Heart Attack Dataset will be {} with accuracy of {}'.format(best_scaler,best_accuracy))

# Discussion
# Model used: RandomForest,DecisionTree,LogisticRegression,KNNClassifier,
# SVClassifier
# For preprocessing data scaling, used: MinMaxScaler and StandardScaler
# To compare which one is the best model including scaling, used: pipeline
# Pipelines created for each model and results demonstrate 
# SS+RandomForest
# 0.6703296703296703
# MMS+RandomForest
# 0.7362637362637363 -->> 73%
# SS+DecisionTree
# 0.6923076923076923
# MMS+DecisionTree
# 0.7032967032967034


#%% Fine tuning using GridSearchCv

step_mms_rf = Pipeline([('Min Max Scaler', MinMaxScaler()),
                        ('RandomForestClassifier', RandomForestClassifier())])

#number of trees (n_estimators)
grid_param = [{'RandomForestClassifier__n_estimators':[10,100,1000], 
               'RandomForestClassifier__max_depth':[3,5,7,10],
               'RandomForestClassifier__min_samples_leaf':np.arange(1,5)}] 
#maxdepth:to stop the operation from split, if No 'max depth', 
# it can produce overfitting which it splits only one side

gridsearch = GridSearchCV(step_mms_rf,grid_param,cv=5,verbose=1,n_jobs=-1)
best_model = gridsearch.fit(x_train,y_train)
print(best_model.score(x_test,y_test))
print(best_model.best_index_)
print(best_model.best_params_)

# Summary 
# Fitting 5 folds for each of 48 candidates, totalling 240 fits
# Model Score: 76% which increases around 3% from previous
# Model Best Index: 6
# RF Classifier max depth: 3, min samples leaf: 3, n_estimators: 10

#%% Retrain Model for Deployment 

step_mms_rf = Pipeline([('Min Max Scaler', MinMaxScaler()),
                        ('RandomForestClassifier', RandomForestClassifier(n_estimators=10,
                                                                          min_samples_leaf=3,
                                                                          max_depth=3))])

step_mms_rf.fit(x_train,y_train)


# Model Saving 
with open(BEST_MODEL_PATH,'wb') as file:
    pickle.dump(step_ss_rf,file)

#%% Classification report,Confusion matrix,Accuracy

y_true = y_test
y_pred = best_model.predict(x_test)

print(classification_report(y_true,y_pred))
print(confusion_matrix(y_true,y_pred))
print(accuracy_score(y_true,y_pred))

# f1-score 0.76
# accuracy 0.76 































