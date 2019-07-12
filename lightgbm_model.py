#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:41:10 2019

@author: delcher
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import warnings
import time

warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn import metrics
# Plotly library
import plotly.plotly as py
import plotly.graph_objs as go
pd.set_option('display.max_columns', 500)

df = pd.read_csv('HCCR0703.csv').drop(columns = 'ID')
df['Rec_class'] = 0
df.iloc[df[df['RFT']<=180].index,-1] = 1 #6months recurrence
df.iloc[df[(df['RFT']>180) & (df['RFT']<=360)].index,-1] = 2#6-12months recurrence
df.iloc[df[(df['RFT']>360) & (df['RFT']<=720)].index,-1] = 3#12-24months recurrence
df.iloc[df[(df['RFT']>720)].index,-1] = 4#more than 24months recurrence
df.iloc[df[(df['RecS']==0)].index,-1] = 0#no recurrence
df.drop(df[(df['RecS']==0)&(df['SuvS']==1)&(df['SuvT']<720)].index,inplace=True)#death without recurrence;
                                                   #observation time less than 2 years


df.head()
num_var = df.shape[1]
num_sample = df.shape[0]

RecS1 = df['Rec_class'].astype(str)
plt.hist(df['Rec_class'])
hist_rec = [go.Histogram(x=RecS1,
                         xbins=dict(start=-0.5,end=3.5,size=0.5),
                        opacity = 0.8)]
layout = go.Layout(title='RecS1',
                  xaxis=go.layout.XAxis(tickmode = 'array',
                                                tick0 = -0.5,
                                                tickvals = [0,1,2,3]),
                  yaxis=dict(title = 'number'),
                  )
fig = go.Figure(data=hist_rec,layout=layout)
py.iplot(fig,filename='hist_recs1')

all_features = df.columns
for feature in all_features:
    sns.distplot(df[feature],kde=False)
    plt.title(feature)
    plt.show()
    

train = df.iloc[:int(0.8*num_sample),:]
test = df.iloc[int(int(0.8*num_sample)+1):,:]
target = train['Rec_class']

param = {'num_leaves': 50,
         'min_data_in_leaf': 10, 
         'objective':'multiclass',
         'num_class':5,
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'multi_logloss',
         "lambda_l1": 0.1,
         "random_state": 133,
         "verbosity": -1}

selected_features = all_features.drop(['SuvS', 'SuvT', 'RecS', 'RFT'])
#all_features.drop(['Gender','Etiology','HCV','HBSAG','HBSAB',
                                       #'HBEAG','HBEAB','HBCAG','WBC','PLT','ALB','TBIL'])
categorical_columns = ['Gender', 'ASJRPD', 'Etiology', 'HCV', 'HBSAG', 'HBSAB',
       'HBEAG', 'HBEAB', 'HBCAG', 'ALBI2F', 'RM', 'RT', 'OBL', 'TN', 'MVI', 'EG',
       'TC2F', 'SN', 'LC', 'AVT', 'TACE']

categorical_columns = [c for c in categorical_columns if c not in ['Rec_class'] and c in selected_features]
features = [c for c in selected_features if c not in ['Rec_class']]

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros((len(train),5))
predictions = np.zeros(len(test))
start_time = time.time()
feature_importance_df = pd.DataFrame()

max_iter = 5

train_idx = [i for i in range(int(0.8*len(train)))]
val_idx = [i+int(0.8*num_sample) for i in range(int(0.2*len(train)))]
from sklearn.metrics import confusion_matrix
import lightgbm as lgb

confusion_matrixes = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    trn_data = lgb.Dataset(train.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
                           categorical_feature = categorical_columns
                          )
    val_data = lgb.Dataset(train.iloc[val_idx][features],
                           label=target.iloc[val_idx],
                           categorical_feature = categorical_columns
                          )

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds = 200)
    
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


    initial_idx = 0
    chunk_size = 100
    current_pred = clf.predict(test[features], num_iteration=clf.best_iteration)
    '''
    while initial_idx < test.shape[0]:
        final_idx = min(initial_idx + chunk_size, test.shape[0])
        idx = range(initial_idx, final_idx)
        current_pred[idx] = clf.predict(test.iloc[idx][features], num_iteration=clf.best_iteration)
        initial_idx = final_idx
    '''
    class_pred = np.argmax(current_pred,axis=1)
    class_true = test['Rec_class']
    confusion_matrixes.append(confusion_matrix(y_true=class_true, y_pred=class_pred, 
                                               labels=[0,1,2,3,4], sample_weight=None))

    '''
    print("time elapsed: {:<5.2}s".format((time.time() - start_time) / 3600))
    score[fold_] = metrics.roc_auc_score(target.iloc[val_idx], oof[val_idx])
    if fold_ == max_iter - 1: break
    '''
pred = pd.DataFrame(current_pred)
pred['class_true'] = test['Rec_class'].values
pred['class_pred'] = np.argmax(current_pred,axis=1)
current_pred
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)