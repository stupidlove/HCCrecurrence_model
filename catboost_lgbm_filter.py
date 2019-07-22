#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:11:01 2019

@author: delcher
"""


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn import metrics
# Plotly library
import plotly.plotly as py
import plotly.graph_objs as go
pd.set_option('display.max_columns', 500)
from sklearn.metrics import confusion_matrix
import catboost
sns.set()

df = pd.read_csv('HCCR0703.csv').drop(columns = 'ID')
df['Rec_class'] = 0
df.iloc[df[(df['RFT']<=720)& (df['RecS']==1)].index,-1] = 1 #6months recurrence
df.drop(df[(df['RecS']==0)&(df['SuvT']<720)].index,inplace=True)#death without recurrence;
                                                   #observation time less than 2 years
df.head()
num_variable = df.shape[1]
num_sample = df.shape[0]
plt.hist(df['Rec_class'])
df['Rec_class'].astype(str).describe()
# 0:2454 1:1759
plt.show()
all_features = df.columns

train_prop = 0.9
train = df.iloc[:int(train_prop*num_sample),:]
test = df.iloc[int(int(train_prop*num_sample)+1):,:]
target = train['Rec_class']
target_ = target.values
weight = np.ones(target.shape[0])
for i in range(target.shape[0]):
    if target_[i] == 1:
        weight[i] = 1.5
        
selected_features = all_features.drop(['SuvS', 'SuvT', 'RecS', 'RFT'])

categorical_columns = ['Gender', 'ASJRPD', 'Etiology', 'HCV', 'HBSAG', 'HBSAB',
       'HBEAG', 'HBEAB', 'HBCAG', 'ALBI2F', 'RM', 'RT', 'OBL', 'TN', 'MVI', 'EG',
       'TC2F', 'SN', 'LC', 'AVT', 'TACE']
categorical_columns = [c for c in categorical_columns if c not in ['Rec_class'] and c in selected_features]
features = [c for c in selected_features if c not in ['Rec_class']]

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))

threshold = 0.45
def two_sorted(x1,x2):
    #by x1
    x1 = np.array(x1,dtype = np.float)
    x2 = np.array(x2,dtype = np.float)
    sorted_indices = np.argsort(x1, axis=0)
    sorted_y1 = x1[sorted_indices]
    sorted_y2 = x2[sorted_indices]
    return sorted_y1,sorted_y2



def auc(true,pred_prob):
    true = np.array(true)
    pred_prob = np.array(pred_prob)
    pred_sorted,true_sorted = two_sorted(pred_prob,true)
    FPs = []
    TPs = []
    for i in pred_sorted:
        pred = np.where(pred_prob < i, 0 , 1)
        conf_matrix = confusion_matrix(y_true=true, y_pred=pred, 
                                               labels=[0,1])
        FP = conf_matrix[0,1]/(conf_matrix[0,0]+conf_matrix[0,1])
        TP = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
        FPs.append(FP)
        TPs.append(TP)
    plt.plot(FPs,TPs)
    plt.xlabel('FP')
    plt.ylabel('TP')
    plt.plot(np.linspace(0,1,100),np.linspace(0,1,100))
    plt.show()
    S = 0
    for i in range(len(FPs)-1):
        delta_x = FPs[i+1] - FPs[i]
        height = TPs[i]
        S += np.abs(delta_x*height)
    print("AUC : {}".format(S))
    return S

#=========catboost
clfs1 = []
iterate = 0
confusion_matrixes = []
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    iterate += 1
    trn_data = catboost.Pool(train.iloc[trn_idx][features],
                   label=target.iloc[trn_idx],
                   cat_features = categorical_columns,
                  weight= weight[trn_idx])
    val_data = catboost.Pool(train.iloc[val_idx][features],
                       label=target.iloc[val_idx],
                       cat_features = categorical_columns
                      )
    
    clf_cat = catboost.CatBoostClassifier(iterations=300,
                                      learning_rate=0.01,
                                      depth=8,
                                      eval_metric='Logloss',
                                       random_seed = 42,
                                       bagging_temperature = 0.2,
                                       od_type='Iter',
                                       metric_period = 50,
                                       od_wait=20)
    clf_cat.fit(trn_data,eval_set=val_data,use_best_model=True,
             verbose=50)
    '''
    oof[val_idx] = clf_cat.predict(train.iloc[val_idx][features], num_iteration=clf_cat.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf_cat.feature_importances_(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    '''

    
    current_pred = clf_cat.predict_proba(test[features])[:,1]
    '''
    while initial_idx < test.shape[0]:
        final_idx = min(initial_idx + chunk_size, test.shape[0])
        idx = range(initial_idx, final_idx)
        current_pred[idx] = clf.predict(test.iloc[idx][features], num_iteration=clf.best_iteration)
        initial_idx = final_idx
    '''
    class_pred = np.where(current_pred< threshold, 0, 1)
    class_true = test['Rec_class']
    conf_matrix = confusion_matrix(y_true=class_true, y_pred=class_pred, 
                                               labels=[0,1], sample_weight=None)
    confusion_matrixes.append(conf_matrix)
    speciality = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    sensitivity = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
    clfs1.append(clf_cat)
    print("sensitivity in iter:{} is : {}".format(iterate,sensitivity))
    print("speciality in iter:{} is : {}".format(iterate,speciality))

final_pred = np.mean([clfs1[i].predict_proba(test[features]) for i in range(5)],axis =0)[:,1]
class_pred = np.where(final_pred< threshold, 0, 1)
class_true = test['Rec_class'].values
conf_matrix = confusion_matrix(y_true=class_true, y_pred=class_pred, 
                                           labels=[0,1], sample_weight=None)
confusion_matrixes.append(conf_matrix)
speciality_all = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
sensitivity_all = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
print("final sensitivity : {}".format(sensitivity_all))
print("final speciality : {}".format(speciality_all))

AUC = auc(class_true,final_pred)