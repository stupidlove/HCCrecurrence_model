#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:06:02 2019

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

sns.set()

df = pd.read_csv('HCCR0703.csv').drop(columns = 'ID')
df['Rec_class'] = 0
df.iloc[df[(df['RFT']<=720)& (df['RecS']==1)].index,-1] = 1 #6months recurrence
df.drop(df[(df['RecS']==0)&(df['SuvT']<720)].index,inplace=True)#death without recurrence;
                                                   #observation time less than 2 years

#df['TD'] = np.log(df['TD'].values)
#df['TD'].hist(bins=30)
df.head()
num_var = df.shape[1]
num_sample = df.shape[0]


plt.hist(df['Rec_class'])
df['Rec_class'].astype(str).describe()
# 0:2454 1:1759
plt.show()
all_features = df.columns
'''
for feature in all_features:
    sns.distplot(df[feature],kde=False)
    plt.title(feature)
    plt.show()
'''

#start    
train_prop = 0.9
train = df.iloc[:int(train_prop*num_sample),:]
test = df.iloc[int(int(train_prop*num_sample)+1):,:]
test['Rec_class'].hist()
plt.show()
target = train['Rec_class']
target_ = target.values
weight = np.ones(target.shape[0])
for i in range(target.shape[0]):
    if target_[i] == 1:
        weight[i] = 1.5

param = {'num_leaves': 40,
         'min_data_in_leaf': 40, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.01,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'binary_logloss',
         "lambda_l1": 0.1,
         "random_state": 133,
         "verbosity": -1
         }

selected_features = all_features.drop(['SuvS', 'SuvT', 'RecS', 'RFT'])

#all_features.drop(['Gender','Etiology','HCV','HBSAG','HBSAB',
                                       #'HBEAG','HBEAB','HBCAG','WBC','PLT','ALB','TBIL'])
categorical_columns = ['Gender', 'ASJRPD', 'Etiology', 'HCV', 'HBSAG', 'HBSAB',
       'HBEAG', 'HBEAB', 'HBCAG', 'ALBI2F', 'RM', 'RT', 'OBL', 'TN', 'MVI', 'EG',
       'TC2F', 'SN', 'LC', 'AVT', 'TACE']

categorical_columns = [c for c in categorical_columns if c not in ['Rec_class'] and c in selected_features]
features = [c for c in selected_features if c not in ['Rec_class']]

folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train))
predictions = np.zeros(len(test))
start_time = time.time()
feature_importance_df = pd.DataFrame()



train_idx = [i for i in range(int(0.9*len(train)))]
val_idx = [i+int(0.1*num_sample) for i in range(int(0.2*len(train)))]

from sklearn.metrics import confusion_matrix
import lightgbm as lgb

confusion_matrixes = []
clfs = []
iterate = 0
threshold = 0.45

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
    print("fold n°{}".format(fold_))
    iterate += 1
    trn_data = lgb.Dataset(train.iloc[trn_idx][features],
                           label=target.iloc[trn_idx],
                           categorical_feature = categorical_columns,
                          weight= weight[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx][features],
                           label=target.iloc[val_idx],
                           categorical_feature = categorical_columns
                          )

    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval = 100,
                    early_stopping_rounds = 300)
    
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


    initial_idx = 0
    current_pred = clf.predict(test[features], num_iteration=clf.best_iteration)
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
    clfs.append(clf)
    print("sensitivity in iter:{} is : {}".format(iterate,sensitivity))
    print("speciality in iter:{} is : {}".format(iterate,speciality))
    '''
    print("time elapsed: {:<5.2}s".format((time.time() - start_time) / 3600))
    score[fold_] = metrics.roc_auc_score(target.iloc[val_idx], oof[val_idx])
    if fold_ == max_iter - 1: break
   '''


#calculate AUC
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


final_pred = np.mean([clfs[i].predict(test[features], 
                        num_iteration=clf.best_iteration) for i in range(5)],axis =0)
class_pred = np.where(final_pred< threshold, 0, 1)
class_true = test['Rec_class'].values
conf_matrix = confusion_matrix(y_true=class_true, y_pred=class_pred, 
                                           labels=[0,1], sample_weight=None)
confusion_matrixes.append(conf_matrix)
speciality = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
sensitivity = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
print("final sensitivity : {}".format(sensitivity))
print("final speciality : {}".format(speciality))

auc(class_true,final_pred)




#================================过滤==========
pred_great_prob = []
true_class_great_prob = []
great_prob = 0.7
for i in range(final_pred.shape[0]):
    if (final_pred[i] <= 1-great_prob+0.05 )or(final_pred[i] > great_prob-0.05):
        pred_great_prob.append(final_pred[i])
        true_class_great_prob.append(class_true[i])
pred_great_prob = np.array(pred_great_prob)
true_class_great_prob = np.array(true_class_great_prob)
pred_class_great_prob = np.where(pred_great_prob < threshold, 0, 1)
conf_matrix = confusion_matrix(y_true=true_class_great_prob, 
                               y_pred=pred_class_great_prob, 
                               labels=[0,1], sample_weight=None)
speciality = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
sensitivity = conf_matrix[1,1]/(conf_matrix[1,0]+conf_matrix[1,1])
print("sensitivity with great probability : {}".format(sensitivity))
print("speciality with great probability : {}".format(speciality))


train_pred = np.mean([clfs[i].predict(train[features], 
                        num_iteration=clf.best_iteration) for i in range(5)],axis =0)
train_small_prob = []
train_class_small_prob = []
train_class_true = train['RecS'].values
for i in range(train_pred.shape[0]):
    if (train_pred[i] >= 1-great_prob+0.05 )and(train_pred[i] < great_prob-0.05):
        train_small_prob.append(train_pred[i])
        train_class_small_prob.append(train_class_true[i])
train_small_prob = np.array(train_small_prob)
train_class_small_prob = np.array(train_class_small_prob)
train_pred_class_small_prob = np.where(train_small_prob < threshold, 0, 1)
confusion_matrix(y_true=train_class_small_prob, 
                               y_pred=train_pred_class_small_prob, 
                               labels=[0,1], sample_weight=None)
plt.hist(train_class_small_prob)



    
X = df.iloc[:,:-1]
y = df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)