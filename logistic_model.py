#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:33:10 2019

@author: delcher
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv('HCC508.csv')
df.drop('no',axis=1,inplace=True)
selected_variables = ['3','10','22','23','24','31']
df = df[selected_variables]
df.iloc[:,:-1] = df.iloc[:,:-1].astype(str)
df_onehot = pd.get_dummies(df)

columns = df.columns
columns_uniq = {}
for i in columns[:-1]:
    columns_uniq[i] = np.unique(df[i])

name = []
for key,value in columns_uniq.items():
    for j in value[:-1]:
        name.append(key+'_'+str(j))
    
from sklearn.linear_model import LogisticRegression
X = df_onehot[name]
y = df['31']
lgr = LogisticRegression()
lgr.fit(X,y)
coeff = lgr.coef_
exp_coeff = np.exp(coeff)

from sklearn.model_selection import cross_val_score  
scores = cross_val_score(lgr,X,y,cv=5,scoring='accuracy') 
np.mean(scores)

precision = cross_val_score(lgr,X,y,cv=5,scoring='precision') 
np.mean(precision)

recalls = cross_val_score(lgr,X,y,cv=5,scoring='recall') 
np.mean(recalls) 

y_true = df['31']
y_pred = lgr.predict(X)
import sklearn
sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
