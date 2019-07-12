#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 20:10:10 2019

@author: delcher
"""

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

all_variables = ['ID', 'Age', 'Gender', 'ASJRPD', 'Etiology', 'HCV', 'HBSAG', 'HBSAB',
       'HBEAG', 'HBEAB', 'HBCAG', 'AFP', 'HBVDNA', 'WBC', 'PLT', 'ALB', 'TBIL',
       'GGT', 'ALP', 'ALBI2F', 'RM', 'RT', 'OBL', 'TD', 'TN', 'MVI', 'EG',
       'TC2F', 'SN', 'LC', 'AVT', 'TACE', 'RecS6F', 'RecS1']

class_variables = ['Gender', 'ASJRPD', 'Etiology', 'HCV', 'HBSAG', 'HBSAB',
       'HBEAG', 'HBEAB', 'HBCAG', 'TBIL', 'ALBI2F', 'RM', 'RT', 'OBL','TN', 'MVI', 'EG',
       'TC2F', 'SN', 'LC', 'AVT', 'TACE', 'RecS6F']
removed_variables = ['AFP','TD','TN','MVI','ASJRPD']

for i in removed_variables:
    all_variables.remove(i)
    if i in class_variables:
        class_variables.remove(i)
df = pd.read_csv('HCC.csv')
df = df[all_variables]
#df.drop('ID',axis=1,inplace=True)

ID = pd.read_csv('HCC508.csv')['no'].astype(str).values
df = df[df['ID'].isin(ID)]
df['RecS1'].describe()
df[class_variables] = df[class_variables].astype(str)
df.iloc[:,-1] = df.iloc[:,-1].replace((2,3),0)

df_onehot = pd.get_dummies(df)
columns = class_variables
columns_uniq = {}
for i in columns[:-1]:
    columns_uniq[i] = np.unique(df[i])
    
name = []
for key,value in columns_uniq.items():
    for j in value[:-1]:
        name.append(key+'_'+str(j))
    
from sklearn.linear_model import LogisticRegression
X = df_onehot[name]
y = df.iloc[:,-1]
lgr = LogisticRegression()
lgr.fit(X,y)
coeff = lgr.coef_
exp_coeff = np.exp(coeff)

from sklearn.model_selection import cross_val_score  
scores = cross_val_score(lgr,X,y,cv=5) 
np.mean(scores)

precision = cross_val_score(lgr,X,y,cv=5,scoring='precision') 
np.mean(precision)

recalls = cross_val_score(lgr,X,y,cv=5,scoring='recall') 
np.mean(recalls) 

y_true = df['RecS1']
y_pred = lgr.predict(X)
import sklearn
sklearn.metrics.confusion_matrix(y_true, y_pred, labels=None, sample_weight=None)
93/(93+18)