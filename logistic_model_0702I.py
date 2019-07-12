#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:18:12 2019

@author: delcher
"""

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
import lightgbm as lgb
sns.set()

all_variables = ['ALP','ALBI2F', 'RT',  'OBL', 'HBSAB','RecS1']

class_variables = ['ALP','ALBI2F', 'RT',  'OBL', 'HBSAB','RecS1']
df = pd.read_excel('HCC508I.xlsx')
df = df[all_variables]

df[class_variables] = df[class_variables].astype(str)
df_onehot = pd.get_dummies(df)
columns = class_variables
columns_uniq = {}
for i in columns[:-1]:
    columns_uniq[i] = np.unique(df[i])
    
name = []
for key,value in columns_uniq.items():
    for j in value[:-1]:
        name.append(key+'_'+str(j))
        
from sklearn.model_selection import train_test_split
X = df_onehot[name]
y = df.iloc[:,-1]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(X_train,y_train)
coeff = lgr.coef_
exp_coeff = np.exp(coeff)

y_true = y_test.values
y_pred = lgr.predict(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_true, y_pred, labels=['1','0'], sample_weight=None)
