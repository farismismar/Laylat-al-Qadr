#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 00:20:37 2017

@author: farismismar
"""

'''
Laylat al-Qadr Predictor
'''

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

random_state = seed = 123

dataset = pd.read_csv('dataset.csv') # The file details are TBD

# If we use 32 bit pictures and 4 MP to capture the sun exactly at sun rise
# We therefore need 16 million features
# Idea is to perform PCA and reduce number of features to say 50k.

'''
Date | Year | Temperature at Sunrise | Humidity at Sunrise | Sun at sunrise pixels one byte per column 0 ... 49999.
19 | 1438 | 76 | .20
20 | 1438 | 77 | .40
21 | 1438 |    
...| ...
30 | 1438 | 79 | .22
'''

train, test = train_test_split(dataset, random_state=random_state, test_size=0.3)

X_train = train.drop(['decision'], axis=1)
y_train = train['decision']
X_test = test.drop(['decision'], axis=1)
y_test = test['decision']


classifier = xgb.XGBClassifier(seed=seed, silent=False, colsample_bytree=0.7,
                             learning_rate=0.05, n_estimators = 1000)

#classifier.get_params().keys()

# Hyperparameters
alphas = np.linspace(0,1,4)
lambdas = np.linspace(0,1,4)
depths=[2,8]
objectives = ['binary:logistic', 'reg:linear']

hyperparameters = {'reg_alpha': alphas, 'reg_lambda': lambdas, 'objective': objectives, 'max_depth': depths}

gs_xgb = GridSearchCV(classifier, hyperparameters, scoring='roc_auc', cv=5) # 5-fold crossvalidation

# Rebalance the training data as this dataset is likely to be highly imbalanced.

# Rescale all features
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

gs_xgb.fit(X_train, y_train)

# This is the best model based on the scoring metric
best_model_xgb = gs_xgb.best_params_
print(best_model_xgb)

clf = gs_xgb.best_estimator_

clf.fit(X_train, y_train, eval_metric='auc')

y_score_xgb = clf.predict_proba(X_test)
y_hat_xgb = clf.predict(X_test)

# Compute ROC curve and ROC area
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_score_xgb[:,1])
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(13,8))
lw = 2

plt.plot(fpr_xgb, tpr_xgb,
     lw=lw, label="ROC curve (AUC = {:.6f})".format(roc_auc_xgb))

plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('output.pdf', format='pdf')
plt.show()


