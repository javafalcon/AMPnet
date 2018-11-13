# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 20:10:42 2018

@author: falcon1
"""
from scipy.io import arff
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve

def loadDataFromArff(filename):
    data,meta = arff.loadarff(filename)
    n = len(data)
    X = np.ndarray((n,512))
    Y = np.zeros(n)
    for i in range(n):
        d = data[i]
        for j in range(512):
            X[i][j] = float(d[j])
        Y[i] = int(d[-1])
    return X, Y

filename = 'e:/repoes/ampnet/amp_and_notamp.arff'
X, y = loadDataFromArff(filename)

# 留一法
y_pred = np.zeros(1600)
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,
                              random_state=0)
    clf.fit(X_train, y_train)
    y_pred[test_index] = clf.predict(X_test)
    
accuracy = accuracy_score(y, y_pred)
fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=0) 
area = auc(fpr, tpr)

print("accuracy={}, auc={}".format(accuracy, area))   
    
    