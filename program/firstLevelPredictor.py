# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 20:10:42 2018

@author: falcon1
"""
from scipy.io import arff
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, auc, roc_curve
from libsvmutil import libsvm_grid_search
from svmutil import svm_train, svm_predict

def loadDataFromArff(filename):
    data,meta = arff.loadarff(filename)
    n = len(data)
    X = np.ndarray((n,256))
    Y = np.zeros(n)
    for i in range(n):
        d = data[i]
        for j in range(256):
            X[i][j] = float(d[j])
        Y[i] = int(d[-1])
    return X, Y

def randomForest(X_train, X_test, y_train):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    p = clf.predict(X_test)
    return p

def gaussionProcess(X_train, X_test, y_train): 
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF
    kernel = 1.0 * RBF(1.0)
    gpc = GaussianProcessClassifier(kernel=kernel,random_state=0).fit(X, y)
    gpc.score(X_train, y_train) 
    p = gpc.predict(X_test)
    return p

def libsvm(X_train, X_test, y_train, y_test):
    c,g,acc = libsvm_grid_search(X_train, y_train)
    print("acc={}".format(acc))
    param = ['-c', str(c), '-g', str(g)]
    
    model = svm_train(y_train, X_train, " ".join(param))
    pred_labels, pred_acc, pred_val = svm_predict(y_test, X_test, model)
    return pred_labels
    
filename = 'e:/repoes/ampnet/amp_and_notamp_alnex.arff'
X, y = loadDataFromArff(filename)

# 留一法
y_pred = np.zeros(1600)
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    print("\r In predicting {}".format(test_index))
    X_train, X_test = X[train_index], [X[test_index]]
    y_train, y_test = y[train_index], [y[test_index]]
    #y_pred[test_index] = gaussionProcess(X_train, X_test, y_train)
    #y_pred[test_index] = randomForest(X_train, X_test, y_train)
    y_pred[test_index] = libsvm(X_train, X_test, y_train, y_test)
accuracy = accuracy_score(y, y_pred)
fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=1) 
area = auc(fpr, tpr)

print("accuracy={}, auc={}".format(accuracy, area))   
    
    