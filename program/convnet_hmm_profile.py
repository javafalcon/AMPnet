# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:59:36 2018

@author: Administrator
"""
import json
import numpy as np
import tflearn
import tensorflow as tf
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from sklearn.model_selection import LeaveOneOut, KFold
from sklearn.metrics import accuracy_score, auc, roc_curve
# 每个序列取前50个氨基酸（共50*20=1000个特征），如果序列长度不足50，则补0
# 如果序列长度大于50，则截取前50个氨基酸
def load_hmm_prof():
    files = ['e:/repoes/ampnet/data/benchmark/AMPs_50_hmm_profil.json',
         'e:/repoes/ampnet/data/benchmark/notAMPs_50_hmm_profil.json']
    N = 1000
    X = np.ndarray((1600,N))
    y = np.ones(1600)
    y[800:] = 0
    k = 0
    for f in files:
        fr = open(f,'r')
        p = json.load(fr)
        for key in p.keys():
            ary = p[key]
            c = len(ary)
            if c < N:
                X[k][:c] = ary
                X[k][c:] = 0
            elif c == N:
                X[k] = ary
            else:
                X[k] = ary[:N]
            k += 1
        fr.close()
        
    return X, y

def net(X_train, y_train, X_test, y_test):
    # Real-time data preprocessing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    
    # Real-time data augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_rotation(max_angle=25.)
    
    # Convolutional network building
    network = input_data(shape=[None, 50, 20, 1],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = dropout(network, 0.75)
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2)
    network = dropout(network, 0.5)
    network = fully_connected(network, 512, activation='relu')
    network = dropout(network, 0.5)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam',
                         loss='categorical_crossentropy',
                         learning_rate=0.001)
    
    # Train using classifier
    model = tflearn.DNN(network, tensorboard_verbose=0)
    model.fit(X_train, y_train, n_epoch=100, shuffle=True, validation_set=0.2,
              show_metric=True, batch_size=32, run_id='cifar10_cnn_mnist')
    return model
 

def jackknife_test(X, y):
    y_pred = np.zeros(1600)
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        print("\r In predicting {}".format(test_index))
        X_train, X_test = X[train_index], [X[test_index]]
        y_train, y_test = y[train_index], [y[test_index]]
        
        y_pred[test_index] = net(X_train, y_train, X_test, y_test)
    return y_pred


def cross_validate(X,y,n_splits=3):
    y_pred = np.zeros([1600,2])
    kf = KFold(n_splits=3)
    for train_index, test_index in kf.split(X):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        tf.reset_default_graph()
        model = net(X_train, y_train, X_test, y_test)
        for (xx,k) in zip(X_test, test_index):
            y = model.predict_label([xx])
            #print(y)
            y_pred[k] = y
    return y_pred

 
X,y = load_hmm_prof()
X = X.reshape([-1,50,20,1])
y = to_categorical(y,2)  
X,y = shuffle(X,y) 
#y_pred = jackknife_test(X,y)
y_pred = cross_validate(X,y)

accuracy = accuracy_score(y, y_pred)
fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=1) 
area = auc(fpr, tpr)             
