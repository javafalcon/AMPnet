# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 14:17:07 2018

@author: falcon1
"""

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.layers.normalization import local_response_normalization
from tflearn.data_utils import shuffle, to_categorical

def convnet_mnist(num_class):
    net = input_data(shape=[None,28,28,1], name='input')
    net = conv_2d(net, 32,3, activation='relu', regularizer='L2')
    net = max_pool_2d(net,2)
    net = local_response_normalization(net)
    net = conv_2d(net,64,3, activation='relu', regularizer='L2')
    net = max_pool_2d(net,2)
    net = local_response_normalization(net)
    net = fully_connected(net, 128, activation='tanh')
    net = dropout(net, 0.8)
    net = fully_connected(net, 256, activation='tanh')
    net = dropout(net,0.8)
    net = fully_connected(net, num_class, activation='softmax', restore=False)
    
    return net

from prepareDataset import load_data
X,Y = load_data('e:/repoes/ampnet/data/img_60/', 'e:/repoes/ampnet/data/benchmark_60_Targets.json')
X = X.reshape((-1,28,28,1))
net = convnet_mnist(6)
net = regression(net, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')
model = tflearn.DNN(net, tensorboard_verbose=0)
model.load('e:/repoes/ampnet/model/convnet_mnist', weights_only=True)

model.fit({'input':X},{'target':Y}, n_epoch=20,
          validation_set=0.1, shuffle=True, batch_size=64,
          snapshot_step=100, show_metric=True, run_id='convet_mint_amp')
