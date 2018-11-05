# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 21:14:47 2018

@author: falcon1
"""
from __future__ import division, print_function, absolute_import

import tflearn
import tflearn.data_utils as du
import os
#import tflearn.datasets.mnist as mnist
#X, Y, testX, testY = mnist.load_data(one_hot=True)

model_path = "e:/repoes/AMPnet/model/"

def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("E:/Repoes/pythonProgram/Tensor/mnist_data", one_hot=True)
    X = mnist.train.images
    Y = mnist.train.labels
    testX = mnist.test.images
    testY = mnist.test.labels
    
    X = X.reshape([-1, 28, 28, 1])
    testX = testX.reshape([-1, 28, 28, 1])
    X, mean = du.featurewise_zero_center(X)
    testX = du.featurewise_zero_center(testX, mean)
    
    return X,Y,testX,testY

def resnet(input, n_class):
    # Building Residual Network
    net = tflearn.conv_2d(input, 64, 3, activation='relu', bias=False)
    # Residual blocks
    net = tflearn.residual_bottleneck(net, 3, 16, 64)
    net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 32, 128)
    net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 64, 256)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net_block = tflearn.global_avg_pool(net)
    # Regression
    net = tflearn.fully_connected(net_block, n_class, activation='softmax', restore=False)
    
    return net


# pre-training residual Network
def pretraining():
    X,Y,testX,testY = load_mnist()
    x = tflearn.input_data(shape=[None, 28, 28, 1], name='input')
    softmax = resnet(x, 10)
    regression = tflearn.regression(softmax, optimizer='adam',
                                    loss='categorical_crossentropy',
                                    learning_rate=0.001)
    
    model = tflearn.DNN(regression, checkpoint_path='resnet-finetuning',
                        max_checkpoints=3, tensorboard_verbose=2,
                        tensorboard_dir="./logs")
    model.fit(X, Y, n_epoch=1, validation_set=(testX, testY),
              show_metric=True, batch_size=256, run_id='resnet_finetuning')
    model.save(os.path.join(model_path,'model_resnet_mnist.tflearn'))

pretraining()
from prepareDataset import load_data
X,Y = load_data('e:/repoes/ampnet/data/img_60/','e:/repoes/ampnet/data/benchmark_60_Targets.json')
X = X.reshape((1378,28,28,1))
x = tflearn.input_data(shape=[None, 28, 28, 1], name='input')
softmax = resnet(x,6)
regression = tflearn.regression(softmax, optimizer='adam',
                                loss='categorical_crossentropy',
                                learning_rate=0.001)
model = tflearn.DNN(regression, checkpoint_path='./log/model_resnet_mnist',
                    max_checkpoints=3, tensorboard_verbose=2,
                    tensorboard_dir="./logs")
model_file = os.path.join(model_path, 'model_resnet_mnist.tflearn')
model.load(model_file, weights_only=True)
model.fit(X, Y, n_epoch=10, validation_set=0.1, shuffle=True,
          show_metric=True, batch_size=64, snapshot_epoch=False,
          snapshot_step=200, run_id='resnet-finetuning')
model.save(os.path.join(model_path,'resnet_finetuning.tflearn'))