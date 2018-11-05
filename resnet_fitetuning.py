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

def build_resnet_mnist():
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

    # Building Residual Network
    net = tflearn.input_data(shape=[None, 28, 28, 1])
    net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)
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
    net = tflearn.fully_connected(net_block, 10, activation='softmax')
    net = tflearn.regression(net, optimizer='momentum',
                             loss='categorical_crossentropy',
                             learning_rate=0.1)
    # Training
    model = tflearn.DNN(net, checkpoint_path='./log/model_resnet_mnist',
                        max_checkpoints=10, tensorboard_verbose=0)
    model.fit(X, Y, n_epoch=1, validation_set=(testX, testY),
              show_metric=True, batch_size=256, run_id='resnet_mnist')
    model.save(os.path.join(model_path,'model_resnet_mnist.tflearn'))
    
def resnet(num_class):
    net = tflearn.input_data(shape=[None, 28, 28, 1])
    net = tflearn.conv_2d(net, 64, 3, activation='relu', bias=False)
    # Residual blocks
    net = tflearn.residual_bottleneck(net, 3, 16, 64)
    net = tflearn.residual_bottleneck(net, 1, 32, 128, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 32, 128)
    net = tflearn.residual_bottleneck(net, 1, 64, 256, downsample=True)
    net = tflearn.residual_bottleneck(net, 2, 64, 256)
    net = tflearn.batch_normalization(net)
    net = tflearn.activation(net, 'relu')
    net = tflearn.global_avg_pool(net)
    # Regression
    net = tflearn.fully_connected(net, num_class, activation='softmax', restore=False)
    
    return net

from prepareDataset import load_data
#build_resnet_mnist()
X,Y = load_data('e:/repoes/ampnet/data/img_60/','e:/repoes/ampnet/data/benchmark_60_Targets.json')
X = X.reshape((1378,28,28,1))
softmax = resnet(6)
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