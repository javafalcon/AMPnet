# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:59:36 2018

@author: Administrator
"""
import json
import numpy as np
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
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

X,y = load_hmm_prof()
y = to_categorical(y,2)                

                