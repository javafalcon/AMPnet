# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:26:47 2018

@author: falcon1
"""
from CA import generateCAImageOfSeq
from Bio import SeqIO
import json
import os
import numpy as np
from PIL import Image

files = ['antiB_60', 'antiC_60','antiF_60','antiH_60','antiP_60','antiV_60']

def getAMPs():
    f1 = open('E:\\Repoes\\AMPnet\\data\\AMPSequence.json','r')
    AMPSequs = json.load(f1)   
    f1.close()

    f2 = open('E:\\Repoes\\AMPnet\\data\\AMPTarget.json','r')
    AMPTargs = json.load(f2)
    f2.close()
   
    return (AMPSequs, AMPTargs)  

def CAImages():
    AMPSequs, AMPTargs = getAMPs()

    for f in files:
        filepath = 'E:\\Repoes\\AMPnet\\data\\img\\' + f
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            
        for seq_record in SeqIO.parse('E:\\Repoes\\AMPnet\\data\\' + f, 'fasta'):
            seq = AMPSequs[seq_record.id]
            imgfileName = filepath + '\\' + seq_record.id + '.jpg'
            img = generateCAImageOfSeq(seq,84,0,300)
            img = img.resize((28,28))
            img.save(imgfileName,'jpeg')
            
            
def saveSeqsCAImages(prots:dict):
    filepath = 'E:\\Repoes\\AMPnet\\data\\img_60\\'
    for key in prots.keys():
        seq = prots[key]
        imgfileName = filepath + key + '.jpg'
        img = generateCAImageOfSeq(seq, 84, 0, 300)
        img = img.resize((28,28))
        img.save(imgfileName,'jpeg')  

          
def getBenchmarkDataset():
    AMPSequs, AMPTargs = getAMPs()
    benchmarkSeqs={}
    benchmarkTars={}
    keyList=[]
    for f in files:
        for seq_record in SeqIO.parse('E:\\Repoes\\AMPnet\\data\\' + f, 'fasta'):
            keyList.append(seq_record.id)
    
    keySet = set(keyList)
    for key in keySet:
        benchmarkTars[key] = AMPTargs[key]
        benchmarkSeqs[key] = AMPSequs[key]
    
    f = open('E:\\Repoes\\AMPnet\\data\\benchmark_60_Targets.json','w')
    json.dump(benchmarkTars,f,indent=4)
    f.close()
    
    f1 = open('E:\\Repoes\\AMPnet\\data\\benchmark_60_Sequence.json','w')   
    json.dump(benchmarkSeqs,f1,indent=4)
    f1.close()
    
    saveSeqsCAImages(benchmarkSeqs)

def load_data(imgsdir, targsfile):
    f = open(targsfile,'r')
    tars = json.load(f)
    f.close()
    Labels = ['antiB', 'antiC','antiF','antiH','antiP','antiV']
    files = os.listdir(imgsdir)
    N = len(files)
    X = np.ndarray((N,784),dtype=np.float32)
    Y = np.ndarray((N,6),dtype=np.float64)
    
    i = 0
    for file in files:
        k = file.index('.')
        key = file[:k]
        img = Image.open(imgsdir+file,"r")
        m = np.array(img)
        m = m.reshape((1,784))
        X[i] = m
        
        y = np.zeros((1,6))
        v = tars.get(key)
        for j in range(6):
            if Labels[j] in v: 
                y[0][j] = 1
        Y[i] = y
        i = i + 1
    return X, Y

#getBenchmarkDataset()
X,Y = load_data('e:/repoes/ampnet/data/img_60/','e:/repoes/ampnet/data/benchmark_60_Targets.json')
        
    
            
