# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 19:26:47 2018

@author: falcon1
"""
from CA import createCAImageFileOfSeq
from Bio import SeqIO
import json
import os

types = ['antiA','antiB','antiV','antiH','antiF','antiP','antiD','antiT','antiC','antiO','taxis','antiI','antiE','antiS','surface','antiW']

def getAMPs():
    f1 = open('E:\\Repoes\\AMPnet\\data\\AMPSequence.json','r')
    AMPSequs = json.load(f1)   
    f1.close()

    f2 = open('E:\\Repoes\\AMPnet\\data\\AMPTarget.json','r')
    AMPTargs = json.load(f2)
    f2.close()
   
    return (AMPSequs, AMPTargs)  

def CAImages():
    #files=['antiA.fasta','antiB.fa','antiC.fa','antiF.fa','antiH.fa']
    AMPSequs, AMPTargs = getAMPs()
    files = ['antiC.fasta']
    for f in files:
        k = f.index('.')
        filepath = 'E:\\Repoes\\AMPnet\\data\\img\\' + f[0:k]
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            
        for seq_record in SeqIO.parse('E:\\Repoes\\AMPnet\\data\\' + f, 'fasta'):
            seq = AMPSequs[seq_record.id]
            imgfileName = filepath + '\\' + seq_record.id + '.jpg'
            createCAImageFileOfSeq(seq,84,0,300,imgfileName)
            
CAImages()
            