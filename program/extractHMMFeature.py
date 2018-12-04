# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 07:50:26 2018

@author: Administrator
"""
#import numpy as np
import json


def HMMProf(chk,out):
    fo = open(chk,'r')
    prot = {}
    key = ''
    val = []
    k, n_seq = 0, 0
    rf = False
    
    for line in fo.readlines():
        line = line.replace("\n","")
        line = line.strip()
    
        if 'NAME' in line:
            key = line.split('  ')[-1]
        elif 'COMPO' in line:
            k = k + 1
            rf = True
        elif '//' in line:
            #m = np.array(val)
            #m = m.reshape((n_seq,20))
            prot[key] =val
            k, n_seq = 0, 0
            rf = False
            val = []
        if rf:    
            if k%3 == 1:
                n_seq += 1
                ls = line.split('  ')
                #print(ls)
                for i in range(1,21):
                    val.append(float(ls[i]))
                
            k = k + 1
    fo.close()
    
    f = open(out,'w')
    json.dump(prot,f,indent=4)
    f.close()            
    
HMMProf('e:/repoes/ampnet/data/benchmark/wpamp-1.hmm','E:\\Repoes\\AMPnet\\data\\benchmark\\wpAMPs_hmm_profil.json')
HMMProf('e:/repoes/ampnet/data/benchmark/wpnotamp-1.hmm','e:/Repoes/ampnet/data/benchmark/wpnotAMPs_hmm_profil.json')