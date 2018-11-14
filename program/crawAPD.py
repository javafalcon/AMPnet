# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:38:25 2018

@author: Administrator
"""

import requests
import re
from bs4 import BeautifulSoup
import json
from Bio import SeqIO

types = ['antiA','antiB','antiV','antiH','antiF','antiP','antiD','antiT','antiC','antiO','taxis','antiI','antiE','antiS','surface','antiW']
qid = 'http://aps.unmc.edu/AP/database/query_output.php?ID='
APDURL = 'http://aps.unmc.edu'

def getHTMLText(url,coding='gbk'):
    try:
        r = requests.get(url,timeout=30)
        #print(r)
        r.raise_for_status()
        r.encoding = coding
        return r.text
    except:
        return ""

def extractSequence(url):
    s = ''
    print(url)
    html = getHTMLText(url)
    soup = BeautifulSoup(html, "html.parser")
    trlist = soup.find_all('tr')
    
    found = False
    for tr in trlist:
        tdlist = tr.find_all('td')
        for td in tdlist:
            if found:
                s = td.string
                break
            if td.string == 'Sequence:':
                found = True
            
        if found:
            break
    
    return s
    

def downAPD(preurl, rootpage, filename):
    AMPList = []
    url = preurl + rootpage
    while True:
        print(url)
        html = getHTMLText(url)
        soup = BeautifulSoup(html,"html.parser")
        data = soup.find_all('a',{'target':'_blank'})
        for a in data:
            AMPList.append(a.string)
            
        nextpage = soup.find('a', string=re.compile('Next page'))
        if nextpage is None:
            break
        else:
            url = preurl + nextpage.attrs['href']
     
    pset = set(AMPList)
    print('{} has {} sequences'.format(filename, len(pset)))
    
    with open(filename,'w') as ws:
        ws.writelines(pset)

def downAMPsFromAPD():
    for a in types:
        rootpage = '/AP/database/'+a+'.php'
        downAPD(APDURL,rootpage, 'E:\\Repoes\\AMPnet\\data\\'+a+'.txt')
        
def createAMPsBenchmark():
    root = 'E:\\Repoes\\AMPnet\\data\\'
    AMPSequs = {}
    AMPTargs = {}
    
    # 获得所有APD序列，以字典id:sequence的格式存储在json文件中
    data = {}
    for seq_record in SeqIO.parse('E:\\Repoes\\AMPnet\\data\\APD_AMPs.fa', 'fasta'):
        data[seq_record.id] = seq_record.seq
    
    for a in types:
        filename = root+a + '.txt'
        fr = open(filename, 'r')
        s = fr.read()
        fr.close()
        pids = s.split()
        for key in pids:
            print(key)
            if AMPSequs.get(key) is None:
                if data.get(key) is None:
                    seq = extractSequence(qid+key[2:])
                else:
                    seq = str(data[key])
                AMPSequs[key] = seq              
   
    f1 = open('E:\\Repoes\\AMPnet\\data\\AMPSequence.json','w')   
    json.dump(AMPSequs,f1,indent=4)
    f1.close()

    # 获得所有APD序列的功能活性标签，以字典id:[fun1,fun2,...]的形式存储在json文件中
    for a in types:
        filename = root + a + '.txt'
        fr = open(filename,'r')
        s = fr.read()
        fr.close()
        pids = s.split()
        for key in pids:
            ls = AMPTargs.get(key,[])
            ls.append(a)
            AMPTargs[key] = ls
            
    f2 = open('E:\\Repoes\\AMPnet\\data\\AMPTarget.json','w')
    json.dump(AMPTargs,f2,indent=4)
    f2.close()

def getAMPs():
    f1 = open('E:\\Repoes\\AMPnet\\data\\AMPSequence.json','r')
    AMPSequs = json.load(f1)   
    f1.close()

    f2 = open('E:\\Repoes\\AMPnet\\data\\AMPTarget.json','r')
    AMPTargs = json.load(f2)
    f2.close()
   
    return (AMPSequs, AMPTargs)  

def writeFastaFile():
    AMPSequs, AMPTargs = getAMPs()  
    for a in types:
        fw = open('E:\\Repoes\\AMPnet\\data\\' + a + '.fasta','w')
        j=0
        for key in AMPTargs.keys():
            if a in AMPTargs[key]:
                fw.write(">"+key+"\n")
                fw.write(AMPSequs[key]+"\n")
                j += 1
        fw.close()
        print(a,j)

 
def main():
    #extractSequence('http://aps.unmc.edu/AP/database/query_output.php?ID=00001')
    writeFastaFile()
          
if __name__ == '__main__':
    main()
# antiB_30:950; antiV:182; antiH:109; antiF_30:441; antiP:103;antiC: 217 ; 