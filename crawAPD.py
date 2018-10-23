# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:38:25 2018

@author: Administrator
"""

import requests
import re
from bs4 import BeautifulSoup
import json


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
    
def fullProteinsList(soup, plist):
    data = soup.find_all('a',{'target':'_blank'})
    for a in data:
        plist.append(a.string)


def downAPD(preurl, rootpage, filename):
    AMPList = []
    url = preurl + rootpage
    while True:
        print(url)
        html = getHTMLText(url)
        soup = BeautifulSoup(html,"html.parser")
        fullProteinsList(soup, AMPList)
        nextpage = soup.find('a', string=re.compile('Next page'))
        if nextpage is None:
            break
        else:
            url = preurl + nextpage.attrs['href']
     
    pset = set(AMPList)
    with open(filename,'w') as ws:
        ws.writelines(pset)

def downAMPsFromAPD():
    for a in types:
        rootpage = '/AP/database/'+a+'.php'
        downAPD(APDURL,rootpage, a+'.txt')
        
def createAMPsBenchmark():
    root = 'E:\\Repoes\\AMPnet\\data\\'
    AMPSequs = {}
    AMPTargs = {}
    for a in types:
        filename = root+a + '.txt'
        fr = open(filename, 'r')
        s = fr.read()
        fr.close()
        pids = s.split()
        for key in pids:
            print(key)
            if AMPSequs.get(key) is None:
                seq = extractSequence(qid+key[2:])
                AMPSequs[key] = seq
                AMPTargs[key] = [a]
            else:
                AMPTargs[key] = AMPTargs.get(key).append(a)
                
        
    f1 = open('AMPSequence.fasta','w')   
    json.dump(AMPSequs,f1,indent=4)
    f1.close()

    f2 = open('AMPTarget.txt','w')
    json.dump(AMPTargs,f2,indent=4)
    f2.close()

def getAMPs():
    f1 = open('AMPSequence.fasta','r')
    AMPSequs = json.load(f1)   
    f1.close()

    f2 = open('AMPTarget.txt','r')
    AMPTargs = json.load(f2)
    f2.close()
   
    return (AMPSequs, AMPTargs)  
'''   
def main():
    #extractSequence('http://aps.unmc.edu/AP/database/query_output.php?ID=00001')
    createAMPsBenchmark()
    AMPSequs, AMPTargs = getAMPs()  
          
if __name__ == '__main__':
    main()
 '''   
createAMPsBenchmark()
AMPSequs, AMPTargs = getAMPs()  