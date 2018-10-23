# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:38:25 2018

@author: Administrator
"""

import requests
import re
from bs4 import BeautifulSoup
from Bio import SeqIO

types = ['antiA','antiB','antiV','antiH','antiF','antiP','antiD','antiT','antiC','antiO','taxis','antiI','antiE','antiS','surface','antiW']
qid = 'http://aps.unmc.edu/AP/database/query_output.php?ID='

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
    
    print(s)
    
def fullProteinsList(soup, plist):
    data = soup.find_all('a',{'target':'_blank'})
    for a in data:
        pid = a.string # for examp: AP00001
        pidurl = qid + pid[2:]
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
    amps = ['antiA','antiB','antiV','antiH','antiF','antiP','antiD','antiT','antiC','antiO','taxis','antiI','antiE','antiS','surface','antiW']
    for a in amps:
        rootpage = '/AP/database/'+a+'.php'
        downAPD('http://aps.unmc.edu',rootpage, a+'.txt')
        
def targetAMPs():
    root = 'E:\\Repoes\\AMPnet\\data\\'
    AMPSeqs = {}
    for seq_record in SeqIO.parse(root+'APD_AMPs.fa', 'fasta'):
        AMPSeqs[seq_record.id] = seq_record.seq
    
    AMPTars = {}
    amps = ['antiA','antiB','antiV','antiH','antiF','antiP','antiD','antiT','antiC','antiO','taxis','antiI','antiE','antiS','surface','antiW']
    for a in amps:
        filename = root+a + '.txt'
        fr = open(filename, 'r')
        s = fr.read()
        pids = s.split()
        for k in pids:
            t = AMPTars.get(k)
            if t is None:
                t = []
            AMPTars[k] = t.append(a)
            
def main():
    extractSequence('http://aps.unmc.edu/AP/database/query_output.php?ID=00001')
            
if __name__ == '__main__':
    main()
    