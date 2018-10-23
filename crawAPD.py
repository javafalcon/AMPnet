# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 11:38:25 2018

@author: Administrator
"""

import requests
from bs4 import BeautifulSoup

def getHTMLText(url,coding='gbk'):
    try:
        r = requests.get(url,timeout=30)
        #print(r)
        r.raise_for_status()
        r.encoding = coding
        return r.text
    except:
        return ""

def fullProteinsList(soup):
    APDList = []
    data = soup.find_all('a',{'target':'_blank'})
    for a in data:
        APDList.append(a.string)
    return APDList

def main():
    url = 'http://aps.unmc.edu/AP/database/antiB.php'
    html = getHTMLText(url)
    soup = BeautifulSoup(html,"html.parser")
    #print(soup)
    ls = fullProteinsList(soup)
    print(ls)
        
    
if __name__ == '__main__':
    main()