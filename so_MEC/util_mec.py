import string
import psycopg2
import sys
import pickle
import math
import numpy as np
import os

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()
print "Connecting to DB: " + str(ver)

def is_tag_in(tag, tags):
    for t in tags.split('|'):
        if t==tag:
            return True
    return False

def loadfile(fname):
    f = open("temp_files/"+fname+".pik")
    data = pickle.load(f)
    f.close()
    return data

def dumpfile(data, fname):
    f = open("temp_files/"+fname+".pik", 'w')
    pickle.dump(data, f)
    f.close()
    
def writeout(a, fname):
    fout = open("data/"+fname+".csv", 'w')
    for x in a:
        fout.write(str(x)+"\n")
    fout.close()
    
def dict2list(dct):
    lst = []
    for k in dct:
        lst.append(dct[k])
    return lst

def dict2list_full(dct):
    lst = []
    for k in dct:
        lst.append([k,dct[k]])
    return lst

def dict2list_mean(dct):
    dct_transform = dict([])
    for k in dct:
        dct_transform[k] = np.mean(dct[k])
    lst = []
    for k in dct_transform:
        lst.append(dct_transform[k])
    return lst
def dict2list_median(dct):
    dct_transform = dict([])
    for k in dct:
        dct_transform[k] = np.median(dct[k])
    lst = []
    for k in dct_transform:
        lst.append(dct_transform[k])
    return lst
def dict2list_std(dct):
    dct_transform = dict([])
    for k in dct:
        dct_transform[k] = np.std(dct[k])
    lst = []
    for k in dct_transform:
        lst.append(dct_transform[k])
    return lst

def dict2list_filter(dct, filter):
    lst = []
    for k in dct:
        if filter[k]>=1:
            lst.append(dct[k])
    return lst

def dict2list_filter_mean(dct, filter):
    dct_transform = dict([])
    for k in dct:
        dct_transform[k] = np.mean(dct[k])
    lst = []
    for k in dct_transform:
        if filter[k]>=1:
            lst.append(dct_transform[k])
    return lst
def dict2list_filter_median(dct, filter):
    dct_transform = dict([])
    for k in dct:
        dct_transform[k] = np.median(dct[k])
    lst = []
    for k in dct_transform:
        if filter[k]>=1:
            lst.append(dct_transform[k])
    return lst
def dict2list_filter_std(dct, filter):
    dct_transform = dict([])
    for k in dct:
        dct_transform[k] = np.std(dct[k])
    lst = []
    for k in dct_transform:
        if filter[k]>=1:
            lst.append(dct_transform[k])
    return lst

def dict2list_filter2(dct, filter):
    lst = []
    for k in dct:
        if filter[k]==1:
            lst.append(dct[k])
    return lst

def dict2list_filter2_mean(dct, filter):
    dct_transform = dict([])
    for k in dct:
        dct_transform[k] = np.mean(dct[k])
    lst = []
    for k in dct_transform:
        if filter[k]==1:
            lst.append(dct_transform[k])
    return lst
def dict2list_filter2_median(dct, filter):
    dct_transform = dict([])
    for k in dct:
        dct_transform[k] = np.median(dct[k])
    lst = []
    for k in dct_transform:
        if filter[k]==1:
            lst.append(dct_transform[k])
    return lst
def dict2list_filter2_std(dct, filter):
    dct_transform = dict([])
    for k in dct:
        dct_transform[k] = np.std(dct[k])
    lst = []
    for k in dct_transform:
        if filter[k]==1:
            lst.append(dct_transform[k])
    return lst

if __name__ == '__main__':
    userranklists = loadfile("userranklists_c#")
    nrans = loadfile("nrans_c#")
    f_user=open('data/nrans_c#.csv', 'w')
    f_qst=open('data/qlen_c#.csv', 'w')
    
    m = 0
    for u in nrans:
        #if nrans[u]>=2 and nrans[u]<10:
        if nrans[u]>=10:
            m += 1
        f_user.write(str(nrans[u])+'\n')
    print m
    
    for urank in userranklists:
        f_qst.write(str(len(urank))+'\n')
    
    f_user.close()
    f_qst.close()