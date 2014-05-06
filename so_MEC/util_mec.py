import string
import psycopg2
import sys
import numpy
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