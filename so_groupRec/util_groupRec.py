import string
import psycopg2
import sys
import numpy
import pickle
import math
import numpy as np
import os

def loadfile(fname):
    f = open("../../temp/groupRec/"+fname+".pik")
    data = pickle.load(f)
    f.close()
    return data

def dumpfile(data, fname):
    f = open("../../temp/groupRec/"+fname+".pik", 'w')
    pickle.dump(data, f)
    f.close()