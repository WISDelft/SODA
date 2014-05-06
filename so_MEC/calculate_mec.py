import string
import psycopg2
import sys
import numpy
import pickle
import math
import numpy as np
import pylab as pl
import os
from util_mec import *

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()

'''
    Figure 1,2,4 can be generated from this script.
'''

def get_MECscores(answers_att, users, tag):
    # get all questions
    qids = dict([])
    userset = dict([])
    nrqst = dict([])
    for a in answers_att:
        qid = a[2]
        uid = a[0]
        qids[qid] = 0
        userset[uid] = 0
        nrqst[uid] = 0

    # get partial ranking for each question
    print  "Nr of questions: "+str(len(qids))
    index = 0
    userranklists = []
    if os.path.exists("temp_files/userranklists_"+tag+".pik"):
        userranklists = loadfile("userranklists_"+tag+".pik")
    else:
        for qid in qids:
            index += 1
            if index%1000 == 0:
                print "the "+str(index)+"th question"
            urank = []
            for a in answers_att:
                if qid == a[2]:
                    uscore = a[3]
                    uid = a[0]
                    urank.append([uid, uscore])
            if len(urank) < 1:
                continue
            urank = sorted(urank, key = lambda urank : urank[1], reverse=True)
            qids[qid] = len(urank)
            userranklists.append(urank)

        dumpfile(userranklists, "userranklists_"+tag+".pik", "w")

    allqstlen = []
    for urank in userranklists:
        allqstlen.append(len(urank))
    print 'mean NO.answerers to questions'+str( numpy.mean(allqstlen))
    '''qids store with qst length, userset store with their performance, and nrqst store with his qst number'''
    
    #calculate user expertise
    ss = []
    for urank in userranklists:
        for i in range(len(urank)):
            pos = urank[i]
            u = pos[0]
            ss.append(pos[1])
            #print str((float(1)/(i+1)))+" "+str((len(urank)))+ " "+str((float(1)/(i+1))*(len(urank)))
            #userset[u] = userset[u] + (float(1)/(i+1))*(numpy.log2(len(urank)+1))
            userset[u] = userset[u] + (float(1)/(i+1))*(float(len(urank)))
            nrqst[u] += 1
    #print numpy.mean(ss)
    #print numpy.std(ss)

    experts = dict([])
    for u in userset:
        if nrqst[u]>=1:
            userset[u] = float(userset[u])/(nrqst[u]*numpy.mean(allqstlen))
            #experts[u] = userset[u]
            if userset[u]>=1:
                experts[u] = userset[u]

    dumpfile(experts, "experts_"+tag+".pik")
    dumpfile(userset, "MECscores_"+tag+".pik")
    
    return experts, userset

