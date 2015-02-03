import string
import psycopg2
import sys
import pickle
import os
import numpy as np
import math


from calculate_mec import *
from compare_performance import *
from characterize_perference import *
from get_userAtt import *
from robustness_component import *
from util_mec import *

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='wistudelft')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()



def v2a():
    fu = open("userranklists_c#.pik")
    userranklists = pickle.load(fu)
    qlen = []
    for urank in userranklists:
        qlen.append(len(urank))
    print "qlen: "+str(numpy.mean(qlen))+" +/- "+str(numpy.std(qlen))

def get_top_users_nrans(nr):
    top_users = dict([])
    nrans = loadfile('nrans_'+tag)
    nrans_lst = dict2list_full(nrans)
    nrans_lst = sorted(nrans_lst, key=lambda nrans_lst : nrans_lst[1],reverse=True)
    for i in range(nr):
        top_users[nrans_lst[i][0]] = nrans_lst[i][1]
    dumpfile(top_users, "TPnrans_"+tag)
    return top_users

def get_top_users_zscore(nr):
    top_users = dict([])
    nrans = loadfile('nrans_'+tag)
    nrqst = loadfile('nrqst_'+tag)
    zscore = []
    for u in nrans:
        zscore.append([u, float(nrans[u]-nrqst[u])/math.sqrt(nrans[u]+nrqst[u])])
    zscore = sorted(zscore, key=lambda zscore : zscore[1],reverse=True)
    for i in range(nr):
        top_users[zscore[i][0]] = zscore[i][1]
    dumpfile(top_users, "TPzscore_"+tag)
    return top_users

def get_top_users_reputation(nr):
    top_users = dict([])
    nrans = loadfile('nrans_'+tag)
    repus = []
    for u in nrans:
        cur.execute("select reputation from users where id="+str(u))
        result = cur.fetchone()
        if result!=None and result[0]!=None:
            repus.append([u, result[0]])
        else:
            repus.append([u, 0])
    repus = sorted(repus, key=lambda repus : repus[1],reverse=True)
    for i in range(nr):
        top_users[repus[i][0]] = repus[i][1]
    dumpfile(top_users, "TPreputation_"+tag)
    return top_users

def get_top_users_logMEC(answers_att, users, tag, nr):
    userset = dict([])
    nrqst = dict([])
    au = dict([])
    ds = dict([])
    for a in answers_att:
        qid = a[2]
        uid = a[0]
        userset[uid] = 0
        au[uid] = []
        ds[uid] = []
        nrqst[uid] = 0
    userranklists = loadfile("userranklists_"+tag)

    allqstlen = []
    for urank in userranklists:
        allqstlen.append(len(urank))
    avgdt = numpy.mean(allqstlen)
    print 'mean NO.answerers to questions: '+str(avgdt)
    
    #calculate user expertise
    for urank in userranklists:
        for i in range(len(urank)):
            pos = urank[i]
            u = pos[0]
            userset[u] = userset[u] + (float(1)/(i+1))*(math.log(float(len(urank)),2))
            au[u].append( (float(1)/(i+1)))
            ds[u].append( len(urank) )
            nrqst[u] += 1


    experts = dict([])
    for u in userset:
        if nrqst[u]>=1:
            userset[u] = float(userset[u])/(nrqst[u]*math.log(numpy.mean(allqstlen), 2))
            #experts[u] = userset[u]
            if userset[u]>=1:
                experts[u] = userset[u]
    dumpfile(experts, "TPlogMEC_"+tag)
    return experts

def get_top_users_logMEC_sameNr(answers_att, users, tag, nr):
    userset = dict([])
    nrqst = dict([])
    au = dict([])
    ds = dict([])
    for a in answers_att:
        qid = a[2]
        uid = a[0]
        userset[uid] = 0
        au[uid] = []
        ds[uid] = []
        nrqst[uid] = 0
    userranklists = loadfile("userranklists_"+tag)

    allqstlen = []
    for urank in userranklists:
        allqstlen.append(len(urank))
    avgdt = numpy.mean(allqstlen)
    print 'mean NO.answerers to questions: '+str(avgdt)
    
    #calculate user expertise
    for urank in userranklists:
        for i in range(len(urank)):
            pos = urank[i]
            u = pos[0]
            userset[u] = userset[u] + (float(1)/(i+1))*(math.log(float(len(urank)),2))
            au[u].append( (float(1)/(i+1)))
            ds[u].append( len(urank) )
            nrqst[u] += 1


    for u in userset:
        if nrqst[u]>=1:
            userset[u] = float(userset[u])/(nrqst[u]*math.log(numpy.mean(allqstlen), 2))
            
    top_users = dict([])
    userset = dict2list_full(userset)
    userset = sorted(userset, key=lambda userset : userset[1],reverse=True)
    for i in range(nr):
        top_users[userset[i][0]] = userset[i][1]
    dumpfile(top_users, "TPlogMEC_sameNr_"+tag)
    return top_users

def get_top_users_kernelMEC(nr):
    top_users = dict([])
    return top_users

if __name__ == '__main__':
    tag = str(sys.argv[1])
    # get answerers and answer attributes
    if os.path.exists("temp_files/all_answerers.pik"):
        all_answerers = loadfile("all_answerers")
    else:
        all_answerers = get_answerers()
    if os.path.exists("temp_files/answers_att_"+tag+".pik"):
        answers_att = loadfile("answers_att_"+tag)
        this_answerers = loadfile("answerers_"+tag)
    else:
        this_answerers, answers_att = get_answers_att(all_answerers, tag)        # figure 1,2,4
    
    # calculate MEC scores and get experts
    if os.path.exists("temp_files/experts_"+tag+".pik"):
        answerer_scores = loadfile("MECscores_"+tag)
        experts = loadfile("experts_"+tag)
    else:        
        experts, answerer_scores = get_MECscores(answers_att, this_answerers, tag)

    nr_subset = len(experts)
    
    # get top users in terms of different expertise metrics
    '''tp_nrans = get_top_users_nrans(nr_subset)
    tp_zscore = get_top_users_zscore(nr_subset)
    tp_reputation = get_top_users_reputation(nr_subset)
    tp_logMEC = get_top_users_logMEC(answers_att, this_answerers, tag, nr_subset)'''
    tp_logMEC_sameNr = get_top_users_logMEC_sameNr(answers_att, this_answerers, tag, nr_subset)
    
    # compare answering quality
    '''get_performance(experts, answerer_scores, tp_nrans, tag, 'TPnrans')
    get_performance(experts, answerer_scores, tp_zscore, tag, 'TPzscore')
    get_performance(experts, answerer_scores, tp_reputation, tag, 'TPreputation')
    get_performance(experts, answerer_scores, tp_logMEC, tag, 'TPlogMEC')'''
    get_performance(experts, answerer_scores, tp_logMEC_sameNr, tag, 'TPlogMEC_sameNr')
    