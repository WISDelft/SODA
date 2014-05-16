from util import *
import string
currentdir = os.path.dirname(__file__)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir+'/ex_tools')
from kappa import *

#typeDict = {'context':0, 'example':1, 'code':2, 'description':3, 'detail':4, 'attempt':5}
typeDict = {'code':2, 'description':3, 'detail':4, 'attempt':5}
#fName = ['edit_jie.csv', 'edit_claudia.csv', 'edit_alessandro.csv']
fName = ['edit_1.csv', 'edit_2.csv', 'edit_3.csv']
extreme_ids = loadfile_flat('ed_type_qids_extreme')
common_ids = extreme_ids[900:1000]
ev = 1
for type in typeDict:
    pos1 = []
    all1 = []
    f = fName[0]
    for line in open('annotations/' + f):
        fields = line.split(';')
        qid = fields[0]
        if qid != '':
            this_q = int(qid)
            all1.append(this_q)
        lb = fields[3][0:-2]
        lb = lb.split('|')
        #print 'qid:' + str(this_q)
        #print lb
        for l in lb:
            if int(l)==0 or int(l)==1:
                l=3
            if typeDict[type] == int(l) and this_q not in pos1:
                pos1.append(this_q)
                break
    pos2 = []
    all2 = []
    f = fName[1]
    for line in open('annotations/' + f):
        fields = line.split(';')
        qid = fields[0]
        if qid != '':
            this_q = int(qid)
            all2.append(this_q)
        lb = fields[3]
        lb = lb.split('|')
        #print 'qid:' + str(this_q)
        #print lb
        for l in lb:
            if int(l)==0 or int(l)==1:
                l=3
            if typeDict[type] == int(l) and this_q not in pos2:
                pos2.append(this_q)
                break
    pos3 = []
    all3 = []
    f = fName[2]
    for line in open('annotations/' + f):
        fields = line.split(';')
        qid = fields[0]
        if qid != '':
            this_q = int(qid)
            all3.append(this_q)
        lb = fields[3][0:-2]
        lb = lb.split('|')
        #print 'qid:' + str(this_q)
        #print lb
        for l in lb:
            if l=='??':
                continue
            if int(l)==0 or int(l)==1:
                l=3
            if typeDict[type] == int(l) and this_q not in pos3:
                pos3.append(this_q)
                break
    

    '''pos1 = []
    all1 = []
    i = 0
    f = fName[0]
    for line in open('result/' + f):
        fields = line.split(';')
        qid = fields[0]
        if qid != '':
            this_q = int(qid)
            all1.append(this_q)
            i = 0
        else:
            i += 1
        if i == ev:
            lb = fields[3][0:-2]
            lb = lb.split('|')
            #print 'qid:' + str(this_q)
            #print lb
            for l in lb:
                if int(l)==0 or int(l)==1:
                    l=3
                if typeDict[type] == int(l) and this_q not in pos1:
                    pos1.append(this_q)
                    break
    pos2 = []
    all2 = []
    f = fName[1]
    for line in open('result/' + f):
        fields = line.split(';')
        qid = fields[0]
        if qid != '':
            this_q = int(qid)
            all2.append(this_q)
            i = 0
        else:
            i += 1
        if i==ev:
            lb = fields[3]
            lb = lb.split('|')
            #print 'qid:' + str(this_q)
            #print lb
            for l in lb:
                if int(l)==0 or int(l)==1:
                    l=3
                if typeDict[type] == int(l) and this_q not in pos2:
                    pos2.append(this_q)
                    break
    pos3 = []
    all3 = []
    f = fName[2]
    for line in open('result/' + f):
        fields = line.split(';')
        qid = fields[0]
        if qid != '':
            this_q = int(qid)
            all3.append(this_q)
            i  = 0
        else:
            i += 1
        if i ==ev:
            lb = fields[3][0:-2]
            lb = lb.split('|')
            #print 'qid:' + str(this_q)
            #print lb
            for l in lb:
                if l=='??':
                    continue
                if int(l)==0 or int(l)==1:
                    l=3
                if typeDict[type] == int(l) and this_q not in pos3:
                    pos3.append(this_q)
                    break'''
    
    agree_mat = []
    for q in common_ids:
        agree_neg = 0
        agree_pos = 0
        if q in pos1:
            agree_pos += 1
        else:
            agree_neg += 1
        if q in pos2:
            agree_pos += 1
        else:
            agree_neg += 1
        if q in pos3:
            agree_pos += 1
        else:
            agree_neg += 1
        agree_mat.append([agree_neg, agree_pos])
    print type+' kappa:'+str(computeKappa(agree_mat))
    
    qlabels = dict([])
    non_common = set(extreme_ids).difference(set(common_ids))
    #print len(non_common)
    #print len(common_ids)
    pos = set(pos1+pos2+pos3)
    pos_nr = 0
    for qid in non_common:
        if qid in pos:
            qlabels[qid] = 1
            pos_nr += 1
        else:
            qlabels[qid] = 0
    for qid in common_ids:
        posc = 0
        if qid in pos1:
            posc += 1
        if qid in pos2:
            posc += 1
        if qid in pos3:
            posc += 1
        if posc >= 2:
            qlabels[qid] = 1
            pos_nr += 1
        else:
            qlabels[qid] = 0
    print '    positive nr: '+str(pos_nr)
    qids = []
    for q in qlabels:
        qids.append(q)
    #dumpfile(qids, 'ed_type_qids_extreme_'+type)
    dumpfile(qlabels, 'ed_type_label_extreme_'+type)
    