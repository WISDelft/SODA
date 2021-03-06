import string
import sys
import psycopg2
import os
import pickle
import numpy as np
import random
from nltk.corpus import stopwords
import json
from scipy import sparse
import scipy.io


from util import *
from feature_extract import *
from classification import *
from tfidf import *

'''settings:
[24/06/14 11:29:07] luyuan: 1:1.2
[24/06/14 11:29:21] luyuan: 1:1
[24/06/14 11:29:25] luyuan: 1:1
[24/06/14 11:29:28] luyuan: 1:0.08
[24/06/14 11:29:34] luyuan: 1:0.28'''


def get_part(vec, part):
    return vec[200000*(part-1)+1:200000*part]
def get_part2(vec, part):
    if part==51:
        return vec[200000*4+1:200000*4+100000]
    else:
        return vec[200000*4+100000+1:200000*5]
def get_label(part = 1):
    eqids = loadfile('ed_qst_ids')
    eqids = get_part(eqids, part)
    
    ed_qids = []
    inds = 0
    
    len1 = []
    for qid in eqids:
        inds += 1
        if inds%1000==0:
            print '..[filter] finished '+str(inds)+'the question'
            print 'pos: '+str(len(ed_qids))+', len:'+str(np.median(len1))
        
        flag, length = select_pos(qid)
        
        if flag:
            ed_qids.append(qid)
            len1.append(length)
    
    dumpfile(ed_qids, 'good_edit_ans_'+str(part))




def load_label_gene(lun, type=0):
    '''pos = loadfile_all('good_edit_ans_', [1,2,3,4,51,52,6,7,8,9,10])
    print len(pos)
    pos = get_most_edit(pos, 25406)
    dumpfile(pos, 'extreme_set')'''
    
    pos = loadfile_flat('extreme_set')
    #output_samples(pos)
    
    '''allqids = loadfile_flat('allqids')
        allqids = allqids[1:400000]+allqids[800001:1200000]
        random.shuffle(allqids)'''
    eqids = sorted(loadfile('ed_qst_ids'))
    print 'hiiii'+str(len(eqids))
    qids = []
    qlabels = dict([])
    print len(pos)
    
    for i in pos:
        if check_date(i, '2013-01-01'):
            qids.append(i)
            qlabels[i] = 1
    now_len = len(qids)
    print now_len
    
    '''auth=loadfile_flat('nrans')
    auth=dict2list(auth)
    random.shuffle(auth)
    auth=sorted(auth, key=lambda auth : auth[1],reverse=True)
    dumpfile(auth, 'sorted_nrans')'''
    auth=loadfile_flat('sorted_nrans')
    k = 0
    m = 0
    #valid = set(allqids).difference(set(eqids))
    #print len(valid)
    for t in auth:
        k+=1
        if k<(lun-1)*10000:
            continue
        if not bi_contains(eqids, t[0]) and check_date(t[0], '2013-01-01'):#t[0] not in eqids: #and check_date(t[0], '2013-01-01'):
            #print t
            qids.append(t[0])
            qlabels[t[0]] = 0
            m+=1
            '''if m%100==0:
                print m'''
            if m==now_len:
                break
    random.shuffle(qids)
    print 'training nr: '+str(len(qids))
    train_len = len(qids)
    
    k = 0
    #pos = loadfile_flat('extreme_set')
    #pos = loadfile_all('good_edit_ans_', [1,2,3,4,51,52,6,7,8,9,10])
    random.shuffle(eqids)
    for i in eqids:
        #print i
        if bi_contains(qids, i) or check_date(i, '2013-01-01'):
            continue
        else:
            qids.append(i)
            qlabels[i] = 1
            k+=1
        #if k==15000:
            #break
    print k
    
    m = 0
    allqids = loadfile('allqids')
    random.shuffle(allqids)#3.
    for i in allqids:
        #1,2, this = i[0]
        this = i
        #this = i[0]
        if not bi_contains(qids,this) and (not check_date(this, '2013-01-01')) and not bi_contains(eqids, this):
            qids.append(this)
            qlabels[this] = 0
            m+=1
            #if m==k:
                #break
    print m
    print 'all nr: ' + str(len(qids))
    
    #random.shuffle(qids)
    
    dumpfile(qids, 'tempqids_ambi')
    dumpfile(qlabels, 'tempqlabels_ambi')
    return qids, qlabels, train_len



def neg_augment_extr(qids,qlabels, nr):
    auth=loadfile_flat('sorted_nrans')
    #extr = loadfile_flat('tempqlabels_extr')
    eqids = sorted(loadfile('ed_qst_ids'))
    k = 0
    for i in auth:
        if not bi_contains(eqids, i[0]) and i[0] not in qids:
            qids.append(i[0])
            qlabels[i[0]] = 0
            k += 1
        if k==nr:
            break

    return qids, qlabels
    
def neg_augment(qids,qlabels, nr):
    extr = loadfile_flat('tempqlabels_extr')
    extr = dict2list(extr)
    random.shuffle(extr)
    for i in range(nr):
        flag=True
        while flag:
            ind = int(np.random.rand()*len(extr))
            elem = extr[ind][0]
            if elem not in qids and extr[ind][1]==0:
                qids.append(elem)
                qlabels[elem]=0
                flag=False

    return qids, qlabels
if __name__ == '__main__':
    pre = []
    rec = []
    f1 = []
    acc = []
    i = 1
    #typeDict = {'code':2, 'description':3, 'attempt':5} 
    #typeDict = {'code':2}
    typeDict = {'description':3} 
    qids = loadfile_flat('ed_type_qids_extreme')
    
    for type in typeDict:
        print type
        qlabels = loadfile_flat('ed_type_label_extreme_'+type)
        if type=='attempt' or type=='code':
            augmented_qids = loadfile_flat('augmented_'+type)
        
        '''for qid in augmented_qids:
            if qid in qids:
                continue
            qlabels[qid] = 1
        qids = list(set(qids+augmented_qids))
        print len(qids)'''

        qids, qlabels = neg_augment_extr(qids,qlabels, 84)
        
        random.shuffle(qids)
        
        get_index = []
        orn = loadfile_flat('ed_type_qids_extreme')
        for i in range(len(qids)):
            if qids[i] in orn:
                get_index.append(i)
        dumpfile(get_index, 'get_index_sec')
                    
        #gene_df(qids)
                
            
        features = []
        labels = []
        this_i = 0
        df = loadfile_flat('df_type')
                
        for qid in qids:
            this_i += 1
            if this_i%1000==0:
                print '..[feature] processing the '+str(this_i)+"th question"
            qfeature = extract_one(qid, df)
            if isinstance( qfeature, int ):
                continue

            #add hypothesis feature
            #qfeature = add_feature4type(qid, qfeature)
                
            features.append(qfeature)
            labels.append(qlabels[qid])
                
                
        features = np.array(features)
        features = remove_feature_type(features, '')
        labels = np.array(labels)
        np.save("temp_files/qfeatures_type_sec.pik", features)
        np.save("temp_files/qlabels_type_sec.pik", labels)
                
        p, r,f,ac = classify()
        pre.append(p)
        rec.append(r)
        f1.append(f)
        acc.append(ac)
        
        
    '''print "precision: "+str(np.mean(pre))+"+/-"+str(np.std(pre))
    print "recall: "+str(np.mean(rec))+"+/-"+str(np.std(rec))
    print "f1-score: "+str(np.mean(f1))+"+/-"+str(np.std(f1))
    print "accuracy: "+str(np.mean(acc))+"+/-"+str(np.std(acc))'''