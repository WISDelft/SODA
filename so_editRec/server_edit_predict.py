import string
import sys
import psycopg2
import os
import pickle
import numpy as np
import random
import json
from scipy import sparse
import scipy.io

from classification import *
def loadfile_flat(fname):
    f = open(fname+".pik")
    data = pickle.load(f)
    f.close()
    return data

def dict2list(d):
    ds = []
    for dd in d:
        ds.append([dd, d[dd]])
    return ds
def dumpfile(data, fname):
    f = open(fname+".pik", 'w')
    pickle.dump(data, f)
    f.close()
def remove_feature(qfeatures, df, tps):
    (m,n) = qfeatures.shape
    print '-- before remove --'
    print m
    print n
    #print len(allu)
    qfeature_sl = []
    worddict = dict2list(df)
    effective_df = []
    effective_tps = []
    for j in range(n):
        if j%1000==0:
            print "processing the "+str(j)+"th column"
    
        if np.count_nonzero(qfeatures[:,j])>=10:
            this=list(qfeatures[:,j])
            qfeature_sl.append(this)
            if j<len(df):
                effective_df.append(worddict[j][0])
            if j>=len(df) and j<len(df)+len(tps):
                effective_tps.append(tps[j-len(df)])
            #worddict[allu[j]] = df2[allu[j]]
            #print qfeature_sl
    print np.array(qfeature_sl).shape
    dumpfile(effective_df, 'effective_df')
    dumpfile(effective_tps, 'effective_tps')
    return np.array(qfeature_sl).T
    
if __name__ == '__main__':
    #get_label(int(sys.argv[1]))
    #sys.exit(1)
    pre = []
    rec = []
    f1 = []
    acc = []
    lun = 5
        
    for i in range(5):
        #load_label_gene(i)
        qids = loadfile_flat('tempqids_extr')
        qlabels = loadfile_flat('tempqlabels_extr')
        print len(qids)
    
        #gene_df(qids)
        #gene_tps(qids)
        features = []
        labels = []
        this_i = 0
        df = loadfile_flat('df')
        print len(df)
        tps = loadfile_flat('tps')
        print len(tps)
            
        features = loadfile_flat('full_features_extr')
        labels = loadfile_flat('full_labels_extr')
        
        features = np.array(features)
        features = remove_feature(features, df, tps)
        labels = np.array(labels)
        np.save("qfeatures_extr_newfeature.pik", features)
        np.save("qlabels_extr_newfeature.pik", labels)
            
        p, r,f,ac = classify(35318)
        pre.append(p)
        rec.append(r)
        f1.append(f)
        acc.append(ac)
        sys.exit(1)
    print "precision: "+str(np.mean(pre))+"+/-"+str(np.std(pre))
    print "recall: "+str(np.mean(rec))+"+/-"+str(np.std(rec))
    print "f1-score: "+str(np.mean(f1))+"+/-"+str(np.std(f1))
    print "accuracy: "+str(np.mean(acc))+"+/-"+str(np.std(acc))
    