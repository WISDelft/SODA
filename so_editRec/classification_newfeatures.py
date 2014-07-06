import string
import psycopg2
from scipy.stats.stats import pearsonr
from scipy.stats.stats import spearmanr
from scipy.stats import ttest_1samp, wilcoxon, ttest_ind, mannwhitneyu, rankdata
import numpy
import matplotlib.pyplot as plt
import pylab
import sys
import math
from scipy import linalg
import math
import os
import pickle

import scipy
import scipy.io
import pylab
import scipy.cluster.hierarchy as sch

from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.feature_selection import *
from sklearn.metrics import *
from sklearn.decomposition import PCA
from sklearn.decomposition import ProjectedGradientNMF

from scipy.stats.mstats import mannwhitneyu
#import gensim

#from util import *

schema = ['code_len', 'list_len', 'note_len', 'em_len', 'codeem_len', 'link_len', 'ask_len', 'explan_len', 'sent_len', "('to', 'add')", "('work', 'fine')", "('the', 'follow')", "('i', 'get')", "('a', 'string')", "('get', 'the')", "('know', 'how')", "('tri', 'to')", "('i', 'run')", "('i', 'use')", "('the', 'class')", '("\'m", \'use\')', "('in', 'java')", "('get', 'a')", "('pleas', 'help')", "('to', 'write')", "('seem', 'to')", "('help', 'me')", "('the', 'method')", "('a', 'lot')", "('i', 'tri')", "('so', 'far')", "('it', 'work')", "('the', 'first')", "('am', 'tri')", "('a', 'new')", "('it', 'doe')", "('doe', 'not')", '("\'m", \'tri\')', "('thank', 'you')", "('ani', 'help')", "('look', 'like')", "('ani', 'idea')", "('run', 'the')", "('is', 'thi')", "('in', 'thi')", "('with', 'thi')", "('give', 'me')", "('to', 'get')", "('make', 'a')", "('i', 'need')", "('a', 'method')", "('need', 'to')", "('the', 'code')", '(\'do\', "n\'t")', "('new', 'to')", "('lot', 'of')", "('thi', 'code')", "('to', 'find')", "('thi', 'is')"]#, 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14']


def expert_explore(qfeatures, labels, j):
    a = []
    for i in range(len(labels)):
        if labels[i] == 0:
            a.append(qfeatures[i, j])

    fig, ax = plt.subplots()
    x,y,z = ax.hist(a,50)
    x1 = []
    for a in x:
        x1.append(round(float(a)/1702, 2))
    print x
    print y
    plt.show()

    '''fig, ax = plt.subplots()
    ax.scatter(a,b, s=0.2)
    ax.set_yscale('log')
    plt.show()
    fig.savefig('scatter.png')'''

def coff_explore(coffs):
    f=open("effective_df.pik", 'rb')
    features = pickle.load(f)
    print 'nr features:'+str(len(features))
    f.close()

    f=open("effective_tps.pik",'rb')
    tmp = pickle.load(f)
    f.close()

    features.extend(tmp)

    coffrank = []
    for i in range(len(coffs)):
        coffrank.append([coffs[i],i])
    coffrank = sorted(coffrank, key = lambda coffrank : coffrank[0], reverse=True)
    for i in range(10):
        print features[coffrank[i][1]]+': '+str(coffrank[i][0])
        print features[coffrank[-i][1]]+': '+str(coffrank[-i][0])
    
def classify(train_len=10000):
    train_len = 35318
    qfeatures = numpy.load("qfeatures_extr_newfeature.pik.npy")
    
    '''corpus = gensim.matutils.Dense2Corpus(qfeatures.T)
    #id2word = gensim.corpora.Dictionary.load_from_text('java_stats/dictionary')
    
    lsi = gensim.models.lsimodel.LsiModel(corpus=corpus,  num_topics=100)#id2word=id2word,
    qfeatures = gensim.matutils.corpus2dense(lsi[corpus], len(lsi.projection.s)).T / lsi.projection.s'''
    
    #lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, num_topics=100, update_every=0, passes=20)
    #qfeatures = gensim.matutils.corpus2dense(lsi[corpus])
    
    labels = numpy.load("qlabels_extr_newfeature.pik.npy")

    #expert_explore(qfeatures, labels, 7)
    #sys.exit(1)
    #qfeatures = remove_feature(qfeatures)
    
    (m,n) = qfeatures.shape
    print "nr of questions: "+str(m)
    posnr = 0
    for i in range(m):
        if labels[i] == 1:
            posnr += 1
    print "nr of editted questions: "+str(posnr)
    print "nr features: "+str(n)
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(qfeatures)
    qfeatures = imp.transform(qfeatures)



    #qfeatures, schema_sl = select_feature(qfeatures, fset, [1, 4, 7, 10])
    #feature_explore(qfeatures, labels)
    #hidden = PCA(n_components=100)
    #hidden = ProjectedGradientNMF(n_components=100, sparseness='data',random_state=0)
    #qfeatures = hidden.fit_transform(qfeatures)
    #print(hidden.explained_variance_ratio_)
    
    
    qfeatures = preprocessing.scale(qfeatures)
    print qfeatures.shape
    
    X_train, X_test = qfeatures[0:train_len], qfeatures[train_len:-1]
    y_train, y_test = labels[0:train_len], labels[train_len:-1]
    clf = LogisticRegression(penalty='l1', tol=0.01, class_weight={1: 0.05})
        #clf = svm.SVC(kernel='linear', class_weight={1: 1})
    clf.fit(X_train, y_train)

    #coff_explore(clf.coef_[0].tolist())

    y_pred = clf.predict(X_train)
    cm = confusion_matrix(y_train, y_pred)
    print cm
    print 'training acc: '+str(accuracy_score(y_train, y_pred))

    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print cm
    print 'test acc: '+str(accuracy_score(y_test, y_pred))
    print 'test pre: '+str(precision_score(y_test,y_pred))
    print 'test rec: '+str(recall_score(y_test,y_pred))
    print 'test f1: '+str(f1_score(y_test,y_pred))
    
    return precision_score(y_test,y_pred), recall_score(y_test,y_pred),f1_score(y_test,y_pred),accuracy_score(y_test, y_pred)
    sys.exit(1)

    get_index = loadfile_flat('get_index')
    y_orn_true = []
    y_orn_pred = []

    pre=[]
    rec=[]
    f1 = []
    acc = []
    coff=[]
    kf = cross_validation.KFold(len(labels), n_folds=5)
    for train_index, test_index in kf:
        
        X_train, X_test = qfeatures[train_index], qfeatures[test_index]
        #print X_train.shape
        
        
        y_train, y_test = labels[train_index], labels[test_index]
        #print len(y_train)
        #print test_index
        
        posnr = 0
        for i in range(len(test_index)):
            if y_test[i] == 1:
                posnr += 1
        #print "nr of experts: "+str(posnr)+" frac: "+str(float(posnr)/len(y_test))


        
        #clf = svm.SVC(kernel='linear', class_weight={1: 2})
        #clf.fit(X_train, y_train)
        '''selector = SelectKBest(f_regression, k=5)
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)'''

        clf = LogisticRegression(penalty='l1', tol=0.01, class_weight={1:0.28})
        #clf = svm.SVC(kernel='linear', class_weight={1: 1})
        clf.fit(X_train, y_train)
        
        '''y_pred = clf.predict(X_train)
        cm = confusion_matrix(y_train, y_pred)
        print cm
        print 'training acc: '+str(accuracy_score(y_train, y_pred))'''
        
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        print cm
        
        for ind in range(len(test_index)):
            if test_index[ind] in get_index:
                y_orn_true.append(y_test[ind])
                y_orn_pred.append(y_pred[ind])
        #print rs
        #confusion_matrix(y_test,rs)
        pre.append( precision_score(y_test,y_pred))
        rec.append( recall_score(y_test,y_pred))
        f1.append(f1_score(y_test,y_pred))
        acc.append(accuracy_score(y_test, y_pred))
        #print pre
        #print rec
        
        
        cof=clf.coef_
        tmp=cof[0]
        coff.append(tmp.tolist())

    #print "precision: "+str(numpy.mean(pre))+"+/-"+str(numpy.std(pre))
    #print "recall: "+str(numpy.mean(rec))+"+/-"+str(numpy.std(rec))
    print "accuracy: "+str(numpy.mean(acc))+"+/-"+str(numpy.std(acc))
    print "precision: "+str(numpy.mean(pre))+"+/-"+str(numpy.std(pre))
    print "recall: "+str(numpy.mean(rec))+"+/-"+str(numpy.std(rec))
    print "fmeasure: "+str(numpy.mean(f1))+"+/-"+str(numpy.std(f1))
    
    print precision_score(y_orn_true,y_orn_pred)
    print recall_score(y_orn_true,y_orn_pred)
    print f1_score(y_orn_true,y_orn_pred)
    print accuracy_score(y_orn_true,y_orn_pred)
    print np.count_nonzero(y_orn_true)
    
    return numpy.mean(pre), numpy.mean(rec), numpy.mean(f1), numpy.mean(acc)
    
    coff=numpy.array(coff)
    for i in range(n):
        print schema[i]
        print str(numpy.mean(coff[:,i]))+"+/-"+str(numpy.std(coff[:,i]))
    
    
    
def select_feature(qfeatures, fset, findex):
    #schema = set_schema(fset)
    (m,n) = qfeatures.shape
    nrf = len(findex)
    qfeatures_sl = numpy.zeros((m,nrf))
    
    for i in range(nrf):
        fi = findex[i]
        #print schema[fi]
        #schema_sl.append(schema[fi])
        qfeatures_sl[:,i] = qfeatures[:, fi]
    
    return qfeatures_sl#, schema_sl


def feature_explore(qfeatures, labels):
    '''(m,n) = qfeatures.shape
    pos = []
    for j in range(n):
        pos = []
        neg = []
        for i in range(m):
            all.append(qfeatures[i,j])
            #if users[i] in ua:
            if labels[i]==1:
                pos.append(qfeatures[i,j])
        print schema[j]
        print "   positive: "+str(numpy.mean(pos))+"+/-"+str(numpy.std(pos))
        print "   all: "+str(numpy.mean(all))+"+/-"+str(numpy.std(all))'''

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    #print "hi "+str(spearmanr(qfeatures[:,0], qfeatures[:,11]))
    (m,n) = qfeatures.shape
    pos = []
    neg = []
    asmdrank = []
    for j in range(n):
        pos = []
        neg = []
        all = []
        for i in range(m):
            all.append(qfeatures[i,j])
            if labels[i] == 1:
                pos.append(qfeatures[i,j])
            else:
                neg.append(qfeatures[i,j])
        #if j==9:
            #print "----------"
    
        print schema[j]
        print mannwhitneyu(pos, neg)
        # discriminative power vis
        print "   positive: "+str(numpy.median(pos))+"+/-"+str(numpy.std(pos))
        print "   negative: "+str(numpy.median(neg))+"+/-"+str(numpy.std(neg))
        #if j==18:
            #print pos
            #print neg
        asmdrank.append([math.fabs(numpy.mean(pos)-numpy.mean(neg))/numpy.std(qfeatures[:,j]), schema[j]])
        #asmdrank.append([math.sqrt(2)*math.fabs(numpy.mean(pos)-numpy.mean(neg))/math.sqrt(numpy.var(pos)+numpy.var(neg)), schema[j]])
    asmdrank = sorted(asmdrank, key = lambda asmdrank : asmdrank[0], reverse=True)
    print asmdrank
    
    '''qfeatures_list = qfeatures.tolist()
    cmatrix = numpy.zeros((n, n))
    for i, txt in enumerate(range(n)):
        for j, tx in enumerate(range(n)):
            if j < i:
                p = spearmanr(column(qfeatures_list, i), column(qfeatures_list, j))
                cmatrix[i,j] = p[0]
            elif j == i:
                cmatrix[i,j] = 1
    for i, txt in enumerate(range(n)):
        for j, tx in enumerate(range(n)):
            if j > i:
                cmatrix[i,j] = cmatrix[j,i]

    ind = cluvis(cmatrix, schema)

    best_in_cluster = dict([])
    for j in range(n):
        tc = ind[j]
        if not best_in_cluster.has_key(tc):
            best_in_cluster[tc] = j
    print best_in_cluster
    for bc in best_in_cluster:
        print schema[best_in_cluster[bc]]'''



    '''expert_list = []
    nonexpert_list = []
    for i in range(m):
        tmp=qfeatures[i,:]
        if labels[i] == 1:
            expert_list.append(tmp.tolist())
        else:
            nonexpert_list.append(tmp.tolist())
    #print expert_list
    
    cmatrix = numpy.zeros((n, n))
    for i, txt in enumerate(range(n)):
        print ".."+str(i) +"th feature"
        for j, tx in enumerate(range(n)):
            #if j%10 == 0:
            #print "..."+ str(j) + "th comparing feature"
            if j < i:
                p = spearmanr(column(expert_list, i), column(expert_list,j))
                cmatrix[i,j] = p[0]
            elif j == i:
                cmatrix[i,j] = 1
    for i, txt in enumerate(range(n)):
        for j, tx in enumerate(range(n)):
            if j > i:
                cmatrix[i,j] = cmatrix[j,i]
    
    cmatrix2 = numpy.zeros((n, n))
    for i, txt in enumerate(range(n)):
        print ".."+str(i) +"th feature"
        for j, tx in enumerate(range(n)):
            #if j%10 == 0:
            #print "..."+ str(j) + "th comparing feature"
            if j < i:
                p = spearmanr(column(nonexpert_list, i), column(nonexpert_list,j))
                cmatrix2[i,j] = p[0]
            elif j == i:
                cmatrix2[i,j] = 1
    for i, txt in enumerate(range(n)):
        for j, tx in enumerate(range(n)):
            if j > i:
                cmatrix2[i,j] = cmatrix2[j,i]
    
    cmatrix_dif = cmatrix - cmatrix2
    for i, txt in enumerate(range(n)):
        for j, tx in enumerate(range(n)):
            cmatrix_dif[i,j] = math.fabs(cmatrix_dif[i,j])
    
            cluvis(cmatrix_dif)'''


################################# activity correlation ##########################

def print_cmatrix(cmatrix):
    for i, txt in enumerate(selected_schema):
        for j, tx in enumerate(selected_schema):
            print cmatrix[i,j]

def get_cmatrix(data):
    cmatrix = numpy.zeros((len(selected_schema), len(selected_schema)))
    numpy.zeros((len(selected_schema), len(selected_schema)))
    for i, txt in enumerate(selected_schema):
        print ".."+str(i) +"th feature"
        for j, tx in enumerate(selected_schema):
            if j%10 == 0:
                print "..."+ str(j) + "th comparing feature"
            if j < i:
                p = spearmanr(column(data, i), column(data,j))
                cmatrix[i,j] = p[0]
            elif j == i:
                cmatrix[i,j] = 1
    for i, txt in enumerate(selected_schema):
        for j, tx in enumerate(selected_schema):
            if j > i:
                cmatrix[i,j] = cmatrix[j,i]
            #print p

    return cmatrix


def column(matrix, i):
    return [row[i] for row in matrix]
def cluvis(D, schema):
    # Compute and plot first dendrogram.
    fig = pylab.figure(figsize=(10,10))
    ax1 = fig.add_axes([0.08,0.1,0.2,0.6])
    Y = sch.linkage(D, method='centroid')
    d = sch.distance.pdist(D)
    #cutoff = 0.52*d.max()
    cutoff = d.max()
    ind = sch.fcluster(Y, cutoff, 'distance')
    print ind
    Z1 = sch.dendrogram(Y, labels=schema, color_threshold=cutoff, orientation='right') #leaf_font_size = 0.02, color_threshold=cutoff
    ax1.set_xticks([])
    ax1.set_yticks([])
    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.75,0.6,0.2])
    Y = sch.linkage(D, method='single')
    Z2 = sch.dendrogram(Y, labels=schema, color_threshold=cutoff, leaf_rotation=20)
    #ax2.set_xticks([])
    #ax2.set_yticks([])
    

    # Plot distance matrix.
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = D[idx1,:]
    D = D[:,idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap=pylab.cm.YlGnBu)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    pylab.colorbar(im, cax=axcolor)
    pylab.show()
    fig.savefig('dendrogram.png')

    return ind


################### expert definition  #######################
def nmrr_com():
    nmrrs_r = []
    nmrrs_len = []
    repus_r = []
    repus_len = []
    lens = []
    same = dict([])
    
    nmrrs = []
    repus = []
    all = []
    bads = []
    records = []
    for line in open('nmrr_repu_'):
        nmrr = line.split(",")
        tmp = nmrr[4]
        tmp = int(tmp[0:-1])
        records.append(int(nmrr[2]))
        #print tmp
        if tmp>3:
            if int(nmrr[2]) != int(nmrr[3]):
                lens.append(tmp)
                all.append(int(nmrr[2]))
                if int(nmrr[0])>int(nmrr[1]):
                    #print nmrr
                    bads.append(int(nmrr[2]))
    
    
        if nmrr[2] == nmrr[3]:
            u = int(nmrr[2])
            same[u] = 1

    nrqst = dict([])
    sum = []
    for u in bads:
        nrqst[u] = 0
        for u1 in records:
            if u1 == u:
                nrqst[u] = nrqst[u] + 1
    for u in nrqst:
        sum.append(nrqst[u])
    print numpy.mean(sum)
    print sum

    nrqst2 = dict([])
    sum = []
    for u in all:
        if u in bads:
            continue
        nrqst2[u] = 0
        for u1 in records:
            if u1 == u:
                nrqst2[u] = nrqst2[u] + 1
    for u in nrqst2:
        sum.append(nrqst2[u])
    print numpy.mean(sum)
    print sum


    '''nmrrs_r = []
    nmrrs_len = []
    repus_r = []
    repus_len = []
    lens = []
    same = dict([])

    nmrrs = []
    repus = []
    all = []

    for line in open('nmrr_repu_'):
        nmrr = line.split(",")
        tmp = nmrr[4]
        tmp = int(tmp[0:-1])
        if tmp>1:
            if int(nmrr[2]) != int(nmrr[3]):
                if int(nmrr[2]) in bads:
                    continue
                nmrrs_r.append(int(nmrr[0]))
                repus_r.append(int(nmrr[1]))
                lens.append(tmp)


    index = []
    nmrrs_mean = []
    nmrrs_std = []
    repus_mean = []
    repus_std = []
    for i in range(40):
        index.append(i+1)
        nmrrs = []
        repus = []
        for j in range(len(nmrrs_r)):
            if lens[j] == i+1:
                nmrrs.append(nmrrs_r[j])
                repus.append(repus_r[j])
        
        
        print str(numpy.mean(nmrrs))+"  "+str(numpy.std(nmrrs))
        print str(numpy.mean(repus))+"  "+str(numpy.std(repus))
        
        nmrrs_mean.append(numpy.mean(nmrrs))
        nmrrs_std.append(numpy.std(nmrrs))
        repus_mean.append(numpy.mean(repus))
        repus_std.append(numpy.std(repus))
        print len(nmrrs)
        print i+1
        print '------'
        '''
    #print len(same)
    


    '''for line in open('nmrr_'):
        rank, uid, lens = line.split(",")
        lens = lens[0:-1]
        
        nmrrs_r.append(int(rank))
        nmrrs_len.append(int(lens))

    for line in open('repu_'):
        rank, uid, lens = line.split(",")
        lens = lens[0:-1]
        #print int(rank)
        repus_r.append(int(rank))
        repus_len.append(int(lens))'''

    fig, ax = plt.subplots()
    '''index = []
    nmrrs_mean = []
    nmrrs_std = []
    repus_mean = []
    repus_std = []
    for i in range(40):
        index.append(i+1)
        nmrrs = []
        repus = []
        for j in range(len(nmrrs_r)):
            if nmrrs_len[j] == i+1:
                nmrrs.append(nmrrs_r[j])
        for j in range(len(repus_r)):
            if repus_len[j] == i+1:
               repus.append(repus_r[j])

        print str(numpy.mean(nmrrs))+"  "+str(numpy.std(nmrrs))
        print str(numpy.mean(repus))+"  "+str(numpy.std(repus))
        
        nmrrs_mean.append(numpy.mean(nmrrs))
        nmrrs_std.append(numpy.std(nmrrs))
        repus_mean.append(numpy.mean(repus))
        repus_std.append(numpy.std(repus))
        print len(nmrrs)
        print i+1
        print nmrrs
        print repus
        print '------'
        '''

    ax.errorbar(index, nmrrs_mean, nmrrs_std)
    ax.errorbar(index, repus_mean, repus_std)
    plt.show()
    
    '''fig, ax = plt.subplots()

    ax.scatter(repus_r, repus_len, color='blue')
    ax.scatter(nmrrs_r, nmrrs_len, color='red')
    plt.show()'''


#################################### main function ##############################

if __name__ == '__main__':
    #nmrr_com()
    classify()
    '''activeness = [0.49658225713459475/math.sqrt(2), 0.3983599677335769/math.sqrt(2), 0.4740995962346733/math.sqrt(2), 0.536481011000546/math.sqrt(2), 0.6685221183394102/math.sqrt(2)]
    print str(numpy.mean(activeness))+" "+str(numpy.std(activeness))
    helpfulness = [0.4137234660119773/math.sqrt(2), 0.3482395044673105/math.sqrt(2), 0.3543762388589458/math.sqrt(2), 0.44050052087595654/math.sqrt(2), 0.5452880938771707/math.sqrt(2)]
    print str(numpy.mean(helpfulness))+" "+str(numpy.std(helpfulness))
    seriousness = [0.4241821158001696/math.sqrt(2), 0.5206697070046687/math.sqrt(2), 0.4170023490089895/math.sqrt(2), 0.2749243008437208/math.sqrt(2), 0.20871781545333068/math.sqrt(2)]
    print str(numpy.mean(seriousness))+" "+str(numpy.std(seriousness))
    consciousness = [0.23532479191016364/math.sqrt(2), 0.24960535141201506/math.sqrt(2), 0.22703189711222443/math.sqrt(2), 0.38154993400773185/math.sqrt(2), 0.2946270691433237/math.sqrt(2)]
    print str(numpy.mean(consciousness))+" "+str(numpy.std(consciousness))
    preferenceq = [0.18943023497551906/math.sqrt(2), 0.2368404013366694/math.sqrt(2), 0.3014060002414233/math.sqrt(2), 0.48080153426829964/math.sqrt(2), 0.36304227974928727/math.sqrt(2)]
    print str(numpy.mean(preferenceq))+" "+str(numpy.std(preferenceq))
    preferenced = [0.08572822042491422/math.sqrt(2), 0.1398492552258453/math.sqrt(2), 0.14471647948720137/math.sqrt(2), 0.03961349657164084/math.sqrt(2), 0.1043465872679438/math.sqrt(2)]
    print str(numpy.mean(preferenced))+" "+str(numpy.std(preferenced))
    print "---"
    preferencec = [0.03628537616499327/math.sqrt(2), 0.08515420290208535/math.sqrt(2), 0.1196515247696036/math.sqrt(2), 0.17832222343359352/math.sqrt(2), 0.1328321402750623/math.sqrt(2)]
    print str(numpy.mean(preferencec))+" "+str(numpy.std(preferencec))
    preferencecd = [0.01852147698405111/math.sqrt(2), 0.06424139449554198/math.sqrt(2), 0.04279400231255952/math.sqrt(2), 0.08227067220678504/math.sqrt(2), 0.10418950669556913/math.sqrt(2)]
    print str(numpy.mean(preferencecd))+" "+str(numpy.std(preferencecd))
    activenessc = [0.33524940856770774/math.sqrt(2), 0.3303345673376099/math.sqrt(2), 0.33604777983792344/math.sqrt(2), 0.4464104887028647/math.sqrt(2), 0.5185690443140935/math.sqrt(2)]
    print str(numpy.mean(activenessc))+" "+str(numpy.std(activenessc))
    persistence = [0.14564876237030108/math.sqrt(2), 0.19492617369042364/math.sqrt(2), 0.08141552335197724/math.sqrt(2), 0.00923711110326988/math.sqrt(2), 0.2078484488831471/math.sqrt(2)]
    print str(numpy.mean(persistence))+" "+str(numpy.std(persistence))'''
