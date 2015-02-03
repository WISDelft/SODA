import numpy as np
import string
import scipy.stats
from random import randrange
from util_mec import *

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='wistudelft')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()

def robust_component(answerer_scores, tag):
    userranklists = loadfile("userranklists_"+tag)
    nrans = loadfile('nrans_'+tag)
    # 1. question size, 2. nrans, 3. rank, 4. 1/rank
    debate = dict([])
    rank = dict([])
    rank_inverse = dict([])
    for urank in userranklists:
        for i in range(len(urank)):
            this_u = urank[i]
            u = this_u[0]
            if debate.has_key(u):
                debate[u].append(len(urank))
                rank[u].append(i+1)
                rank_inverse[u].append(float(1)/(i+1))
            else:
                debate[u]=[len(urank)]
                rank[u] = [i+1]
                rank_inverse[u] = [float(1)/(i+1)]
    f = open('data/all_robustness.csv', 'w')
    f_out = open('data/out_robustness.csv', 'w')
    f_owl = open('data/owl_robustness.csv', 'w')
    for u in answerer_scores:
        #if randrange(10)==1:
        f.write(str(answerer_scores[u])+','+str(nrans[u])\
                +','+str(np.mean(debate[u]))+','+str(np.median(debate[u]))+','+str(np.std(debate[u]))\
                +','+str(np.mean(rank[u]))+','+str(np.median(rank[u]))+','+str(np.std(rank[u]))\
                +','+str(np.mean(rank_inverse[u]))+','+str(np.median(rank_inverse[u]))+','+str(np.std(rank_inverse[u]))+'\n')
        if nrans[u] == 1: #and randrange(4)==1:
            f_out.write(str(answerer_scores[u])+','+str(nrans[u])\
                +','+str(np.mean(debate[u]))+','+str(np.median(debate[u]))+','+str(np.std(debate[u]))\
                +','+str(np.mean(rank[u]))+','+str(np.median(rank[u]))+','+str(np.std(rank[u]))\
                +','+str(np.mean(rank_inverse[u]))+','+str(np.median(rank_inverse[u]))+','+str(np.std(rank_inverse[u]))+'\n')
        if answerer_scores[u] >= 1: #and randrange(4)==1:
            f_owl.write(str(answerer_scores[u])+','+str(nrans[u])\
                +','+str(np.mean(debate[u]))+','+str(np.median(debate[u]))+','+str(np.std(debate[u]))\
                +','+str(np.mean(rank[u]))+','+str(np.median(rank[u]))+','+str(np.std(rank[u]))\
                +','+str(np.mean(rank_inverse[u]))+','+str(np.median(rank_inverse[u]))+','+str(np.std(rank_inverse[u]))+'\n')
    
    print "mec-nrans : "+str(scipy.stats.pearsonr(dict2list(answerer_scores), dict2list(nrans)))
    print "mec-debate_mean : "+str(scipy.stats.pearsonr(dict2list(answerer_scores), dict2list_mean(debate)))
    print "mec-debate_median : "+str(scipy.stats.pearsonr(dict2list(answerer_scores), dict2list_median(debate)))
    print "mec-debate_std : "+str(scipy.stats.pearsonr(dict2list(answerer_scores), dict2list_std(debate)))
    print "mec-rank_mean : "+str(scipy.stats.pearsonr(dict2list(answerer_scores), dict2list_mean(rank)))
    print "mec-rank_median : "+str(scipy.stats.pearsonr(dict2list(answerer_scores), dict2list_median(rank)))
    print "mec-rank_std : "+str(scipy.stats.pearsonr(dict2list(answerer_scores), dict2list_std(rank)))
    print "mec-rank_inverse_mean : "+str(scipy.stats.pearsonr(dict2list(answerer_scores), dict2list_mean(rank_inverse)))
    print "mec-rank_inverse_median : "+str(scipy.stats.pearsonr(dict2list(answerer_scores), dict2list_median(rank_inverse)))
    print "mec-rank_inverse_std : "+str(scipy.stats.pearsonr(dict2list(answerer_scores), dict2list_std(rank_inverse)))
    print 'owls: -----------------------'
    print "mec-nrans : "+str(scipy.stats.pearsonr(dict2list_filter(answerer_scores, answerer_scores), dict2list_filter(nrans, answerer_scores)))
    print "mec-debate_mean : "+str(scipy.stats.pearsonr(dict2list_filter(answerer_scores, answerer_scores), dict2list_filter_mean(debate, answerer_scores)))
    print "mec-debate_median : "+str(scipy.stats.pearsonr(dict2list_filter(answerer_scores, answerer_scores), dict2list_filter_median(debate, answerer_scores)))
    print "mec-debate_std : "+str(scipy.stats.pearsonr(dict2list_filter(answerer_scores, answerer_scores), dict2list_filter_std(debate, answerer_scores)))
    print "mec-rank_mean : "+str(scipy.stats.pearsonr(dict2list_filter(answerer_scores, answerer_scores), dict2list_filter_mean(rank, answerer_scores)))
    print "mec-rank_median : "+str(scipy.stats.pearsonr(dict2list_filter(answerer_scores, answerer_scores), dict2list_filter_median(rank, answerer_scores)))
    print "mec-rank_std : "+str(scipy.stats.pearsonr(dict2list_filter(answerer_scores, answerer_scores), dict2list_filter_std(rank, answerer_scores)))
    print "mec-rank_inverse_mean : "+str(scipy.stats.pearsonr(dict2list_filter(answerer_scores, answerer_scores), dict2list_filter_mean(rank_inverse, answerer_scores)))
    print "mec-rank_inverse_median : "+str(scipy.stats.pearsonr(dict2list_filter(answerer_scores, answerer_scores), dict2list_filter_median(rank_inverse, answerer_scores)))
    print "mec-rank_inverse_std : "+str(scipy.stats.pearsonr(dict2list_filter(answerer_scores, answerer_scores), dict2list_filter_std(rank_inverse, answerer_scores)))
    print 'out: -----------------------'
    print "mec-nrans : "+str(scipy.stats.pearsonr(dict2list_filter2(answerer_scores, nrans), dict2list_filter2(nrans, nrans)))
    print "mec-debate_mean : "+str(scipy.stats.pearsonr(dict2list_filter2(answerer_scores, nrans), dict2list_filter2_mean(debate, nrans)))
    print "mec-debate_median : "+str(scipy.stats.pearsonr(dict2list_filter2(answerer_scores, nrans), dict2list_filter2_median(debate, nrans)))
    print "mec-debate_std : "+str(scipy.stats.pearsonr(dict2list_filter2(answerer_scores, nrans), dict2list_filter2_std(debate, nrans)))
    print "mec-rank_mean : "+str(scipy.stats.pearsonr(dict2list_filter2(answerer_scores, nrans), dict2list_filter2_mean(rank, nrans)))
    print "mec-rank_median : "+str(scipy.stats.pearsonr(dict2list_filter2(answerer_scores, nrans), dict2list_filter2_median(rank, nrans)))
    print "mec-rank_std : "+str(scipy.stats.pearsonr(dict2list_filter2(answerer_scores, nrans), dict2list_filter2_std(rank, nrans)))
    print "mec-rank_inverse_mean : "+str(scipy.stats.pearsonr(dict2list_filter2(answerer_scores, nrans), dict2list_filter2_mean(rank_inverse, nrans)))
    print "mec-rank_inverse_median : "+str(scipy.stats.pearsonr(dict2list_filter2(answerer_scores, nrans), dict2list_filter2_median(rank_inverse, nrans)))
    print "mec-rank_inverse_std : "+str(scipy.stats.pearsonr(dict2list_filter2(answerer_scores, nrans), dict2list_filter2_std(rank_inverse, nrans)))
    
    return 0