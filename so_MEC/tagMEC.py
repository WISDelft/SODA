'''
    TODO:
        1. preference & temporal code cleaning
        2. mec for individuals
'''

import string
import psycopg2
import sys
import pickle
import os
import numpy as np


from calculate_mec import *
from compare_performance import *
from characterize_perference import *
from get_userAtt import *
from robustness_component import *
from util_mec import *
from get_motivation import *

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


def get_sparrows(tag):
    userranklists = loadfile("userranklists_"+tag)
    nrans = dict([])
    for urank in userranklists:
        for i in range(len(urank)):
            this_u = urank[i]
            u = this_u[0]
            if nrans.has_key(u):
                nrans[u] += 1
            else:
                nrans[u] = 1
    dumpfile(nrans, 'nrans_'+tag)
    sparrows = []
    for u in nrans:
        if nrans[u]>=10:
            sparrows.append(u)
    dumpfile(sparrows, 'sparrows_'+tag)
    return sparrows
    
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

    
    # get sparrow set and motivating plots
    sparrows = get_sparrows(tag)
    print "nr of experts: "+str(len(experts))+', nr of sparrows: '+str(len(sparrows))
    
    
    gene_question_agg_get_randomUser(tag)
    sys.exit(1)
    
    #get_motiv_example(experts, answers_att)                                           # table 2
    if not os.path.exists("data/illustration.csv"):
        get_motiv_scatter(experts, answerer_scores, tag)                             # figure 3
    sys.exit(1)
    # analyze robustness of the metric 1. in terms of components, and 2. cross-topic
    robust_component(answerer_scores, tag)
    sys.exit(1)
    
    # get basic statistics
    get_performance(experts, answerer_scores, sparrows, tag, 'sparrow')                              # figure 6
    
    # characterize preference
    ansd_qst_properties(experts, sparrows, answerer_scores, tag)
    post_qst_properties(experts, sparrows, answerer_scores, tag)
    
    
    #temp_cluster(users, experts, tag)
    #temp_analyze(users, experts, tag)
    #
    #experts_explore(users, experts, tag)
    #comlook()
    


