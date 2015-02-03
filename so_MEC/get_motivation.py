import string
import sys
import numpy
import pickle
import math
import numpy as np
import os
import copy
from util_mec import *

def gene_answerer_question_pair(tag):
    userranklists = loadfile("userranklists_"+tag)
    nrans = loadfile("nrans_"+tag)
    f = open('data/answerer_questions_pair_'+tag, 'w')
    f.write('Uid,#AnswersByU,Rank,Qid,#AnswersToQ\n')
    qid = 0
    for userrank in userranklists:
        qid += 1
        for i in range(len(userrank)):
            uid = userrank[i][0]
            urank = i+1
            f.write(str(uid)+','+str(nrans[uid])+','+str(urank)+','+str(qid)+','+str(len(userrank))+'\n')
    f.close()
    return 0

def median_std_k_in_n_consecutive_numbers(n, k):
    #print '---------------------------------'
    ranklist = []
    for i in range(n):
        ranklist.append(i+1)
    #print ranklist
    stds = []
    medians = []
    for m in range(1000):
        ar = copy.deepcopy(ranklist)
        #ar = numpy.random.randint(1,n+1,size=k)
        np.random.shuffle(ar)
        #print ar
        ar = ar[:k]
        #print ar
        std_one_step = numpy.std(ar)
        stds.append(std_one_step)
        median_one_step = numpy.median(ar)
        medians.append(median_one_step)
    #print str(k+1)+' mean stds: '+str(numpy.mean(stds))+';std stds: '+str(numpy.std(stds))
    #print str(k+1)+' mean medians: '+str(numpy.mean(medians))+';std stds: '+str(numpy.std(medians))
    return numpy.mean(medians), numpy.mean(stds)

# for each question, compare the averaged rank of sparrows, and averaged rank of randomly selected answerers to the question
def gene_question_agg(tag):
    userranklists = loadfile("userranklists_"+tag)
    nrans = loadfile("nrans_"+tag)
    f = open('data/questions_agg_'+tag, 'w')
    f.write('Qid,#AnswersToQ,#SparrowAnswers,Median_rank_sparrow,Std_rank_sparrow,Median_rank_others,Std_rank_others\n')
    qid = 0
    for userrank in userranklists:
        qid += 1
        no_sparrow_answers = 0
        sparrow_ranks = []
        for i in range(len(userrank)):
            uid = userrank[i][0]
            if nrans[uid]>=10:
                no_sparrow_answers += 1
                sparrow_ranks.append(i+1)
        Median_rank_others, Std_rank_others = median_std_k_in_n_consecutive_numbers(len(userrank), no_sparrow_answers)
        f.write(str(qid)+','+str(len(userrank))+','+str(no_sparrow_answers)+','+str(numpy.median(sparrow_ranks))+','+str(numpy.std(sparrow_ranks))+','+\
                                 str(Median_rank_others)+','+str(Std_rank_others)+'\n')
    f.close()
        
    return 0

def gene_question_agg_get_randomUser(tag):
    userranklists = loadfile("userranklists_"+tag)
    nrans = loadfile("nrans_"+tag)
    selectedU = []
    for u in nrans:
        selectedU.append(u)
    np.random.shuffle(selectedU)
    selectedU = selectedU[:17072]
    
    f = open('data/questions_agg_randomUser_'+tag, 'w')
    f.write('Qid,#AnswersToQ,#SparrowAnswers,Median_rank_sparrow,Std_rank_sparrow,Median_rank_others,Std_rank_others\n')
    qid = 0
    for userrank in userranklists:
        qid += 1
        no_sparrow_answers = 0
        sparrow_ranks = []
        selected_ranks = []
        for i in range(len(userrank)):
            uid = userrank[i][0]
            if nrans[uid]>=10:
                no_sparrow_answers += 1
                sparrow_ranks.append(i+1)
            if uid in selectedU:
                selected_ranks.append(i+1)
        f.write(str(qid)+','+str(len(userrank))+','+str(no_sparrow_answers)+','+str(numpy.median(sparrow_ranks))+','+str(numpy.std(sparrow_ranks))+','+\
                                str(numpy.median(selected_ranks))+','+str(numpy.std(selected_ranks))+'\n')
    f.close()
        
    return 0

def gene_answerer_agg(tag):
    return 0

    
if __name__ == '__main__':
    #for i in range(15):
    print median_std_k_in_n_consecutive_numbers(5,0)