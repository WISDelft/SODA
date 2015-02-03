'''
From User Side:
    1. Distribution of signals
    2. Correlation between signals
From Topic Side
    1. Difference between topics 
        - properties of the best answerers
    
Assumption:
    - easy questions <- more active users <- windows
    - difficult question <- more expert users <- 
    - debatable question <- more interested users <- OOP
'''

import string
import psycopg2
import sys
import pickle
import os
import numpy as np
import random
from scipy.stats import spearmanr,pearsonr

def loadfile(fname):
    f = open('../test_data/'+fname+".pik")
    data = pickle.load(f)
    f.close()
    return data

def get_u_info(u, dct):
    if u in dct:
        info = dct[u]
    else:
        info = 0
    return info
def tuple_to_dict(t, tuples):
    dct = dict([])
    for tp in tuples:
        if tp[1]==t:
            dct[tp[0]]=tp[2]
    return dct

def get_gt(t, test_data):
    tag_questions = []
    for qinfo in test_data:
        qid = qinfo[0]
        qtags = qinfo[1]
        qscores = qinfo[2]
        if t in qtags:
            tag_questions.append(qscores)
    return tag_questions
    
def get_rank(par, tq, dct):
    par_list = []
    for us in tq:
        par_list.append(get_u_info(us[0], dct))
    par_list = sorted(par_list, reverse=True)
    return par_list.index(par)
    
def get_rank_all(value, dct_list):
    i = 0
    for i in xrange(len(dct_list)):
        if value>=dct_list[i]:
            return float(i)/len(dct_list)
    #print value, dct_list[i], i, len(dct_list)
    return float(i)/len(dct_list)
        
def get_pearson(dct, tq):
    gr = []
    pr = []
    for t in tq:
        u = t[0]
        if u in dct:
            pr.append(dct[u])
            gr.append(t[1])
    return pearsonr(gr,pr)[0]

def normalize_dct(par_dict, act_dict):
    par_dict_new = dict([])
    for u in par_dict:
        if u in act_dict and act_dict[u]!=0:
            par_dict_new[u] = float(par_dict[u])/act_dict[u]
    return par_dict_new
    
def show_distribution_correaltion(t, act_data, exp_data, par_data, exp_name):
    act_dict = tuple_to_dict(t, act_data)
    exp_dict = tuple_to_dict(t, exp_data)
    par_dict = tuple_to_dict(t, par_data)
    #par_dict = normalize(par_dict, act_dict)
    
    uset = set([u for u in act_dict]).union(set([u for u in exp_dict]).union(set([u for u in par_dict])))
    print 'Topic: ', t,'Total number of users: ', len(uset)
    f = open('exploratory_analysis/par_normed_'+exp_name+'_users_'+t+'.csv', 'wb')
    f.write('tag,user,act,exp,par\n')
    for u in uset:
        act = get_u_info(u, act_dict)
        exp = get_u_info(u, exp_dict)
        par = get_u_info(u, par_dict)
        f.write(t+','+str(u)+','+str(act)+','+str(exp)+','+str(par)+'\n')
    f.close()
    return act_dict, exp_dict, par_dict


def explore_user_topic_relation(t, test_data, act_dict, exp_dict, par_dict, exp_name):
    par_dict_norm = normalize_dct(par_dict, act_dict)
    
    act_dict_list = sorted([act_dict[u] for u in act_dict], reverse=True)
    exp_dict_list = sorted([exp_dict[u] for u in exp_dict], reverse=True)
    par_dict_list = sorted([par_dict[u] for u in par_dict], reverse=True)
    par_dict_list_norm = sorted([par_dict_norm[u] for u in par_dict_norm], reverse=True)
    
    tag_questions = get_gt(t, test_data)
    #u_best = [us[0] for us in tag_questions]
    print 'Topic: ', t,'Total number of questions: ', len(tag_questions)
    f = open('exploratory_analysis/par_normed_'+exp_name+'_topics_'+t+'.csv', 'wb')
    f.write('tag,NoQ,bestUid,act,exp,par,r_act,r_exp,r_par,pear_act,pear_exp,pear_par\n')
    for tq in tag_questions:
        tq = sorted(tq, key=lambda tq : tq[1], reverse=True)
        u_best = tq[0][0]
        
        act = get_u_info(u_best, act_dict)
        exp = get_u_info(u_best, exp_dict)
        par = get_u_info(u_best, par_dict)
        
        '''r_act = get_rank(act, tq, act_dict)
        r_exp = get_rank(exp, tq, exp_dict)
        r_par = get_rank(par, tq, par_dict)'''
        
        r_act_all = get_rank_all(act, act_dict_list)
        r_exp_all = get_rank_all(exp, exp_dict_list)
        if act == 0:
            r_par_all = None
        else:
            r_par_all = get_rank_all(float(par)/act, par_dict_list_norm)
        
        pear_act = get_pearson(act_dict, tq)
        pear_exp = get_pearson(exp_dict, tq)
        pear_par = get_pearson(par_dict_norm, tq)
        #print r_act, r_exp, r_par
        f.write(t+','+str(len(tq))+','+str(u_best)+','+\
                str(act)+','+str(exp)+','+str(par)+','+\
                #str(r_act)+','+str(r_exp)+','+str(r_par)+','+\
                str(r_act_all)+','+str(r_exp_all)+','+str(r_par_all)+','+\
                str(pear_act)+','+str(pear_exp)+','+str(pear_par)+'\n')
    f.close()
    return 0

def user_topic_analysis(t_test_set, exp_name):
    exp_data = loadfile(exp_name)
    act_data = loadfile('act_data')
    par_data = loadfile('parti_data_act')
    test_data = loadfile('test_data')
    
    for t in t_test_set:
        act_dict, exp_dict, par_dict = show_distribution_correaltion(t, act_data, exp_data, par_data, exp_name)
        explore_user_topic_relation(t, test_data, act_dict, exp_dict, par_dict, exp_name)
        
if __name__ == '__main__':
    t_test_set = ['c#', 'asp.net-mvc', 'windows', 'oop', 'regex', 'assembly']
    user_topic_analysis(t_test_set, sys.argv[1])