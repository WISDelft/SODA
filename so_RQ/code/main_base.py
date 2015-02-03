'''
MUST start with toy data!!
'''

import string
import psycopg2
import sys
sys.path.append(sys.path[0]+'/../lib_multi/')
import pickle
import os
import numpy as np
import random

from load_data import *
from two_property_model import *
from metrics import *
def dumpfile(data, fname):
    f = open('./'+fname+".pik", 'w')
    pickle.dump(data, f)
    f.close()
def loadfile(fname):
    f = open('../'+fname+".pik")
    data = pickle.load(f)
    f.close()
    return data

def loadfile_here(fname):
    f = open('./'+fname+".pik")
    data = pickle.load(f)
    f.close()
    return data

def dump_all_data_split_from_2013():
    exp_data_repu,exp_data_repu_norm,act_data, exp_data_mec_naive,exp_data_mec, exp_data_mec_log = load_train_data()
    dumpfile(exp_data_mec_naive,'exp_data_mec_naive')
    dumpfile(exp_data_mec,'exp_data_mec')
    dumpfile(exp_data_mec_log,'exp_data_mec_log')
    dumpfile(exp_data_repu,'exp_data_repu')
    dumpfile(exp_data_repu_norm,'exp_data_repu_norm')
    dumpfile(act_data,'act_data')
    
    test_data = load_test_data()
    dumpfile(test_data, 'test_data')
    
    sys.exit(1)
    
def pre_process_data():
    exp_data_repu = loadfile('exp_data_repu')
    exp_data_repu_norm = loadfile('exp_data_repu_norm')
    act_data = loadfile('act_data')
    exp_data_mec = loadfile('exp_data_mec')
    exp_data_mec_log = loadfile('exp_data_mec_log')
    exp_data_mec_naive = loadfile('exp_data_mec_naive')
    
    test_data = loadfile('test_data')
    
    #random.shuffle(test_data)
    #test_data = test_data[0:100000]

    exp_data_repu, exp_data_repu_norm, act_data, test_data, exp_data_mec_naive, exp_data_mec, exp_data_mec_log = \
        data_filter(exp_data_repu, exp_data_repu_norm, act_data, test_data, exp_data_mec_naive, exp_data_mec, exp_data_mec_log)
    
    dumpfile(exp_data_mec_naive,'exp_data_mec_naive')
    dumpfile(exp_data_mec,'exp_data_mec')
    dumpfile(exp_data_mec_log,'exp_data_mec_log')
    dumpfile(exp_data_repu,'exp_data_repu')
    dumpfile(exp_data_repu_norm,'exp_data_repu_norm')
    dumpfile(act_data,'act_data')
    dumpfile(test_data, 'test_data')
    sys.exit(1)

if __name__ == '__main__':
    '''
    exp_data=[(1,'c#',10),(2,'c#',5),(3,'c#',1),(4,'c#',0)]
    act_data=[(1,'c#',1),(2,'c#',2),(3,'c#',1),(4,'c#',1)]
    exp_data_all=[(1,'c#',10),(2,'c#',10),(3,'c#',1),(4,'c#',0)]
    test_data = [[1,['c#'],[[1,2],[4,1]]],[2,['c#'],[[3,1],[1,2],[4,1]]]]
    '''
    ############################################ LOAD DATA ############################ 
    #dump_all_data_split_from_2013()
    #pre_process_data()
    
    #exp_data_all = loadfile('exp_data_all')
    print sys.argv[1]
    exp_data = loadfile(sys.argv[1])
    act_data = loadfile('act_data')
    test_data = loadfile('test_data')
    
    exp_data, act_data, test_data = \
        data_filter_single(exp_data, act_data, test_data)
    
    u_set = set([em[0] for em in exp_data])
    t_set = set([em[1] for em in exp_data])
    ############################################ MODEL&PREDICT ############################
    augment = False
    if len(sys.argv)>=4 and sys.argv[3]=='True':
        augment = True
    print augment
    #exp_all_dict_uid, exp_all_dict_score = separate_mat(exp_data_all, augment)
    exp_dict_uid, exp_dict_score = separate_mat(exp_data, augment)
    act_dict_uid, act_dict_score = separate_mat(act_data, augment)
    
    #del exp_data_all
    del exp_data
    del act_data
    
    # predict ranked answerer list for each test question
    alpha = float(float(sys.argv[2]))
    print 'ALPHA = '+str(alpha)
    
    #model = loadfile_here('optimal_alphas_'+sys.argv[1])#+'_'+str(p))
    
    predicted = answerer_rec_base_old(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score, alpha)
    '''predicted_multi = answerer_rec_base_old(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score, alpha)
    #print predicted_old
    evaluate(test_data, predicted_multi)
    del exp_dict_uid
    del exp_dict_score
    del act_dict_uid
    del act_dict_score
    
    predicted_all = answerer_rec_base(test_data, t_set, exp_all_dict_uid, exp_all_dict_score)
    evaluate(test_data, predicted_all)
    #print predicted_all
    del exp_all_dict_uid
    del exp_all_dict_score'''
    ############################################ MODEL EVALUATION ############################
    del exp_dict_uid
    del exp_dict_score
    del act_dict_uid
    del act_dict_score
    evaluate(test_data, predicted)
    evaluate(test_data, predicted, 'rank')