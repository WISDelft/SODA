import string
import psycopg2
import sys
sys.path.append(sys.path[0]+'/lib/')
import pickle
import os
import numpy as np
import random
import gc 

from load_data import *
from two_property_model import *
from metrics import *
def dumpfile(data, fname):
    f = open('./parameter_gridSearch/'+fname+".pik", 'w')
    pickle.dump(data, f)
    f.close()
def loadfile(fname):
    f = open('../train_data/'+fname+".pik")
    data = pickle.load(f)
    f.close()
    return data

if __name__ == '__main__':
    #print 'part = '+sys.argv[2]
    exp_data = loadfile(sys.argv[1])
    act_data = loadfile('act_data')
    test_data = loadfile('test_data')
    exp_data, act_data, test_data = \
        data_filter_single(exp_data, act_data, test_data)
    #u_set = set([em[0] for em in exp_data])
    t_set = [em[1] for em in exp_data]
    ############################################ MODEL&PREDICT ############################
    augment = False
    if len(sys.argv)>=3 and sys.argv[2]=='True':
        augment = True
    print augment
    #exp_all_dict_uid, exp_all_dict_score = separate_mat(exp_data_all, augment)
    exp_dict_uid, exp_dict_score = separate_mat_log(exp_data, augment)
    act_dict_uid, act_dict_score = separate_mat_log(act_data, augment)
    
    #del exp_data_all
    del exp_data
    del act_data
    gc.collect()
    # predict ranked answerer list for each test question
    alphas, alphas_rank, each_alpha_perform = answerer_rec_train(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score)#, int(sys.argv[2]))

    dumpfile(alphas, 'log_optimal_alphas_'+sys.argv[1])#+'_'+sys.argv[2])
    dumpfile(alphas_rank, 'log_optimal_alphas_rank_'+sys.argv[1])#+'_'+sys.argv[2])
    dumpfile(each_alpha_perform, 'log_each_alpha_perform_'+sys.argv[1])