'''
    functions in this script serve for complementation of two_property_model
'''
import graphlab
import numpy as np
import math
import sys
import gc 
import copy
from metrics import *
from l2r import *
from two_property_model import *

def data_filter_single_3Property(exp_data, act_data, par_data, test_data):
    train_u = set([r[0] for r in exp_data])
    print '#answerers in Training data: '+str(len(train_u))
    test_u = set([u[0] for qi in test_data for u in qi[2]])
    print '#answerers in Test data: '+str(len(test_u))
    intersect_u = train_u.intersection(test_u)
    print '#answerers in both sets: '+str(len(intersect_u))
    
    exp_data_new = []
    act_data_new = []
    par_data_new = []
    test_data_new = []

    for r in exp_data:
        if r[0] in intersect_u:
            exp_data_new.append(r)
    del exp_data
    for r in act_data:
        if r[0] in intersect_u:
            act_data_new.append(r)
    del act_data
    for r in par_data:
        if r[0] in intersect_u:
            par_data_new.append(r)
    del par_data
    for r in test_data:
        r_new = copy.deepcopy(r)
        for us in r[2]:
            if us[0] not in intersect_u:
                r_new[2].remove(us)
        if len(r_new[2])>=2:
            test_data_new.append(r_new)
    del test_data
    #print test_data_new
    print 'AFTER the first filtering: removing the answerers not in the intersection of the two sets; delete questions<2 answerers in the test set.'
    train_u = set([r[0] for r in exp_data_new])
    print '#answerers in Training data: '+str(len(train_u))
    test_u = set([u[0] for qi in test_data_new for u in qi[2]])
    print '#answerers in Test data: '+str(len(test_u))
    intersect_u = train_u.intersection(test_u)
    print '#answerers in both sets: '+str(len(intersect_u))
    

    exp_data = []
    act_data = []
    par_data = []
    test_data = []
    for r in exp_data_new:
        if r[0] in intersect_u:
            exp_data.append(r)
    del exp_data_new
    for r in act_data_new:
        if r[0] in intersect_u:
            act_data.append(r)
    del act_data_new
    for r in par_data_new:
        if r[0] in intersect_u:
            par_data.append(r)
    del par_data_new
    test_data = test_data_new
    del test_data_new
    print 'remaining test questions: '+str(len(test_data))
    
    print 'AFTER the second filtering: remove again the answerers in the training set, who do not show up in the test set.'
    train_u = set([r[0] for r in exp_data])
    print '#answerers in Training data: '+str(len(train_u))
    test_u = set([u[0] for qi in test_data for u in qi[2]])
    print '#answerers in Test data: '+str(len(test_u))
    intersect_u = train_u.intersection(test_u)
    print '#answerers in both sets: '+str(len(intersect_u))
    return exp_data, act_data, par_data, test_data  

def answerer_rec_train_L2R_3Property(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score, par_dict_uid, par_dict_score):#, part):
    act_dict = dict([])
    for t in act_dict_uid:
        act_dict[t] = dict([])
        us_act = act_dict_uid[t]
        scores_act = act_dict_score[t]
        for j in range(len(us_act)): 
            u_act = us_act[j]
            score_act = scores_act[j]
            act_dict[t][u_act] = score_act
        
    del act_dict_uid
    del act_dict_score
    
    par_dict = dict([])
    for t in par_dict_uid:
        par_dict[t] = dict([])
        us_par = par_dict_uid[t]
        scores_par = par_dict_score[t]
        for j in range(len(us_par)): 
            u_par = us_par[j]
            score_par = scores_par[j]
            par_dict[t][u_par] = score_par
        
    del par_dict_uid
    del par_dict_score
    
    tag_dict = dict([])
    for qinfo in test_data:
        qid = qinfo[0]
        qtags = qinfo[1]
        qscores = qinfo[2]
        for t in qtags:
            if t in tag_dict:
                tag_dict[t].append(qinfo)
            else:
                tag_dict[t] = [qinfo]
    del test_data
    gc.collect()
    
    t_set = [t for t in tag_dict]
    
    alphas = dict([])
    
    print 'total number of tags: '+str(len(tag_dict))
    k = 0
    for t in t_set:
        #t = 'java'#this is for debug
        alphas_t = []
        k += 1
        if k%100 == 0:
            print '...'+str(k)+'th tag trained.'
        alphas_t = get_coefficient_L2R_per_tag_3Property(t, tag_dict[t], exp_dict_uid, exp_dict_score, act_dict, par_dict)
        #print alphas_t
        if alphas_t==[]:
            continue
        #if len(alphas_t[0]) < 2:
            #continue
        alphas[t] = alphas_t
        #print t, alphas_t
    return alphas

def get_coefficient_L2R_per_tag_3Property(t, test_data, exp_dict_uid, exp_dict_score, act_dict, par_dict):
    alphas_t = []
    gt_test = test_data
    if len(gt_test) == 0 or t not in exp_dict_uid:
        return alphas_t
    
    X = []
    Y = []
    for qinfo in test_data:
        qid = qinfo[0]
        qusers = [qu[0] for qu in qinfo[2]]
        
        us = exp_dict_uid[t]
        scores = exp_dict_score[t]
        
        overlapped_users = set(qusers).intersection(set(us))
        for i in xrange(len(us)):
            u = us[i]
            score = scores[i]
            if u not in overlapped_users:
                continue
            
            if t in par_dict:
                if u in par_dict[t]:
                    this_X = [act_dict[t][u], score, float(par_dict[t][u])/act_dict[t][u]]
                else:
                    this_X = [act_dict[t][u], score, 0]
            else:
                this_X = [act_dict[t][u], score, 0]
            this_Y = [get_score_in_gt(u,qinfo[2]),qid]
            X.append(this_X)
            Y.append(this_Y)
        del us
        del scores
    if len(X)<2:
        return alphas_t
    try:
        rank_svm = RankSVM().fit(np.array(X), np.array(Y))
    except:
        #print Y
        return alphas_t
    print 'Tag ',t, '#Q', len(X),'Coefficients', rank_svm.coef_
    
    alphas_t = rank_svm.coef_
    return alphas_t

def get_score_in_gt(u,gt):
    for item in gt:
        if item[0]==u:
            return item[1]
        
def answerer_rec_using_trained_model_L2R_3Property(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score, par_dict_uid, par_dict_score, model, alpha_default = 0.5):
    act_dict = dict([])
    for t in act_dict_uid:
        act_dict[t] = dict([])
        us_act = act_dict_uid[t]
        scores_act = act_dict_score[t]
        for j in range(len(us_act)): 
            u_act = us_act[j]
            score_act = scores_act[j]
            act_dict[t][u_act] = score_act
        
    del act_dict_uid
    del act_dict_score
    
    par_dict = dict([])
    for t in par_dict_uid:
        par_dict[t] = dict([])
        us_par = par_dict_uid[t]
        scores_par = par_dict_score[t]
        for j in range(len(us_par)): 
            u_par = us_par[j]
            score_par = scores_par[j]
            par_dict[t][u_par] = score_par
        
    del par_dict_uid
    del par_dict_score
    
    k = 0
    predicted = []
    for qinfo in test_data:
        k += 1
        if k%1000 == 0:
            print '...'+str(k)+'th question predicted.'
        qid = qinfo[0]
        qtags = qinfo[1]
        qusers = [qu[0] for qu in qinfo[2]]
        qrec = dict([])
        nr_tags = 0
        us = []
        scores = []
        for qt in qtags:
            if qt not in t_set:
                continue
            nr_tags += 1

            us = exp_dict_uid[qt]
            scores = exp_dict_score[qt]
            
            for i in range(len(us)):
                u = us[i]
                score = scores[i]
                if u not in qusers:
                    continue
                alpha1 = alpha_default
                alpha2 = alpha_default
                alpha3 = alpha_default
                if qt in model:
                    alpha1 = model[qt][0]
                    alpha2 = model[qt][1]
                    alpha3 = model[qt][2]
                if u in qrec:
                    qrec[u] += alpha2*score
                else:
                    qrec[u] = alpha2*score
                if u in act_dict[qt]:
                    qrec[u] += alpha1*act_dict[qt][u]
                else:
                    pass
                if qt in par_dict:
                    if u in par_dict[qt]:
                        qrec[u] += alpha3*(float(par_dict[qt][u])/act_dict[qt][u])
                    else:
                        pass
                else:
                    pass
                

        final_scores = [[u,qrec[u]] for u in qrec]
        final_scores = sorted(final_scores, key=lambda final_scores : final_scores[1], reverse=True)
        #final_scores = final_scores[0:len(qinfo[2])]
        predicted.append([qid,final_scores])
        
    return predicted