import graphlab
import numpy as np
import math
import sys
import gc 
from metrics import *
from l2r import *

def normalize_exp(mat):
    new_mat = []
    
    values = []
    for record in mat:
        values.append(record[2])
    mean_v = np.mean(values)
    std_v = np.std(values)
    for record in mat:
        new_record = list(record)
        new_record[2] = float(new_record[2]-mean_v)/std_v
        new_mat.append(new_record)        
    
    return new_mat

def get_log(mat):
    new_mat = []
    for record in mat:
        new_record = list(record)
        if new_record[2]+1<=0:
            new_record[2] = 0
            new_mat.append(new_record)
            continue
        new_record[2] = math.log(new_record[2]+1,10)
        new_mat.append(new_record)
    return new_mat

def normalize_exp_log(mat):
    mat = get_log(mat)

    new_mat = []
    
    values = []
    for record in mat:
        values.append(record[2])
    mean_v = np.mean(values)
    std_v = np.std(values)
    for record in mat:
        new_record = list(record)
        new_record[2] = float(new_record[2]-mean_v)/std_v
        new_mat.append(new_record)        
    
    return new_mat

def separate_mat(mat, augment=False):
    if augment:
        print 'start augmenting user profiles...'
        aug_mat = augment_mat(mat)
    mat = normalize_exp(mat)
    
    dict_uid = dict([])
    dict_score = dict([])
    for record in mat:
        u = record[0]
        t = record[1]
        score = record[2]
        if t in dict_uid:
            dict_uid[t].append(u)
            dict_score[t].append(score)
            # exp_u_dict[t].add(u)
        else:
            dict_uid[t] = [u]
            dict_score[t] = [score]
    
    if augment:
        aug_mat = normalize_exp(aug_mat)
        for record in aug_mat:
            u = record[0]
            t = record[1]
            score = record[2]
            assert t in dict_uid
            assert u not in dict_uid[t]
            dict_uid[t].append(u)
            dict_score[t].append(score)

    return dict_uid, dict_score

def separate_mat_base(mat, augment=False):
    if augment:
        print 'start augmenting user profiles...'
        aug_mat = augment_mat(mat)
    #mat = normalize_exp(mat)
    
    dict_uid = dict([])
    dict_score = dict([])
    for record in mat:
        u = record[0]
        t = record[1]
        score = record[2]
        if t in dict_uid:
            dict_uid[t].append(u)
            dict_score[t].append(score)
            # exp_u_dict[t].add(u)
        else:
            dict_uid[t] = [u]
            dict_score[t] = [score]
    
    if augment:
        #aug_mat = normalize_exp(aug_mat)
        for record in aug_mat:
            u = record[0]
            t = record[1]
            score = record[2]
            assert t in dict_uid
            assert u not in dict_uid[t]
            dict_uid[t].append(u)
            dict_score[t].append(score)

    return dict_uid, dict_score

def separate_mat_log(mat, augment=False):
    if augment:
        print 'start augmenting user profiles...'
        aug_mat = augment_mat(mat)
    mat = normalize_exp_log(mat)
    
    dict_uid = dict([])
    dict_score = dict([])
    for record in mat:
        u = record[0]
        t = record[1]
        score = record[2]
        if t in dict_uid:
            dict_uid[t].append(u)
            dict_score[t].append(score)
            # exp_u_dict[t].add(u)
        else:
            dict_uid[t] = [u]
            dict_score[t] = [score]
    
    if augment:
        #aug_mat = normalize_exp(aug_mat)
        for record in aug_mat:
            u = record[0]
            t = record[1]
            score = record[2]
            assert t in dict_uid
            assert u not in dict_uid[t]
            dict_uid[t].append(u)
            dict_score[t].append(score)

    return dict_uid, dict_score


def augment_mat(mat):
    aug_mat = []
    sf = graphlab.SFrame({'tag_id':[em[1] for em in mat],'answerer_id':[em[0] for em in mat],'expertise': [em[2] for em in mat]})
    recs = graphlab.factorization_recommender.create(sf, target='expertise', num_factors=10, max_iterations=100, side_data_factorization=False, user_id='tag_id', item_id='answerer_id')
    rec_list = recs.recommend(k=40)
    for rl in rec_list:
        aug_mat.append((rl['answerer_id'], rl['tag_id'], rl['score']))
    return aug_mat

def answerer_rec(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score, alpha = 0.5):
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
                if u in qrec:
                    qrec[u] += (1-alpha)*score
                else:
                    qrec[u] = (1-alpha)*score
                if u in act_dict[qt]:
                    qrec[u] += alpha*act_dict[qt][u]
                else:
                    pass

        final_scores = [[u,qrec[u]] for u in qrec]
        final_scores = sorted(final_scores, key=lambda final_scores : final_scores[1], reverse=True)
        #final_scores = final_scores[0:len(qinfo[2])]
        predicted.append([qid,final_scores])
        
    return predicted

def answerer_rec_using_trained_model(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score, model, alpha_default = 0.5):
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
                alpha = alpha_default
                if qt in model:
                    alpha = model[qt]
                if u in qrec:
                    qrec[u] += (1-alpha)*score
                else:
                    qrec[u] = (1-alpha)*score
                if u in act_dict[qt]:
                    qrec[u] += alpha*act_dict[qt][u]
                else:
                    pass

        final_scores = [[u,qrec[u]] for u in qrec]
        final_scores = sorted(final_scores, key=lambda final_scores : final_scores[1], reverse=True)
        #final_scores = final_scores[0:len(qinfo[2])]
        predicted.append([qid,final_scores])
        
    return predicted

def answerer_rec_using_trained_model_L2R(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score, model, alpha_default = 0.5):
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
                if qt in model:
                    alpha1 = model[qt][0][0]
                    alpha2 = model[qt][0][1]
                if u in qrec:
                    qrec[u] += alpha2*score
                else:
                    qrec[u] = alpha2*score
                if u in act_dict[qt]:
                    qrec[u] += alpha1*act_dict[qt][u]
                else:
                    pass

        final_scores = [[u,qrec[u]] for u in qrec]
        final_scores = sorted(final_scores, key=lambda final_scores : final_scores[1], reverse=True)
        #final_scores = final_scores[0:len(qinfo[2])]
        predicted.append([qid,final_scores])
        
    return predicted


def answerer_rec_base(test_data, t_set, exp_dict_uid, exp_dict_score):
    
    k = 0
    predicted = []
    for qinfo in test_data:
        k += 1
        if k%1000 == 0:
            print '...'+str(k)+'th question predicted.'
        qid = qinfo[0]
        qtags = qinfo[1]
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
                if u in qrec:
                    qrec[u] += score
                else:
                    qrec[u] = score

        final_scores = [[u,qrec[u]] for u in qrec]
        final_scores = sorted(final_scores, key=lambda final_scores : final_scores[1], reverse=True)
        #final_scores = final_scores[0:len(qinfo[2])]
        predicted.append([qid,final_scores])
        
    return predicted

def answerer_rec_base_stateoftheart(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score, alpha = 0.5):
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
                this_tag_perf = 0
                
                this_tag_perf = (1-alpha)*score
                this_tag_perf *= alpha*act_dict[qt][u]
                qrec[u] += this_tag_perf

        final_scores = [[u,qrec[u]] for u in qrec]
        final_scores = sorted(final_scores, key=lambda final_scores : final_scores[1], reverse=True)
        #final_scores = final_scores[0:len(qinfo[2])]
        predicted.append([qid,final_scores])
        
    return predicted

def answerer_rec_base_old(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score, alpha = 0.5):
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
                if u in qrec:
                    qrec[u] += (1-alpha)*score
                else:
                    qrec[u] = (1-alpha)*score
                if u in act_dict[qt]:
                    qrec[u] += alpha*act_dict[qt][u]
                else:
                    pass

        final_scores = [[u,qrec[u]] for u in qrec]
        final_scores = sorted(final_scores, key=lambda final_scores : final_scores[1], reverse=True)
        #final_scores = final_scores[0:len(qinfo[2])]
        predicted.append([qid,final_scores])
        
    return predicted



def answerer_rec_train(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score):#, part):
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
    
    alpha_options = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    optimal_alphas = dict([])
    optimal_alphas_rank = dict([])
    each_alpha_perform = dict([])
    
    t_set = [t for t in tag_dict]
    #t_set = t_set[4750*(part-1):4750*part-1]
    
    print 'total number of tags: '+str(len(tag_dict))
    k = 0
    for t in t_set:
        #t = 'java'#this is for debug
        k += 1
        if k%100 == 0:
            print '...'+str(k)+'th tag trained.'
        alpha_performs = []     
        alpha_performs_rank = []  
        #sys.stdout.write(str(t)+',')
        alpha_performs, alpha_performs_rank, qlen = get_alpha_per_tag(t, tag_dict[t], exp_dict_uid, exp_dict_score, act_dict, alpha_options)
        #print alpha_performs
        if len(alpha_performs) == 0:
            #sys.stdout.write('\n')
            continue
        #sys.stdout.write(str(qlen)+',')
        #sys.stdout.write(",".join(str(x) for x in alpha_performs))
        #sys.stdout.write('\n')
        opt_idx = get_opt_idx(alpha_performs)
        opt_idx_rank = get_opt_idx(alpha_performs_rank)
        optimal_alphas[t] = alpha_options[opt_idx]
        optimal_alphas_rank[t] = alpha_options[opt_idx_rank]
        each_alpha_perform[t] = (qlen, alpha_performs)
    return optimal_alphas, optimal_alphas_rank, each_alpha_perform

def get_opt_idx(list):
    base = 5
    m = max(list)
    for i in xrange(6):
        if i==0 and list[base]==m:
            return base
        left = list[base-i]
        right = list[base+i]
        if left!=right and left==m:
            return base-i
        if left!=right and right==m:
            return base+i
        if left==right and left==m:
            return base

def append_predict(qid, qrec, predicted):
    final_scores = [[u, qrec[u]] for u in qrec]
    final_scores = sorted(final_scores, key=lambda final_scores : final_scores[1], reverse=True)
    predicted.append([qid, final_scores])
    del qrec
    del final_scores
    return predicted

def get_alpha_per_tag(t, test_data, exp_dict_uid, exp_dict_score, act_dict, alpha_options):
    alpha_performs = []
    alpha_performs_rank = []
    gt_test = test_data
    predicted = []
    for alpha in alpha_options:
        #print alpha
        if len(gt_test) == 0 or t not in exp_dict_uid:
            break
        predicted = []
        for qinfo in test_data:
            qid = qinfo[0]
            qtags = qinfo[1]
            qusers = [qu[0] for qu in qinfo[2]]
            qrec = dict([])
            us = exp_dict_uid[t]
            scores = exp_dict_score[t]
            for i in xrange(len(us)):
                u = us[i]
                score = scores[i]
                if u not in qusers:
                    continue
                if u in qrec:
                    qrec[u] += (1 - alpha) * score
                else:
                    qrec[u] = (1 - alpha) * score
                # if u in act_dict[qt]:
                qrec[u] += alpha * act_dict[t][u]
            del us
            del scores
            #print 'before final score'
            predicted = append_predict(qid, qrec, predicted)
            #print 'after final score'
        evl = evaluate_train(gt_test, predicted)
        if np.isnan(evl):
            break
        #print '...alpha: ' + str(alpha) + '; eval: ' + str(evl) + '; #qst: ' + str(len(predicted))
        alpha_performs.append(evl)
        alpha_performs_rank.append(evaluate_train(gt_test, predicted, 'rank'))
        
    return alpha_performs, alpha_performs_rank, len(predicted)

def answerer_rec_train_L2R(test_data, t_set, exp_dict_uid, exp_dict_score, act_dict_uid, act_dict_score):#, part):
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
        alphas_t = get_coefficient_L2R_per_tag(t, tag_dict[t], exp_dict_uid, exp_dict_score, act_dict)
        if len(alphas_t) < 2:
            continue
        alphas[t] = alphas_t
    return alphas

def get_coefficient_L2R_per_tag(t, test_data, exp_dict_uid, exp_dict_score, act_dict):
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
            this_X = [act_dict[t][u], score]
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
    
    alphas_t = list(rank_svm.coef_)
    return alphas_t

def get_score_in_gt(u,gt):
    for item in gt:
        if item[0]==u:
            return item[1]