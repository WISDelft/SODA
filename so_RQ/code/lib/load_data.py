import string
import psycopg2
import sys
import pickle
import os
import numpy as np
import copy

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='wistudelft')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()

'''
load training data
'''
def load_train_data(option='na'):
    at_NoAnswer = dict([])
    at_NoVotes = dict([])
    at_MEC_quantities = dict([])
    at_NoComment_q = dict([])
    at_NoCommentVotes_q = dict([])
    at_NoComment_a = dict([])
    at_NoCommentVotes_a = dict([])
    
    ## export the data from database
    if option=='train':
        cur.execute("select id from questions where TO_CHAR (creationdate,'YYYY-MM-DD') < '2012-07-01'")
    else:
        cur.execute("select id from questions where TO_CHAR (creationdate,'YYYY-MM-DD') < '2013-01-01'")
    results = cur.fetchall()
    '''for rs in results:
        qid = rs[0]
        if qid == None:
            continue
        
        ##### activeness, expertise #####
        if option=='train':
            cur.execute("select aowneruserid, ascore, qtags from sim_qa where qid = "+str(qid)+" and TO_CHAR (acreationdate,'YYYY-MM-DD') < '2012-07-01'")
        else:
            cur.execute("select aowneruserid, ascore, qtags from sim_qa where qid = "+str(qid)+" and TO_CHAR (acreationdate,'YYYY-MM-DD') < '2013-01-01'")
        
        aresults = cur.fetchall()
        if aresults == None or len(aresults) ==0:
            continue
        
        qtags = []
        try:
            for t in aresults[0][2].split('|'): 
                qtags.append(t)
        except:
            print aresults
        if len(qtags) == 0:
            continue
        
        mec_quantities = []
        for ar in aresults:
            ascore = ar[1]
            aowneruserid = ar[0]
            mec_quantities.append([aowneruserid, ascore])
            for t in qtags:
                if (aowneruserid, t) in at_NoAnswer:
                    at_NoAnswer[(aowneruserid, t)] += 1
                    at_NoVotes[(aowneruserid, t)] += ascore
                else:
                    at_NoAnswer[(aowneruserid, t)] = 1
                    at_NoVotes[(aowneruserid, t)] = ascore
        mec_quantities = sorted(mec_quantities, key = lambda mec_quantities : mec_quantities[1], reverse=True)
        rank = 1
        for mq in mec_quantities:
            aowneruserid = mq[0]
            for t in qtags:
                if (aowneruserid, t) in at_MEC_quantities:
                    at_MEC_quantities[(aowneruserid, t)].append([rank,len(mec_quantities)])
                else:
                    at_MEC_quantities[(aowneruserid, t)] = [[rank,len(mec_quantities)]]
            rank += 1'''
    for rs in results:
        qid = rs[0]
        if qid == None:
            continue
        
        ##### participation #####
        if option=='train':
            cur.execute("select cuserid, cscore, qtags from sim_cqa where qid = "+str(qid)+" and cuserid <> qowneruserid and cuserid<>aowneruserid and TO_CHAR (ccreationdate,'YYYY-MM-DD') < '2012-07-01'")
        else:
            cur.execute("select cuserid, cscore, qtags from sim_cqa where qid = "+str(qid)+" and cuserid <> qowneruserid and cuserid<>aowneruserid and TO_CHAR (ccreationdate,'YYYY-MM-DD') < '2013-01-01'")
        caresults = cur.fetchall()
        if caresults == None or len(caresults) ==0:
            continue
        
        qtags = []
        try:
            for t in caresults[0][2].split('|'): 
                qtags.append(t)
        except:
            print caresults
        if len(qtags) == 0:
            continue
        for car in caresults:
            cascore = car[1]
            if cascore == None:
                cascore = 0
            causerid = car[0]
            for t in qtags:
                if (causerid, t) in at_NoComment_a:
                    at_NoComment_a[(causerid, t)] += 1
                    at_NoCommentVotes_a[(causerid, t)] += cascore
                else:
                    at_NoComment_a[(causerid, t)] = 1
                    at_NoCommentVotes_a[(causerid, t)] = cascore
    for rs in results:
        qid = rs[0]
        if qid == None:
            continue
        
        if option=='train':
            cur.execute("select cuserid, cscore, qtags from sim_cq where qid = "+str(qid)+" and cuserid<>qowneruserid and TO_CHAR (ccreationdate,'YYYY-MM-DD') < '2012-07-01'")
        else:
            cur.execute("select cuserid, cscore, qtags from sim_cq where qid = "+str(qid)+" and cuserid<>qowneruserid and TO_CHAR (ccreationdate,'YYYY-MM-DD') < '2013-01-01'")
        cqresults = cur.fetchall()
        if cqresults == None or len(cqresults) ==0:
            continue
        
        qtags = []
        try:
            for t in cqresults[0][2].split('|'): 
                qtags.append(t)
        except:
            print cqresults
        if len(qtags) == 0:
            continue
        for cqr in cqresults:
            cqscore = cqr[1]
            if cqscore == None:
                cqscore = 0
            cquserid = cqr[0]
            for t in qtags:
                if (cquserid, t) in at_NoComment_q:
                    at_NoComment_q[(cquserid, t)] += 1
                    at_NoCommentVotes_q[(cquserid, t)] += cqscore
                else:
                    at_NoComment_q[(cquserid, t)] = 1
                    at_NoCommentVotes_q[(cquserid, t)] = cqscore
        
    ## add to data structure
    mec = []
    mec_log = []
    mec_naive = []
    zscore = []
    exp_data_repu = []
    exp_data_repu_norm = []
    act_data = []
    '''for ut in at_NoAnswer:
        if ut[0] == None:
            continue
        exp_data_repu.append((ut[0],ut[1], at_NoVotes[(ut[0],ut[1])]))
        exp_data_repu_norm.append((ut[0],ut[1], float(at_NoVotes[(ut[0],ut[1])])/at_NoAnswer[(ut[0],ut[1])]))
        act_data.append((ut[0],ut[1], at_NoAnswer[(ut[0],ut[1])]))
        
    for ut in at_MEC_quantities:
        if ut[0] == None:
            continue
        #if ut[0]==616639 and ut[1]=='hacking':
            #print at_MEC_quantities[(ut[0],ut[1])]
        mec_mec_quantities = at_MEC_quantities[(ut[0],ut[1])]
        iranks_naive = [float(1)/r[0] for r in mec_mec_quantities]
        iranks = [float(r[1])/r[0] for r in mec_mec_quantities]
        iranks_log = [np.log2(r[1])/r[0] for r in mec_mec_quantities]
        mec_naive.append((ut[0],ut[1],  np.mean(iranks_naive)))
        mec.append((ut[0],ut[1],  np.mean(iranks)))
        mec_log.append((ut[0],ut[1],  np.mean(iranks_log)))
        #if ut[0]==616639 and ut[1]=='hacking':
            #sys.exit(1)'''

    parti_act_data = []
    parti_exp_data = []
    for ut in at_NoComment_a:
        if ut[0] == None:
            continue
        if ut in at_NoComment_q:
            parti_act_data.append((ut[0],ut[1], at_NoComment_a[(ut[0],ut[1])]+at_NoComment_q[(ut[0],ut[1])]))
            parti_exp_data.append((ut[0],ut[1], at_NoCommentVotes_a[(ut[0],ut[1])]+at_NoCommentVotes_q[(ut[0],ut[1])]))
        else:
            parti_act_data.append((ut[0],ut[1], at_NoComment_a[(ut[0],ut[1])]))
            parti_exp_data.append((ut[0],ut[1], at_NoCommentVotes_a[(ut[0],ut[1])]))
    for ut in at_NoComment_q:
        if ut[0] == None:
            continue
        if ut in at_NoComment_a:
            continue
        parti_act_data.append((ut[0],ut[1], at_NoComment_q[(ut[0],ut[1])]))
        parti_exp_data.append((ut[0],ut[1], at_NoCommentVotes_q[(ut[0],ut[1])]))
    
    
    return exp_data_repu, exp_data_repu_norm, act_data, mec_naive, mec, mec_log, parti_act_data, parti_exp_data

def load_train_data_old(expertise_option='novotes'):
    train_data = []
    cur.execute("select * from qr_ut_matrix_train")
    records = cur.fetchall()
    for rc in records:
        if rc[2]==0:
            continue
        if expertise_option=='novotes':
            train_data.append((rc[0], rc[1], rc[3]))
    return train_data

'''
for each question posted after 2013-01-01:
    get the ranked list of answerers
    get the tags
'''
def load_test_data(option='na'):
    qinfos = []
    if option=='train':
        cur.execute("select id from questions where TO_CHAR (creationdate,'YYYY-MM-DD') < '2013-01-01' and TO_CHAR (creationdate,'YYYY-MM-DD') >= '2012-07-01'")
    else:
        cur.execute("select id from questions where TO_CHAR (creationdate,'YYYY-MM-DD') >= '2013-01-01'")
    results = cur.fetchall()
    for rs in results:
        qid = rs[0]
        if qid == None:
            continue
        
        cur.execute("select aowneruserid, ascore, qtags from sim_qa where qid = "+str(qid))
        aresults = cur.fetchall()
        if aresults == None or len(aresults) ==0:
            continue
        
        qtags = []
        try:
            for t in aresults[0][2].split('|'): 
                qtags.append(t)
        except:
            print aresults
        if len(qtags) == 0:
            continue
        
        qrank = []
        for ar in aresults:
            qrank.append([ar[0], ar[1]])
        qrank = sorted(qrank, key = lambda qrank : qrank[1], reverse=True)
            
        qinfo = []
        qinfo.append(qid) 
        qinfo.append(qtags)
        qinfo.append(qrank)
        qinfos.append(qinfo)
        
    return qinfos
    
    
def check_date_(qid, date):
    cur.execute("select qid from sim_qa where qid="+str(qid)+" and TO_CHAR(acreationdate,'YYYY-MM-DD') <'" + date +"'")
    result = cur.fetchone()
    if result == None:
        return False
    else:
        return True
    
def data_filter_single(exp_data, act_data, test_data):
    train_u = set([r[0] for r in exp_data])
    print '#answerers in Training data: '+str(len(train_u))
    test_u = set([u[0] for qi in test_data for u in qi[2]])
    print '#answerers in Test data: '+str(len(test_u))
    intersect_u = train_u.intersection(test_u)
    print '#answerers in both sets: '+str(len(intersect_u))
    
    exp_data_new = []
    act_data_new = []
    test_data_new = []

    for r in exp_data:
        if r[0] in intersect_u:
            exp_data_new.append(r)
    del exp_data
    for r in act_data:
        if r[0] in intersect_u:
            act_data_new.append(r)
    del act_data
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
    test_data = []
    for r in exp_data_new:
        if r[0] in intersect_u:
            exp_data.append(r)
    del exp_data_new
    for r in act_data_new:
        if r[0] in intersect_u:
            act_data.append(r)
    del act_data_new
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
    return exp_data,act_data, test_data  
    
def data_filter(exp_data_all, exp_data, act_data, test_data, exp_data_mec_naive, exp_data_mec, exp_data_mec_log):
    train_u = set([r[0] for r in exp_data])
    print '#answerers in Training data: '+str(len(train_u))
    test_u = set([u[0] for qi in test_data for u in qi[2]])
    print '#answerers in Test data: '+str(len(test_u))
    intersect_u = train_u.intersection(test_u)
    print '#answerers in both sets: '+str(len(intersect_u))
    
    exp_data_mec_new = []
    exp_data_mec_naive_new = []
    exp_data_mec_log_new = []
    exp_data_new = []
    exp_data_all_new = []
    act_data_new = []
    test_data_new = []
    for r in exp_data_mec:
        if r[0] in intersect_u:
            exp_data_mec_new.append(r)
    del exp_data_mec
    for r in exp_data_mec_naive:
        if r[0] in intersect_u:
            exp_data_mec_naive_new.append(r)
    del exp_data_mec_naive
    for r in exp_data_mec_log:
        if r[0] in intersect_u:
            exp_data_mec_log_new.append(r)
    del exp_data_mec_log
    for r in exp_data:
        if r[0] in intersect_u:
            exp_data_new.append(r)
    del exp_data
    for r in exp_data_all:
        if r[0] in intersect_u:
            exp_data_all_new.append(r)
    del exp_data_all
    for r in act_data:
        if r[0] in intersect_u:
            act_data_new.append(r)
    del act_data
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
    
    exp_data_mec = []
    exp_data_mec_naive = []
    exp_data_mec_log = []
    exp_data = []
    exp_data_all = []
    act_data = []
    test_data = []
    for r in exp_data_mec_new:
        if r[0] in intersect_u:
            exp_data_mec.append(r)
    del exp_data_mec_new
    for r in exp_data_mec_naive_new:
        if r[0] in intersect_u:
            exp_data_mec_naive.append(r)
    del exp_data_mec_naive_new
    for r in exp_data_mec_log_new:
        if r[0] in intersect_u:
            exp_data_mec_log.append(r)
    del exp_data_mec_log_new
    for r in exp_data_new:
        if r[0] in intersect_u:
            exp_data.append(r)
    del exp_data_new
    for r in exp_data_all_new:
        if r[0] in intersect_u:
            exp_data_all.append(r)
    del exp_data_all_new
    for r in act_data_new:
        if r[0] in intersect_u:
            act_data.append(r)
    del act_data_new
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
    return exp_data_all, exp_data,act_data, test_data, exp_data_mec_naive, exp_data_mec, exp_data_mec_log


if __name__ == '__main__':
    exp_data=[(1,'c#',10),(2,'c#',5),(3,'java',1),(5,'java',1),(6,'java',2)]
    act_data=[(1,'c#',1),(2,'c#',2),(3,'java',1),(5,'java',1)]
    test_data = [[1,'c#',[[6,2],[4,1],[9,1],[10,1]]],[2,'c#',[[1,2],[4,1],[3,1]]]]
    exp_data,act_data,test_data = data_filter(exp_data,act_data,test_data)
    print exp_data
    print test_data