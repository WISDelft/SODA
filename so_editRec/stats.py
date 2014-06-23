import json
import sys
import psycopg2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import scipy.stats
from util import *

con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()
print "Connecting to DB: " + str(ver)

def plothist():
    a = []
    for line in open('images/days_inf'):
        val = int(float(line[0:-1]))
        if val<0:
            continue
        #if val>100:
            #continue
        a.append(val)
    savehist(a, 'days_inf')
    '''pl.hist(a,bins=50)
    pl.show()'''

def savehist(vals, fname):
    pl.hist(vals,bins=50)
    pl.savefig('images/'+fname+'.pdf')
    f = open('images/'+fname, 'w')
    for v in vals:
        f.write(str(v)+'\n')
    f.close()
    
def savescatter(dic, fname):
    a = []
    b = []
    f = open('images/'+fname,'w')
    for d in dic:
        f.write(str(d)+','+str(dic[d])+'\n')
        a.append(d)
        b.append(dic[d])
    f.close()
    print scipy.stats.spearmanr(a, b)
    pl.scatter(a,b)
    pl.savefig('images/'+fname+'.pdf')
    
def topic_inf(pos):
    m = 0
    k = 0
    neg = []
    auth=loadfile_flat('sorted_nrans')
    eqids = sorted(loadfile('ed_qst_ids'))
    for t in auth:
        k+=1
        if not bi_contains(eqids, t[0]):#t[0] not in eqids: #and check_date(t[0], '2013-01-01'):
            #print t
            neg.append(t[0])
            m+=1
            if m==len(pos):
                break
            
    nrqst_tag_pos = dict([])
    i = 0
    for qid in pos:
        i += 1
        if i%1000 == 0:
            print i
        cur.execute("select qtags from sim_qa where qid="+str(qid))
        tags = cur.fetchall()[0][0]
        if tags == None:
            continue
        tags = tags.split('|')
        for t in tags:
            if t in nrqst_tag_pos:
                nrqst_tag_pos[t] += 1
            else:
                nrqst_tag_pos[t] = 1
    te = []
    i = 0
    nrqst_tag_neg = dict([])
    for qid in neg:
        i += 1
        if i%1000 == 0:
            print i
        cur.execute("select qtags from sim_qa where qid="+str(qid))
        tags = cur.fetchall()[0][0]
        if tags == None:
            continue
        tags = tags.split('|')
        for t in tags:
            if t in nrqst_tag_neg:
                nrqst_tag_neg[t] += 1
            else:
                nrqst_tag_neg[t] = 1
    nrqst_tag = loadfile_flat('nrqst_tag')
    for t in nrqst_tag:
        if nrqst_tag[t]<10000:
            continue
        if t in nrqst_tag_pos and t in nrqst_tag_neg:
            te.append([t, float(nrqst_tag_pos[t])/nrqst_tag_neg[t]])
    tags = ['symfony2', 'asp.net-mvc-4', 'android-layout', 'jsf', 'asp.net-mvc-3', 'database-design', 'vim', 'testing', 'design', 'svn']
    
    te = sorted(te, key=lambda te : te[1],reverse=True)
    print te
    print te[0:5]
    print te[-5:-1]
    for t in tags:
        print t
        print nrqst_tag_pos[t]+nrqst_tag_neg[t]

def user_inf(pos):
    return 0
    
def klg_inf(conf):
    tag = loadfile_flat('nrqst_tag')
    tag = dict2list(tag)
    tag = sorted(tag, key=lambda tag : tag[1],reverse=True)
    i = 0
    for t in tag:
        if i==20:
            return
        i += 1
        print t[0]
        #pos = loadfile_all('java_good_edit_ans_', [1,2,3,4])
        pos = []
        for q in conf:
            if is_tag_in(t[0], q):
                pos.append(q)
        uids = dict([]) #{u1:[q1, q2, q3], u2:[q4, q5, q6]}
        this_i = 0
        for qid in pos:
            this_i += 1
            if this_i%1000==0:
                print '..[klg inf] processing the '+str(this_i)+"th question"
            cur.execute("select qowneruserid, qcreationdate from sim_qa where qid="+str(qid))
            uid = cur.fetchone()[0]
            if uid == None:
                continue
            if uid not in uids:
                uids[uid] = [qid]
            else:
                uids[uid].append(qid)
        date = '2012-01-01'
        days_eq = dict([])
        ac_eq = dict([])
    
        for u in uids:
            if check_date(u, date):
                for qid in uids[u]:
                    cur.execute("select EXTRACT(day FROM sim_qa.qcreationdate-users.creationdate) from sim_qa, users where sim_qa.qid="+str(qid)+" and users.id=sim_qa.qowneruserid and users.id="+str(u))
                    result = cur.fetchone()
                    if result==None or len(result)==0:
                        print '[warning] no date for question'
                        continue
                    day = result[0]
                    if day==None:
                        print '[warning] no date for question'
                        continue
                    if day in days_eq:
                        days_eq[day]+=1
                    else:
                        days_eq[day]=1
                        
                    '''cur.execute("select count(distinct S1.aid) from sim_qa as S1, sim_qa as S2 where S1.aowneruserid="+str(uid)+" and S1.acreationdate<S2.qcreationdate and S2.qid="+str(qid))
                    nrac = cur.fetchone()[0]
                    cur.execute("select count(distinct S1.qid) from sim_qa as S1, sim_qa as S2 where S1.qowneruserid="+str(uid)+" and S1.qcreationdate<S2.qcreationdate and S2.qid="+str(qid))
                    nrac += cur.fetchone()[0]
                    if nrac in ac_eq:
                        ac_eq[int(nrac)]+=1
                    else:
                        ac_eq[int(nrac)]=1'''
        #print days_eq
        #print ac_eq
        savescatter(days_eq, 'days_eq'+t[0])
        #savescatter(ac_eq, 'ac_eq'+t[0])
    return 0


def check_date(id, date):
    cur.execute("select id from users where id="+str(id)+" and TO_CHAR(creationdate,'YYYY-MM-DD') <'" + date +"'")
    result = cur.fetchone()
    if result == None:
        return False
    else:
        return True
def temp_inf(pos):
    m = 0
    k = 0
    neg = []
    auth=loadfile_flat('sorted_nrans')
    eqids = sorted(loadfile('ed_qst_ids'))
    for t in auth:
        k+=1
        if not bi_contains(eqids, t[0]):#t[0] not in eqids: #and check_date(t[0], '2013-01-01'):
            #print t
            neg.append(t[0])
            m+=1
            if m==len(pos):
                break
            
    time_ed = dict([])
    i = 0
    for qid in pos:
        i += 1
        if i%1000 == 0:
            print i
        cur.execute("select TO_CHAR(qcreationdate,'YYYY-MM-DD') from sim_qa where qid="+str(qid))
        time = cur.fetchone()[0]
        if time == None:
            continue
        if time in time_ed:
            time_ed[time] += 1
        else:
            time_ed[time] = 1
    time_ned = dict([])
    i = 0
    for qid in neg:
        i += 1
        if i%1000 == 0:
            print i
        cur.execute("select TO_CHAR(qcreationdate,'YYYY-MM-DD') from sim_qa where qid="+str(qid))
        time = cur.fetchone()[0]
        if time == None:
            continue
        if time in time_ned:
            time_ned[time] += 1
        else:
            time_ned[time] = 1
    time_diff = dict([])
    for time in time_ed:
        if time in time_ned:
            time_diff[time] = time_ed[time] - time_ned[time]
        else:
            time_diff[time] = time_ed[time]
    for time in time_ned:
        if time not in time_ed:
            time_diff[time] = -time_ned[time]
    time_diff = dict2list(time_diff)
    time_diff = sorted(time_diff, key=lambda time_diff : time_diff[0])
    f = open('time_inf', 'w')
    for t in time_diff:
        f.write(str(t[0])+','+str(t[1]))
    f.close
    return 0

def beginer_inf():
    typeDict = {'code':2, 'description':3, 'detail':4, 'attempt':5}
    for type in typeDict:
        print type
        qlabels = loadfile_flat('ed_type_qids_extreme_'+type)
        pos = []
        neg = []
        for qid in qlabels:
            cur.execute("select qowneruserid, qcreationdate from sim_qa where qid="+str(qid))
            uid = cur.fetchone()[0]
            if uid == None:
                continue
            cur.execute("select EXTRACT(day FROM sim_qa.qcreationdate-users.creationdate) from sim_qa, users where sim_qa.qid="+str(qid)+" and users.id=sim_qa.qowneruserid and users.id="+str(uid))
            result = cur.fetchone()
            if result==None or len(result)==0:
                print '[warning] no date for question'
                continue
            day = result[0]
            if day==None:
                print '[warning] no date for question'
                continue
            if qlabels[qid] == 1:
                pos.append(day)
            else:
                neg.append(day)
        print scipy.stats.mannwhitneyu(pos, neg)
        f = open('images/begin_inf_'+type, 'w')
        for d in pos:
            f.write('Edited,'+str(d)+'\n')
        for d in neg:
            f.write('Non-edited,'+str(d)+'\n')
        f.close()

if __name__ == '__main__':
    #plothist()
    #pos = loadfile_all('good_edit_ans_', [1,2,3,4,51,52,6,7,8,9,10])
    #pos = loadfile_flat('extreme_set')
    #topic_inf(pos)
    #user_inf(pos)
    #klg_inf(pos)
    #temp_inf(pos)
    beginer_inf()