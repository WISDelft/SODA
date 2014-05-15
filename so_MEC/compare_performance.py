import string
import psycopg2
import sys
import numpy
import pickle
import math
import numpy as np
import os
from util_mec import *

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()

def get_motiv_example(experts, answers_att):
    fu = open("userranklists_c#.pik")
    userranklists = pickle.load(fu)
    nrans = dict([])
    nrans = dict([])
    for urank in userranklists:
        for i in range(len(urank)):
            this_u = urank[i]
            u = this_u[0]
            if nrans.has_key(u):
                nrans[u] += 1
            else:
                nrans[u] = 1
    print nrans[145907]
    print nrans[254190]
    print nrans[33213]
    print nrans[153865]
    sys.exit(1)
    qids = dict([])
    userset = dict([])
    nrqst = dict([])
    for a in answers_att:
        qid = a[2]
        uid = a[0]
        qids[qid] = 0
        userset[uid] = 0
        nrqst[uid] = 0
    index = 0
    userranklists = []
    for qid in qids:
        index += 1
        if index%1000 == 0:
            print "the "+str(index)+"th question"
        urank = []
        for a in answers_att:
            if qid == a[2]:
                uscore = a[3]
                uid = a[0]
                urank.append([uid, uscore])
        if len(urank) < 1:
            continue
        urank = sorted(urank, key = lambda urank : urank[1], reverse=True)
        if urank[0][0] not in experts or nrans[urank[0][0]]>=10:
            continue
        a = 0
        b = 0
        for u in urank:
            if nrans[u[0]]>10:
                a += 1
            if u[0] in experts:
                b += 1
        print str(qid)+', len:'+str(len(urank))+', owls: '+str(b)+', sparrows: '+str(a)
        qids[qid] = len(urank)
        userranklists.append(urank)
        
def get_motiv_scatter(experts, answerer_scores, tag):
    userranklists = loadfile("userranklists_"+tag)
    sparrows = loadfile("sparrows_"+tag)
    
    debate = dict([])
    debate_eu = dict([])
    debate_s = dict([])
    rank = dict([])
    rank_eu = dict([])
    rank_s = dict([])
    for urank in userranklists:
        for i in range(len(urank)):
            this_u = urank[i]
            u = this_u[0]
            if debate.has_key(u):
                debate[u].append(len(urank))
                rank[u].append(i+1)
            else:
                debate[u]=[len(urank)]
                rank[u] = [i+1]
            if u in experts:
                if debate_eu.has_key(u):
                    debate_eu[u].append(len(urank))
                    rank_eu[u].append(i+1)
                else:
                    debate_eu[u]=[len(urank)]
                    rank_eu[u] = [i+1]
            if u in sparrows:
                if debate_s.has_key(u):
                    debate_s[u].append(len(urank))
                    rank_s[u].append(i+1)
                else:
                    debate_s[u]=[len(urank)]
                    rank_s[u] = [i+1]

    debate_medium = []
    debate_medium_eu = []
    debate_medium_s = []
    for u in answerer_scores:
        debate_medium.append( numpy.median(debate[u]))
        if u in experts>=1:
            debate_medium_eu.append( numpy.median(debate_eu[u]))
        if u in sparrows:
            debate_medium_s.append(numpy.median(debate_s[u]))
    qd = [] # question debatableness
    aq = [] # answer quality
    qd_s = []
    aq_s = []
    qd_o = []
    aq_o = []
    qd_all = []
    aq_all = []

    ideals = []
    exs = []
    for u in debate:
        leni = []
        rri = []
        for i in range(min(40,max(debate[u]))):
            r = [] #the rank of u in answering questions of debatableness i
            for j in range(len(debate[u])):
                temp_debate_list = debate[u]
                if temp_debate_list[j]==i+1:
                    temp_rank_list = rank[u]
                    r.append(float(temp_rank_list[j]-1)/(i+1))
                #print r
            if len(r) != 0:
                #leni.append(i+0.5 * np.random.randn())
                #rri.append(numpy.mean(r)+0.05 * np.random.randn())
                leni.append(i)
                rri.append(numpy.mean(r))
        a = numpy.mean(leni)
        if math.isnan(a):
            continue
        #qd_all.append(int(round(a)))
        qd_all.append(a)
        aq_all.append(1-numpy.mean(rri))
        if u in sparrows:
            qd_s.append(numpy.mean(leni)+0.5 * np.random.randn())#
            aq_s.append(1-numpy.mean(rri)+0.04 * np.random.randn())#
        else:
            qd.append(numpy.mean(leni)+0.5 * np.random.randn())
            aq.append(1-numpy.mean(rri)+0.04 * np.random.randn())
        if u in experts:
            qd_o.append(numpy.mean(leni)+0.5 * np.random.randn())
            aq_o.append(1-numpy.mean(rri)+0.04 * np.random.randn())
            exs.append(u)
        if numpy.mean(leni)>3.0 and numpy.mean(rri)<0.37:
            ideals.append(u)

    output = open("data/illustration.csv", 'w')

    for k in range(len(qd_s)):
        #print k
        if qd_s[k]<1:
            continue
        if aq_s[k]<0 or aq_s[k]>1:
            continue
        output.write("sparrow, "+str(qd_s[k])+", "+str(aq_s[k])+"\n")
    for k in range(len(qd)):
        if qd[k]<1:
            continue
        if aq[k]<0 or aq[k]>1:
            continue
        output.write("others, "+str(qd[k])+", "+str(aq[k])+"\n")
    output.close()
    print len(ideals)

    st1 = 0
    for s in qd_all:
        if s!=None:
            st1 += s

    st1 = numpy.mean(qd_all)
    print 'qlen:'+str(st1)+ ' median: '+str(numpy.median(qd_all))+' std:'+str(numpy.std(qd_all))
    st2 = numpy.mean(aq_all)
    print 'aq:'+str(st2)+ ' median: '+str(numpy.median(aq_all))+' std:'+str(numpy.std(aq_all))


    print float(len(list(set(exs) & set(ideals))))/len(ideals)
    print float(len(list(set(sparrows) & set(ideals))))/len(ideals)
    
    
def get_nrqst(answerer_scores,tag):
    nrqst = dict([])
    for u in answerer_scores:
        cur.execute("select count(*) from questions where owneruserid="+str(u))
        result = cur.fetchone()
        if result!=None:
            nrqst[u] = result[0]
        else:
            nrqst[u] = 0
    dumpfile(nrqst, "nrqst_"+tag)
    return nrqst
    
def get_performance(experts, answerer_scores, tag):
    userranklists = loadfile("userranklists_"+tag)
    nrans = loadfile('nrans_'+tag)
    if os.path.exists("temp_files/nrans_"+tag+"pik"):
        nrqst = loadfile("nrqst_"+tag)
    else:
        nrqst = get_nrqst(answerer_scores,tag)
        
    ua = loadfile("sparrows_"+tag)
    exs = experts
    
    output = open("data/nr_ans_qst.csv", 'w')

    sa = []
    sb = []
    oa = []
    ob = []
    alla = []
    allb = []
    for u in ua:
        sa.append(nrans[u])
        sb.append(nrqst[u])
        output.write("sparrow, "+str(nrans[u])+", "+str(nrqst[u])+"\n")
    for u in exs:
        output.write("owl, "+str(nrans[u])+", "+str(nrqst[u])+"\n")
    for u in answerer_scores:
        output.write("overall, "+str(nrans[u])+", "+str(nrqst[u])+"\n")
    output.close()
    ################################################################################################################
    debate = dict([])
    debate_eu = dict([])
    debate_s = dict([])
    rank = dict([])
    rank_eu = dict([])
    rank_s = dict([])
    for urank in userranklists:
        for i in range(len(urank)):
            this_u = urank[i]
            u = this_u[0]
            if debate.has_key(u):
                debate[u].append(len(urank))
                rank[u].append(i+1)
            else:
                debate[u]=[len(urank)]
                rank[u] = [i+1]
            if u in exs:
                if debate_eu.has_key(u):
                    debate_eu[u].append(len(urank))
                    rank_eu[u].append(i+1)
                else:
                    debate_eu[u]=[len(urank)]
                    rank_eu[u] = [i+1]
            if u in ua:
                if debate_s.has_key(u):
                    debate_s[u].append(len(urank))
                    rank_s[u].append(i+1)
                else:
                    debate_s[u]=[len(urank)]
                    rank_s[u] = [i+1]

    debate_medium = []
    debate_medium_eu = []
    debate_medium_s = []
    for u in answerer_scores:
        debate_medium.append( numpy.median(debate[u]))
        if u in exs:
            debate_medium_eu.append( numpy.median(debate_eu[u]))
        if u in ua:
            debate_medium_s.append(numpy.median(debate_s[u]))
    output = open("data/qst_deb.csv", 'w')

    for u in ua:
        output.write("sparrow, "+str(round(numpy.median(debate_s[u]), 2))+"\n")
    for u in exs:
        output.write("owl, "+str(round(numpy.median(debate_eu[u]), 2))+"\n")
    for u in answerer_scores:
        output.write("overall, "+str(round(numpy.median(debate[u]), 2))+"\n")
    output.close()
    ################################################################################################################
    qlen = []
    for urank in userranklists:
        qlen.append(len(urank))
    #print "qlen: "+str(numpy.mean(qlen))+" +/- "+str(numpy.std(qlen))

    showup = 0
    showup_nmrr = 0
    showup_ans = 0
    com = []
    com_nmrr = []
    com_ans = []
    com_or = []
    for urank in userranklists:
        pos_nmrr = []
        pos_repu = []
        u_nmrr = []
        u_repu = []
        
        for i in range(len(urank)):
            this_u = urank[i]
            u = this_u[0]
            if u in exs:
                pos_nmrr.append(i)
                u_nmrr.append(u)
            if u in ua:
                pos_repu.append(i)
                u_repu.append(u)
        if len(pos_nmrr)!=0:
            showup_nmrr += 1
            com_nmrr.append(float(numpy.mean(pos_nmrr))/len(urank))
        if len(pos_repu)!=0:
            showup_ans += 1
            com_ans.append(float(numpy.mean(pos_repu))/len(urank))
        if len(pos_nmrr)!=0 or len(pos_repu)!=0:
            com_or.append(float(numpy.mean(list(set(exs) | set(ua))))/len(urank))
        if len(pos_nmrr)!=0 and len(pos_repu)!=0:
            '''if pos_nmrr[0]<pos_repu[0]:
                print '---'
                print urank
                print pos_nmrr
                print pos_repu'''
            showup += 1
            c = [numpy.mean(pos_nmrr), numpy.mean(pos_repu), len(urank)]
            #print c
            com.append(c)
    print "answered by owl: "+str(showup_nmrr)+" answered by sparrow: "+str(showup_ans)
    print showup_nmrr+showup_ans-showup
    print "---"
    print numpy.mean(com_nmrr)
    print numpy.mean(com_ans)
    print numpy.mean(com_or)
                
                
    '''if pos_nmrr!=-1 and pos_repu!=-1:
            output.write(str(pos_nmrr)+","+str(pos_repu)+","+str(u_nmrr)+","+str(u_repu)+","+str(len(urank))+"\n")
        if (pos_nmrr!=-1 and pos_repu==-1) or (pos_nmrr!=-1 and pos_nmrr!=pos_repu):
            output2.write(str(pos_nmrr)+","+str(u_nmrr)+","+str(len(urank))+"\n")
        if (pos_nmrr==-1 and pos_repu!=-1) or (pos_repu!=-1 and pos_nmrr!=pos_repu):
            output3.write(str(pos_repu)+","+str(u_repu)+","+str(len(urank))+"\n")'''
    '''for c in com:
        print com'''

    f = open("com.pik", 'w')
    dumpfile(com, "com_"+tag)
    print 'com done!'
    comlook(tag)
#savehist(userset)

def comlook(tag):
    com = loadfile("com_"+tag)
    print len(com)
    index = []
    nmrrs_mean = []
    nmrrs_std = []
    repus_mean = []
    repus_std = []
    for i in range(40):
        index.append(i+1)
        nmrrs = []
        repus = []
        for j in range(len(com)):
            c = com[j]
            if c[2] == i+1:
                nmrrs.append(float(c[0])/c[2])
                repus.append(float(c[1])/c[2])
        
        print str(numpy.mean(nmrrs))+"  "+str(numpy.std(nmrrs))
        print str(numpy.mean(repus))+"  "+str(numpy.std(repus))
        
        nmrrs_mean.append(1-numpy.mean(nmrrs))
        nmrrs_std.append(numpy.std(nmrrs))
        repus_mean.append(1-numpy.mean(repus))
        repus_std.append(numpy.std(repus))
        print len(nmrrs)
        print i+1
        print '------'
    output = open("datas/ans_quality.csv", 'w')
    for i in index:
        output.write(str(i-1)+", "+str(nmrrs_mean[i])+", "+str(repus_mean[i])+"\n")
    output.close()

