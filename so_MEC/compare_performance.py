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
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='wistudelft')
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
    output_noAgg = open("data/illustration_noAgg.csv", 'w')
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
                output_noAgg.write("sparrow, "+str(len(urank)+0.5 * np.random.randn())+", "+str(1-float(i)/len(urank)+0.04 * np.random.randn())+"\n")
                if debate_s.has_key(u):
                    debate_s[u].append(len(urank))
                    rank_s[u].append(i+1)
                else:
                    debate_s[u]=[len(urank)]
                    rank_s[u] = [i+1]
            else:
                output_noAgg.write("other, "+str(len(urank)+0.5 * np.random.randn())+", "+str(1-float(i)/len(urank)+0.04 * np.random.randn())+"\n")

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
                leni.append(i+1)
                rri.append(numpy.median(r))
        a = numpy.median(leni)
        if math.isnan(a):
            continue
        #qd_all.append(int(round(a)))
        qd_all.append(a)
        aq_all.append(1-numpy.median(rri))
        if u in sparrows:
            qd_s.append(numpy.median(leni)+0.5 * np.random.randn())#
            aq_s.append(1-numpy.median(rri)+0.04 * np.random.randn())#
        else:
            qd.append(numpy.median(leni)+0.5 * np.random.randn())
            aq.append(1-numpy.median(rri)+0.04 * np.random.randn())
        if u in experts:
            qd_o.append(numpy.median(leni)+0.5 * np.random.randn())
            aq_o.append(1-numpy.median(rri)+0.04 * np.random.randn())
            exs.append(u)
        if numpy.median(leni)>3.0 and numpy.median(rri)<0.37:
            ideals.append(u)

    output_noAgg.close()
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

    st1 = numpy.median(qd_all)
    print 'qlen:'+str(st1)+ ' median: '+str(numpy.median(qd_all))+' std:'+str(numpy.std(qd_all))
    st2 = numpy.median(aq_all)
    print 'aq:'+str(st2)+ ' median: '+str(numpy.median(aq_all))+' std:'+str(numpy.std(aq_all))


    print float(len(list(set(exs) & set(ideals))))/len(ideals)
    print float(len(list(set(sparrows) & set(ideals))))/len(ideals)
    
    
def get_nrqst(answerer_scores,tag):
    nrqst = dict([])
    for u in answerer_scores:
        cur.execute("select count(distinct qid) from sim_qa where qowneruserid="+str(u))
        result = cur.fetchone()
        if result!=None:
            nrqst[u] = result[0]
        else:
            nrqst[u] = 0
    dumpfile(nrqst, "nrqst_"+tag)
    return nrqst
    
def get_performance(experts, answerer_scores, ua, tag, name):
    userranklists = loadfile("userranklists_"+tag)
    nrans = loadfile('nrans_'+tag)
    if os.path.exists("temp_files/nrans_"+tag+".pik"):
        nrqst = loadfile("nrqst_"+tag)
    else:
        nrqst = get_nrqst(answerer_scores,tag)
        
    exs = experts
    
    output = open("data/nr_ans_qst_"+name+".csv", 'w')

    sa = []
    sb = []
    oa = []
    ob = []
    alla = []
    allb = []
    for u in ua:
        sa.append(nrans[u])
        sb.append(nrqst[u])
        output.write(name+","+str(nrans[u])+", "+str(nrqst[u])+"\n")
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
    output = open("data/qst_deb_"+name+".csv", 'w')

    for u in ua:
        output.write(name+","+str(round(numpy.median(debate_s[u]), 2))+"\n")
    for u in exs:
        output.write("owl, "+str(round(numpy.median(debate_eu[u]), 2))+"\n")
    for u in answerer_scores:
        output.write("overall, "+str(round(numpy.median(debate[u]), 2))+"\n")
    output.close()
    ################################################################################################################
    if os.path.exists("temp_files/com_"+name+"_"+tag+".pik"):
        comlook(tag)
        return
    qlen = []
    for urank in userranklists:
        qlen.append(len(urank))
    #print "qlen: "+str(numpy.mean(qlen))+" +/- "+str(numpy.std(qlen))

    showup = 0
    showup_nmrr = 0
    showup_ans = 0
    nmrr_better = 0
    nmrr_worse = 0
    com = []
    com_median = []
    com_nmrr = []
    com_ans = []
    com_or = []
    com_nmrr_median = []
    com_ans_median = []
    com_or_median = []
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
            #com_nmrr.append(float(numpy.mean(pos_nmrr))/len(urank))
            #com_nmrr_median.append(float(numpy.median(pos_nmrr))/len(urank))
        if len(pos_repu)!=0:
            showup_ans += 1
            #com_ans.append(float(numpy.mean(pos_repu))/len(urank))
            #com_ans_median.append(float(numpy.median(pos_repu))/len(urank))
        #if len(pos_nmrr)!=0 or len(pos_repu)!=0:
            #com_or.append(float(numpy.mean(list(set(exs) | set(ua))))/len(urank))
            #com_or_median.append(float(numpy.median(list(set(exs) | set(ua))))/len(urank))
        if len(pos_nmrr)!=0 and len(pos_repu)!=0:
            showup += 1
            c = [numpy.mean(pos_nmrr), numpy.mean(pos_repu), len(urank)]
            com.append(c)
            c_median = [numpy.median(pos_nmrr), numpy.median(pos_repu), len(urank)]
            com_median.append(c_median)
            if numpy.median(pos_nmrr)<numpy.median(pos_repu):
                nmrr_better+=1
            if numpy.median(pos_nmrr)>numpy.median(pos_repu):
                nmrr_worse+=1
    print "answered by owl: "+str(showup_nmrr)+", answered by "+name+" : "+str(showup_ans)
    print 'overall both-answered questions: '+str(showup)+', better by nmrr: '+str(nmrr_better)+', worse by nmrr: '+str(nmrr_worse)
    print "---"

    dumpfile(com, "com_"+name+"_"+tag)
    print 'com done!'
    comlook(name+'_'+tag, name)
    dumpfile(com_median, "com_median_"+name+"_"+tag)
    print 'com_median done!'
    comlook_median(name+'_'+tag, name)
#savehist(userset)

def comlook_median(filename, name):
    com = loadfile("com_median_"+filename)
    print len(com)
    index = []
    nmrrs_mean = []
    nmrrs_std = []
    repus_mean = []
    repus_std = []
    for i in range(40):
        better = 0
        worse = 0
        index.append(i+1)
        nmrrs = []
        repus = []
        for j in range(len(com)):
            c = com[j]
            if c[2] == i+1:
                nmrrs.append(float(c[0])/c[2])
                repus.append(float(c[1])/c[2])
                if c[0]<c[1]:
                    better+=1
                if c[0]>c[1]:
                    worse+=1
        
        #print str(numpy.median(nmrrs))+"  "+str(numpy.std(nmrrs))
        #print str(numpy.median(repus))+"  "+str(numpy.std(repus))
        
        nmrrs_mean.append(1-numpy.median(nmrrs))
        nmrrs_std.append(numpy.std(nmrrs))
        repus_mean.append(1-numpy.median(repus))
        repus_std.append(numpy.std(repus))
        print 'overall questions: '+str(len(nmrrs))+', nmrr better: '+str(better)+', nmrr worse: '+str(worse)
        print i+1
        print '------'
    output = open("data/ans_quality_median_"+name+".csv", 'w')
    for i in index:
        output.write(str(i)+", "+str(nmrrs_mean[i-1])+", "+str(repus_mean[i-1])+"\n")
    output.close()

def comlook(filename, name):
    com = loadfile("com_"+filename)
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
        
        #print str(numpy.mean(nmrrs))+"  "+str(numpy.std(nmrrs))
        #print str(numpy.mean(repus))+"  "+str(numpy.std(repus))
        
        nmrrs_mean.append(1-numpy.mean(nmrrs))
        nmrrs_std.append(numpy.std(nmrrs))
        repus_mean.append(1-numpy.mean(repus))
        repus_std.append(numpy.std(repus))
        #print len(nmrrs)
        #print i+1
        #print '------'
    output = open("data/ans_quality_"+name+".csv", 'w')
    for i in index:
        output.write(str(i)+", "+str(nmrrs_mean[i-1])+", "+str(repus_mean[i-1])+"\n")
    output.close()

