import string
import psycopg2
import sys
import numpy
import pickle
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
import os
from matplotlib.patches import Ellipse

con = None
con = psycopg2.connect(database='sim_so', user='postgres', password='021709Yj')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()
print "Connecting to DB: " + str(ver)

def is_tag_in(tag, tags):
    for t in tags.split('|'):
        if t==tag:
            return True
    return False

def get_questions_att(users, tag):
    questions_att = []
    for u in users:
        cur.execute("select sim_questions.owneruserid, sim_questions.id, sim_questions.score, sim_questions.viewcount, sim_questions.duration, sim_questions.acceptedanswer, sim_questions.tags from sim_questions where sim_questions.owneruserid = "+str(u))
        result = cur.fetchall()
        if result != None:
            for r in result:
                if is_tag_in(tag, r[6]):
                    questions_att.append([r[0], r[1], r[2], r[3], r[4], r[5]])
        #print questions_att
    f = open("questions_att_"+tag+".pik", "w")
    pickle.dump(questions_att, f)
    return questions_att

def get_answers_att(users, tag):
    answers_att = []
    nr = 0
    for u in users:
        nr += 1
        if nr%1000 == 0:
            print "extracting the answer record of the "+str(nr)+"th answerer"
        cur.execute("select aowneruserid, aid, qid, ascore, qviewcount, qduration, qscore, qtags from sim_qa where aowneruserid = "+str(u))
        result = cur.fetchall()
        if result != None:
            for r in result:
                if is_tag_in(tag, r[7]):
                    answers_att.append([r[0], r[1], r[2], r[3], r[4], r[5], r[6]])
    f = open("answers_att_"+tag+".pik", "w")
    pickle.dump(answers_att, f)
    return answers_att


''' features
    answers:     quality std
    preference:  a->q quality mean/std, popularity mean/std, difficult mean/std
    activeness:  #questions, #answers, #both
    consumeness: #q/#a
    questions:   quality mean/std, popularity mean/std, difficult mean/std, uacc
    seriousness: #doac, q->a fraction, scholar
    self-learner: #a->q, #a/#q(own)
    helpfulness: #a/#q'''
def extract_ka_feature(questions_att, answers_att, users, experts):
    uf_base = []
    uf_q = []
    lb_base = []
    lb_q = []
    users_q = []
    
    nr = 0
    for u in users:
        nr += 1
        if nr%1000 == 0:
            print "extracting the feature of the "+str(nr)+"th answerer"
        qatt = []
        aatt = []
        for question in questions_att:
            if question[0] != u:
                continue
            qatt.append(question)
        for answer in answers_att:
            if answer[0] != u:
                continue
            aatt.append(answer)
                
        ufa = []
        ascores=[]
        aqscores = []
        apop=[]
        adiff=[]
        self_ans = []
        #print aatt
        for a in aatt:
            ascores.append(a[3])
            apop.append(a[4])
            adiff.append(a[5])
            aqscores.append(a[6])
            self_ans.append(a[2])
        adiff = filter(lambda x: x != None, adiff)
        #if len(ascores) == 1:
            #ascores = []
        ufa=[numpy.std(ascores), numpy.mean(aqscores), numpy.std(aqscores), numpy.mean(apop), numpy.std(apop), numpy.mean(adiff), numpy.std(adiff)]
        #print ufa

        ufq = []
        self_learner = 0
        if len(qatt) != 0:
            qscores=[]
            qpop = []
            qdiff = []
            qacc = 0
            uacc = 0
            doac = 0
            for q in qatt:
                qscores.append(q[2])
                qpop.append(q[3])
                if q[4] == None:
                    uacc += 1
                else:
                    qdiff.append(q[4])
                if q[5] != None:
                    qacc += 1
                    doac += 1
                if q[1] in self_ans:
                    self_learner += 1
            qdiff = filter(lambda x: x != None, qdiff)
            ufq=[numpy.mean(qscores), numpy.std(qscores), numpy.mean(qpop), numpy.std(qpop), numpy.mean(qdiff), numpy.std(qdiff), float(uacc)/len(qatt), doac, float(qacc)/len(qatt)]
            cur.execute("select scholar from ufeatures where id="+str(u))
            result = cur.fetchall()
            tmp = result[0]
            if tmp[0] == None:
                ufq.append(0)
            else:
                ufq.append(1)
            uf_q.extend((self_learner, float(self_learner)/len(qatt), float(len(aatt))/len(qatt)))
    
        record_base = []
        record_q = []
        for elem in ufa:
            record_base.append(elem)
            record_q.append(elem)
        record_base.extend((len(aatt)))
        record_q.extend((len(qatt), len(aatt), len(qatt)+len(aatt), float(len(qatt))/len(aatt)))
        for elem in ufq:
            record_q.append(elem)

        uf_base.append(record_base)
        if u in experts:#experts.has_key(u):
            lb_base.append(1)
        else:
            lb_base.append(0)
                
        if len(qatt) !=0:
            users_q.append(u)
            uf_q.append(record_q)
            if u in experts:#experts.has_key(u):
                lb_q.append(1)
            else:
                lb_q.append(0)
                    
    return (uf_base, lb_base, uf_q, lb_q, users_q)


def get_labels_a(answers_att, users):
    au = []
    for u in users:
        aatt = []
        for answer in answers_att:
            if answer[0] != u:
                continue
            aatt.append(answer)
    
        ascores = []
        for a in aatt:
            ascores.append(a[3])
        if len(ascores) < 3:     # accuracy too low if uses with only one answer is included: no siginificant signal, however xxx
            continue
        au.append([u,numpy.mean(ascores)])
    print len(au)
    au = sorted(au, key = lambda au : au[1], reverse=False)
    experts = []
    eindex = int(len(au)*0.8)
    eu = au[eindex]
    for u in au:
        if u[1]>eu[1]:
            #print str(u[0])+" "+str(u[1])
            experts.append(u[0])
    return experts

def get_labels_bestanswerer(answers_att, users):
    expert_dict=dict([])
    score_dict=dict([])
    experts = dict([])
    for a in answers_att:
        qid = a[2]
        score = a[3]
        uid = a[0]
        if not score_dict.has_key(qid):
            score_dict[qid] = score
            expert_dict[qid] = uid
        elif score_dict[qid] < score:
            score_dict[qid] = score
            expert_dict[qid] = uid

    for qid in expert_dict:
        if score_dict[qid]>=10:
            experts[expert_dict[qid]]=1
    return experts

def get_labels_repu(users, nums):
    experts = []
    repus = []
    for u in users:
        cur.execute("select reputation from users where id="+str(u))
        result = cur.fetchone()
        repus.append([u,result[0]])
    repus = sorted(repus, key = lambda repus : repus[1], reverse=True)
    for i in range(len(users)):
        if i<nums:
            uinfo = repus[i]
            #print uinfo
            experts.append(uinfo[0])
    return experts

def get_motiv(experts, answers_att):
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


def get_labels(answers_att, users, tag):
    # get all questions
    qids = dict([])
    userset = dict([])
    nrqst = dict([])
    for a in answers_att:
        qid = a[2]
        uid = a[0]
        qids[qid] = 0
        userset[uid] = 0
        nrqst[uid] = 0

    # get partial ranking for each question
    print  "Nr of questions: "+str(len(qids))
    index = 0
    userranklists = []
    if os.path.exists("userranklists_aa"+tag+".pik"):
        fu = open("userranklists_"+tag+".pik")
        userranklists = pickle.load(fu)
    else:
        #output = open("ranking_"+tag, 'w')
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
            qids[qid] = len(urank)
            userranklists.append(urank)
            '''output.write("[")
            for i in range(len(urank)):
                u = urank[i]
                output.write(str(u[0]))
                if i!=len(urank)-1:
                    output.write(",")
            output.write("],")'''

        fp = open("userranklists_"+tag+".pik", "w")
        pickle.dump(userranklists, fp)



    '''f = open("experts_log_"+tag+".pik")
    experts = pickle.load(f)


    for urank in userranklists:
        for i in range(len(urank)):
            pos = urank[i]
            u = pos[0]
            nrqst[u] += 1

    for u in experts:
        if nrqst[u]<5:
            print u'''
    #print "write ground truth done!"
    allqstlen = []
    for urank in userranklists:
        #allqstlen.append(numpy.log2(len(urank)+1))
        allqstlen.append(len(urank))
    print numpy.sum(allqstlen)
    print numpy.mean(allqstlen)
    print '----'
    savehist(allqstlen)
    '''qids store with qst length, userset store with their performance, and nrqst store with his qst number'''
    #calculate user expertise
    ss = []
    for urank in userranklists:
        for i in range(len(urank)):
            pos = urank[i]
            u = pos[0]
            ss.append(pos[1])
            #print str((float(1)/(i+1)))+" "+str((len(urank)))+ " "+str((float(1)/(i+1))*(len(urank)))
            #userset[u] = userset[u] + (float(1)/(i+1))*(numpy.log2(len(urank)+1))
            userset[u] = userset[u] + (float(1)/(i+1))*(float(len(urank)))
            nrqst[u] += 1
    print numpy.mean(ss)
    print numpy.std(ss)
    loglogplot(ss)
    print '----'
    experts = dict([])
    for u in userset:
        if nrqst[u]>=1:
        #print "user "+str(u) + " obtains " +str(userset[u]) + " for answering "+str(nrqst[u])+" questions"
            userset[u] = float(userset[u])/(nrqst[u]*numpy.mean(allqstlen))
            #experts[u] = userset[u]
            if userset[u]>=1:
                experts[u] = userset[u]

    f = open("experts_"+tag+".pik", "w")
    pickle.dump(experts, f)
    
    return experts

def overlapping(a, exs):
    ass = []
    for u in a:
        ass.append(a[u])
    ass = sorted(ass,reverse=True)
    cutoff = int(0.1*len(ass))
    thresh = ass[cutoff]
    assu = []
    for u in a:
        if a[u]>thresh:
            assu.append(u)
    return float(len(list(set(exs) & set(assu))))/len(assu)
def plot_bars(repu_dict, nrans, exs, ua, type):
    all = []
    s = []
    o = []
    
    if type=='reputation':
        a = dict([])
        a = repu_dict
        print overlapping(a, exs)
    if type=='#answers':
        a = dict([])
        a = nrans
        print overlapping(a, exs)
    if type=='Zscore':
        a = dict([])
        fqst = open("nrqst.pik")
        nrqst = pickle.load(fqst)
        for u in nrqst:
            a[u] = float(nrans[u]-nrqst[u])/math.sqrt(nrans[u]+nrqst[u])
        fz = open("zscore.pik", 'w')
        pickle.dump(a, fz)
        print overlapping(a, exs)
    for u in repu_dict:
        all.append(a[u])
        if u in ua:
            s.append(a[u])
        if u in exs:
            o.append(a[u])

    output = open(type+".csv", 'w')
    for item in s:
        output.write("sparrow, "+str(item)+"\n")
    for item in o:
        output.write("owl, "+str(item)+"\n")
    for item in all:
        output.write("overall, "+str(item)+"\n")

    sys.exit(1)
    means   = [numpy.mean(s),numpy.mean(o),numpy.mean(all)]           # Mean Data
    print means
    stds    = [(0,0,0), [numpy.std(s),numpy.std(o),numpy.std(all)]] # Standard deviation Data
    #peakval = ['26.82','26.4','61.17']   # String array of means

    ind = np.arange(len(means))
    width = 0.35
    colours = ['red','blue','green']

    fig,ax = plt.subplots()
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(30)
    #ax.bar(ind, means, width, color=colours, align='center', yerr=stds, ecolor='k')
    ax.boxplot((s,o,a))
    ax.set_ylabel(type)
    ax.set_xticks(ind)
    ax.set_xticklabels(['sparrow','owl','overall'])
#   autolabel(means,peakval, ind)
    plt.show()
    return 0
    
def autolabel(bars,peakval, ind):
    for ii,bar in enumerate(bars):
        height = bars[ii]
        plt.text(ind[ii], height-5, '%s'% (peakval[ii]), ha='center', va='bottom')


def plot_illustration(users, experts, tag):
    
    
    
    fu = open("userranklists_"+tag+".pik")
    userranklists = pickle.load(fu)
    nrans = dict([])
    for urank in userranklists:
        for i in range(len(urank)):
            this_u = urank[i]
            u = this_u[0]
            if nrans.has_key(u):
                nrans[u] += 1
            else:
                nrans[u] = 1

    fqst = open("nrqst.pik")
    nrqst = pickle.load(fqst)

    fua = open("ua_"+str(tag)+".pik")
    ua = pickle.load(fua)
    fe = open("experts_"+tag+".pik")
    exs = pickle.load(fe)
    '''output = open("datas/nr_ans_qst.csv", 'w')

    sa = []
    sb = []
    oa = []
    ob = []
    alla = []
    allb = []
    for u in ua:
        sa.append(nrans[u])
        sb.append(nrqst[u])
        #output.write("sparrow, "+str(nrans[u])+", "+str(nrqst[u])+"\n")
    for u in exs:
        #output.write("owl, "+str(nrans[u])+", "+str(nrqst[u])+"\n")
    for u in experts:
        #output.write("overall, "+str(nrans[u])+", "+str(nrqst[u])+"\n")

    sys.exit(1)'''
    # hist motivation graph
    '''fr = open("repu.pik")
    repu_dict = pickle.load(fr)

    plot_bars(repu_dict, nrans, exs, ua, 'Zscore')
    sys.exit(1)'''

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
            if experts[u]>=1:
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
    for u in users:
        debate_medium.append( numpy.median(debate[u]))
        if experts[u]>=1:
            debate_medium_eu.append( numpy.median(debate_eu[u]))
        if u in ua:
            debate_medium_s.append(numpy.median(debate_s[u]))
    '''output = open("datas/qst_deb.csv", 'w')

    for u in ua:
        output.write("sparrow, "+str(round(numpy.median(debate_s[u]), 2))+"\n")
    for u in exs:
        output.write("owl, "+str(round(numpy.median(debate_eu[u]), 2))+"\n")
    for u in experts:
        output.write("overall, "+str(round(numpy.median(debate[u]), 2))+"\n")'''
    loglogplot(debate_medium)
    
    sys.exit(1)
    '''print '********'
    print len(debate_medium_eu)
    print len(debate_medium_s)

    figt, ax2 =  plt.subplots()
    ax2.hold(False)
    y1, x1, z1 = ax2.hist(debate_medium_eu, 50)
    y2, x2, z2 = ax2.hist(debate_medium, 50)
    y3, x3, z3 = ax2.hist(debate_medium_s,10)

    fig =  plt.figure()
    ax = fig.add_subplot(111)
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(30)
    ax.hold(False)

    x1 = x1[0:-1]
    for i in range(0, len(y1)):
        y1[i] += 0.001
    ax.hold(True)
    ax.loglog(x1, y1, label = "answered by the owl", lw=2)
    print x1
    print y1
    x2 = x2[0:-1]
    for i in range(0, len(y2)):
        y2[i] += 0.001
    ax.loglog(x2, y2, label = "answered by all users", lw=2)


    x3 = x3[0:-1]
    for i in range(0, len(y3)):
        y3[i] += 0.001
    ax.loglog(x3, y3, label = "answered by the sparrow", lw=2)
    print x3
    print y3

    ax.set_ylabel('#questions')
    ax.set_xlabel('question debatableness (#answers)')

    ax.legend(prop={'size':40})
    plt.show()
    plt.savefig("owllen.pdf")
    sys.exit(1)'''


    #----illustration graph

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
        if u in ua:
            qd_s.append(numpy.mean(leni)+0.5 * np.random.randn())#
            aq_s.append(1-numpy.mean(rri)+0.04 * np.random.randn())#
        else:
            qd.append(numpy.mean(leni)+0.5 * np.random.randn())
            aq.append(1-numpy.mean(rri)+0.04 * np.random.randn())
        if experts[u]>=1:
            qd_o.append(numpy.mean(leni)+0.5 * np.random.randn())
            aq_o.append(1-numpy.mean(rri)+0.04 * np.random.randn())
            exs.append(u)
        if numpy.mean(leni)>3.0 and numpy.mean(rri)<0.37:
            ideals.append(u)

    output = open("datas/illustration.csv", 'w')

    for k in range(len(qd_s)):
        print k
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
    print float(len(list(set(ua) & set(ideals))))/len(ideals)
    print "-----------------hi"
    sz = []

    fig,ax = plt.subplots()
    #ax = fig.add_axes()#[0.1,0.1,0.8,0.8]
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    ax.plot([1,40],[0.63,0.63], lw=2, color='black')
    ax.plot([3,3], [0, 1], lw=2, color='black')
    ax.scatter(qd, aq, color='#CCCCFF', label='Others', s = 0.2)
    ax.scatter(qd_s, aq_s, color='#FF3366',  label='sparrow', s=0.03)
    ax.set_xscale('log')
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    qd_perf = []
    aq_perf = []
    for i in range(len(qd)):
        if qd[i]> st1:
            qd_perf.append(qd[i])
            aq_perf.append(aq[i])
    #ax.scatter(qd_perf, aq_perf, color='black', label='ideal users', s = 0.5)
    ax.set_xlabel("Question debatableness (#answers)")
    ax.set_ylabel("Answering quality (1-relative rank)")
    ax.set_ylim([0.0,1.0])
    ax.set_xlim([1,40])
    ax.set_xticks([1,2,3,5,10,20,30,40])
    #ax.legend(prop={'size':30})
    #plt.gca().invert_yaxis()
    ax2 = fig.add_axes([0.1,0.92,0.8,0.08])
    ax2.boxplot(qd_all,0, vert=False)
    ax2.set_xlim([1,40])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xscale('log')

    ax3 = fig.add_axes([0.92,0.1,0.8,0.08])
    ax3.boxplot(aq_all,0)
    ax3.set_ylim([0.0,1.0])
    ax3.set_xticks([])
    ax3.set_yticks([])
    #ax.boxplot(aq_all,positions = 8)
    plt.show()
    plt.savefig('scatter_nmrr_nrans_all.png')
    sys.exit(1)




    #----illustration graph 2
    repus = []
    nans = []
    repus_s = []
    nans_s = []
    repus_perf = []
    nans_perf = []

    index = 0
    fr = open("repu.pik")
    repu_dict = pickle.load(fr)
    for u in nrans:
        '''index += 1
        if index%1000 == 0:
            print 'the '+str(index)+"th user"
        cur.execute("select reputation from users where id="+str(u))
        result = cur.fetchone()
        repu_dict[u] = result[0]'''
        if u in ua:
            repus_s.append(repu_dict[u])
            nans_s.append(nrans[u])
        else:
            repus.append(repu_dict[u])
            nans.append(nrans[u])
        if u in ideals:
            repus_perf.append(repu_dict[u])
            nans_perf.append(nrans[u])



    fig,ax = plt.subplots()
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    #ax.scatter(nans, repus, color='black', label='Others', s = 0.05)
    #ax.scatter(nans_s, repus_s, color='red',  label='sparrow', s=0.05)
    el = Ellipse((numpy.mean(nans), numpy.mean(repus)), numpy.std(nans), numpy.std(repus), color='black')
    ax.add_patch(el)
    el_s = Ellipse((numpy.mean(nans_s), numpy.mean(repus_s)), numpy.std(nans_s), numpy.std(repus_s), color='red')
    ax.add_patch(el_s)
    el_perf = Ellipse((numpy.mean(nans_perf), numpy.mean(repus_perf)), numpy.std(nans_perf), numpy.std(repus_perf), color='black')
    ax.add_patch(el_perf)
    ax.annotate('the blue users', xy=(numpy.mean(nans_perf), numpy.mean(repus_perf)), xytext=(10, 3000), size=20,
                arrowprops=dict(facecolor='green', shrink=0.02),
                )
    ax.annotate('the black users', xy=(numpy.mean(nans), numpy.mean(repus)), xytext=(10, 500), size=20,
                arrowprops=dict(facecolor='green', shrink=0.02),
                )
    #ax.scatter(nans_perf, repus_perf, color='blue', label='ideal users', s = 0.5)
    '''ax.plot([0,40], [0.5,0.5],  lw=2, color='black')
    ax.plot([20,20], [0, 1], lw=2, color='black')'''
    ax.set_ylim([0, 16000])
    ax.set_xlim([0, 200])
    ax.set_xlabel("#answers given by a user")
    ax.set_ylabel("Reputation")
    #ax.legend(prop={'size':30})
    plt.show()
    plt.savefig('scatter_nrans_repu.png')
    sys.exit(1)
    '''qd = [] # question debatableness
    aq = [] # answer quality
    qd_s = []
    aq_s = []
    for i in range(40):
        r = [] #the rank of u in answering questions of debatableness i
        rua = []
        for u in debate:
            for j in range(len(debate[u])):
                temp_debate_list = debate[u]
                if temp_debate_list[j]==i+1:
                    temp_rank_list = rank[u]
                    r.append(temp_rank_list[j])
                    if u in ua:
                        rua.append(temp_rank_list[j])
        if len(r) != 0:
            qd.append(i)
            aq.append(numpy.mean(r))
        if len(rua) != 0:
            qd_s.append(i)
            aq_s.append(numpy.mean(rua))


    fig,ax = plt.subplots()
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    ax.scatter(qd, aq, color='black', label='Others', s=0.5)
    ax.scatter(qd_s, aq_s, color='red', label='sparrow', s=0.5)
    ax.set_xlabel("Question debatableness")
    ax.set_ylabel("Answering quality")
    ax.legend()
    plt.savefig('scatter_nmrr_nrans_avg.pdf')'''
    sys.exit(1)


    k = 1
    perf = dict([])
    for urank in userranklists:
        if len(urank)<k:
            continue
        for i in range(len(urank)):
            this_u = urank[i]
            u = this_u[0]
            if perf.has_key(u):
                perf[u].append(float(len(urank))/(i+1))
            else:
                perf[u]=[float(len(urank))/(i+1)]
                           
   


    a0 = []
    b0 = []
    a1 = []
    b1 = []
    a2 = []
    b2 = []
    for u in experts:
        flag1 = 0
        flag2 = 0
        a0.append(numpy.median(perf[u]))
        b0.append(numpy.log(nrans[u]))
        if experts[u]>=1:
            a1.append(numpy.median(perf[u]))
            b1.append(numpy.log(nrans[u]))
            flag1 = 1
        if u in ua:
            a2.append(numpy.median(perf[u]))
            b2.append(numpy.log(nrans[u]))
            flag2 = 1

    fig,ax = plt.subplots()
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    ax.scatter(a0,b0, label='Others', s=0.5)
    ax.scatter(a1,b1, color='blue', label='Owl', s=0.5)
    ax.scatter(a2,b2, color='green', label='Sparrow', s=0.5)
    ax.set_xlabel("Relative rank")
    ax.set_ylabel("log(#Answers)")
    ax.legend()
    plt.savefig('scatter_nmrr_nrans.pdf')

    sys.exit(1)

    fd = open("debate.pik",'w')
    pickle.dump(debate,fd)
    fd2 = open("debate_eu.pik",'w')
    pickle.dump(debate_eu,fd2)



def writeout(a, fname):
    fout = open("datas/"+fname+".csv", 'w')
    for x in a:
        fout.write(str(x)+"\n")
    fout.close()

def experts_explore(users, experts,tag):
    '''a = []
    for u in experts:
        a.append(experts[u])
    loglogplot(a)
    print 'csharp done!'''

    '''experts_repu = get_labels_repu(users, len(experts))
    print '-----'
    print len(list(set(experts) & set(experts_repu)))'''
    '''output = open("nmrr_repu_", 'w')
    output2 = open("nmrr_", 'w')
    output3 = open("repu_", 'w')'''
    fu = open("userranklists_"+tag+".pik")
    userranklists = pickle.load(fu)
    
    qlen = []
    for urank in userranklists:
        qlen.append(len(urank))
    #print "qlen: "+str(numpy.mean(qlen))+" +/- "+str(numpy.std(qlen))
    #loglogplot(qlen)
    #sys.exit(1)
    nrans = dict([])
    for urank in userranklists:
        for i in range(len(urank)):
            this_u = urank[i]
            u = this_u[0]
            if nrans.has_key(u):
                nrans[u] += 1
            else:
                nrans[u] = 1

    '''a = []
    b = []
    nran = []
    for u in experts:
        a.append(experts[u])
        b.append(nrans[u])
        nran.append(nrans[u])
    loglogplot(nran)
    sys.exit(1)'''
    '''pl.scatter(a,b)
    pl.savefig('scatter_nmrr_nrans.png')'''
    
    '''print numpy.mean(nran)
    print numpy.std(nran)
    print '----'

    figt, ax2 =  plt.subplots()
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.hold(False)
    y1, x1, z1 = ax2.hist(qlen, 50)
    y2, x2, z2 = ax2.hist(nran, 50)
    
    fig =  plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(False)
    ax = fig.add_subplot(111)
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    x1 = x1[0:-1]
    for i in range(0, len(y1)):
        y1[i] += 0.001
    ax.hold()

    xd,yd = numpy.log10(x1), numpy.log10(y1)
    xd = xd[0:8]
    yd = yd[0:8]
    polycoef = numpy.polyfit(xd, yd, 1)
    yfit = 10**( polycoef[0]*xd+polycoef[1] )

    ax.plot(x1, y1, label = "#question histogram", lw=2)
    #ax.plot(x1[0:8], yfit, '.')


    x2 = x2[0:-1]
    for i in range(0, len(y2)):
        y2[i] += 0.001
    ax.plot(x2, y2, label = "#user histogram", lw=2)
    xd,yd = numpy.log10(x2), numpy.log10(y2)
    xd = xd[0:8]
    yd = yd[0:8]
    polycoef = numpy.polyfit(xd, yd, 1)

    yfit = 10**( polycoef[0]*xd+polycoef[1] )
    #ax.plot(x2[0:8], yfit, '.')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('#question/user')
    ax.set_ylabel('#answer to a question or by a user')

    ax.legend()
    plt.show()
    plt.savefig("nrans.pdf")
    sys.exit(1)'''

    '''#nran = sorted(nran,reverse=True)
    cutoff = int(0.1*len(nran))
    cutoff = 15650
    thresh = nran[cutoff]
    print thresh
    
    a0 = []
    b0 = []
    a1 = []
    b1 = []
    a2 = []
    b2 = []
    a3 = []
    b3 = []
    experts2 = []
    experts_ans = []
    for u in experts:
        flag1 = 0
        flag2 = 0
        if experts[u]>=1:
            experts2.append(u)
            a1.append(numpy.log(experts[u]))
            b1.append(numpy.log(nrans[u]))
            flag1 = 1
        if len(experts_ans)<15650 and nrans[u]>=thresh:
            a2.append(numpy.log(experts[u]))
            b2.append(numpy.log(nrans[u]))
            experts_ans.append(u)
            flag2 = 1
        if flag1 ==1 and flag2 == 1:
            a0.append(numpy.log(experts[u]))
            b0.append(numpy.log(nrans[u]))
        if flag1 == 0 and flag2 == 0:
            a3.append(numpy.log(experts[u]))
            b3.append(numpy.log(nrans[u]))

        if experts[u]<0.2 and nrans[u]>10:
            print "hi "+str(u)
    fig,ax = plt.subplots()
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        item.set_fontsize(20)
    ax.scatter(a3,b3, color='yellow', label='Others', s=0.5)
    ax.scatter(a2,b2, color='blue', label='Sparrow', s=0.5)
    ax.scatter(a1,b1, color='green', label='Owl', s=0.5)
    ax.scatter(a0,b0, color='red', label='Sparrow&Owl', s=0.5)
    ax.set_xlabel("log(NMRR)")
    ax.set_ylabel("log(#Answers)")
    ax.legend()
    plt.savefig('scatter_nmrr_nrans.pdf')

    a = []
    for u in experts_ans:
        a.append(experts[u])
    loglogplot(a)
    a = sorted(a)
    print a
    
    print len(experts2)
    print len(experts_ans)
    print len(list(set(experts2) & set(experts_ans)))'''
    print "---"
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
            if u in experts2:
                pos_nmrr.append(i)
                u_nmrr.append(u)
            if u in experts_ans:
                pos_repu.append(i)
                u_repu.append(u)
        if len(pos_nmrr)!=0:
            showup_nmrr += 1
            com_nmrr.append(float(numpy.mean(pos_nmrr))/len(urank))
        if len(pos_repu)!=0:
            showup_ans += 1
            com_ans.append(float(numpy.mean(pos_repu))/len(urank))
        if len(pos_nmrr)!=0 or len(pos_repu)!=0:
            com_or.append(float(numpy.mean(list(set(experts2) | set(experts_ans))))/len(urank))
        if len(pos_nmrr)!=0 and len(pos_repu)!=0:
            if pos_nmrr[0]<pos_repu[0]:
                print '---'
                print urank
                print pos_nmrr
                print pos_repu
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
    pickle.dump(com, f)
    print 'com done!'
    sys.exit(1)
#savehist(userset)
def loglogplot(qlen):
    
    fig, ax = plt.subplots()
    y, x, z = ax.hist(qlen, 50)
    ax.hold(False)
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(25)
    x = x[0:-1]
    for i in range(0, len(y)):
        y[i] += 0.001
    print x
    print y

    output = open("datas/4b_overall.csv", 'w')
    for i in range(len(x)):
        output.write(str(x[i])+", "+str(y[i])+"\n")
    output.close()
    sys.exit(1)

    ax.loglog(x, y, lw=2)
    ax.set_ylabel('#users')
    ax.set_xlabel('#log(MEC)')
    ax.set_xlim([0.3, 10])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xticks([0.5, 1, 2, 5, 10])
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.show()
    plt.savefig('nrans_aq.pdf',bbox_inches='tight')


def savehist(userscore):
    scores = []
    for us in userscore:
        scores.append(userscore[us])
    pl.hist(scores,bins=50,log=True)
    pl.savefig('qlen.pdf') #bbox_inches=0,




