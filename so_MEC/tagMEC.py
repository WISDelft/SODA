import string
import psycopg2
import sys
import pickle
import os
import numpy as np


from calculate_mec import *
from compare_performance import *
from characterize_perference import *
from get_userAtt import *
from util_mec import *

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()

def read_expert(tag):
    input = open('totalranking_'+tag)


def comlook():
    f = open("com.pik")
    com = pickle.load(f)
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
    sys.exit(1)
    fig, ax = plt.subplots()
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(30)
    ax.set_xlabel('question debatableness (#answers)')
    ax.set_ylabel('1-Relative Rank')
    ax.plot(index, nmrrs_mean,  label='owl', lw=2)
    ax.plot(index, repus_mean, label='sparrow', lw=2)
    
    ax.legend(prop={'size':40})
    ax.set_ylim([0.5,1.0])
    #plt.gca().invert_yaxis()
    plt.show()
    plt.savefig('answerquality.pdf')
    sys.exit(1)

def v2a():
    fu = open("userranklists_c#.pik")
    userranklists = pickle.load(fu)
    qlen = []
    for urank in userranklists:
        qlen.append(len(urank))
    print "qlen: "+str(numpy.mean(qlen))+" +/- "+str(numpy.std(qlen))
    loglogplot(qlen)

def get_nrqst(users):
    nrqst = dict([])
    for u in users:
        cur.execute("select count(*) from sim_questions where owneruserid="+str(u))
        result = cur.fetchone()
        if result!=None:
            nrqst[u] = result[0]
        else:
            nrqst[u] = 0
    fnrq = open("nrqst", 'w')
    pickle.dump(nrqst, fnrq)
    sys.exit(1)


if __name__ == '__main__':
    tag = str(sys.argv[1])
    # get answerers and answer attributes
    if os.path.exists("temp_files/all_answerers.pik"):
        all_answerers = loadfile("all_answerers.pik")
    else:
        all_answerers = get_answerers()
    if os.path.exists("temp_files/answers_att_"+tag+".pik"):
        answers_att = loadfile("answers_att_"+tag+".pik")
        this_answerers = loadfile("answerers_"+tag+".pik")
    else:
        this_answerers, answers_att = get_answers_att(all_answerers, tag)
    
    # calculate MEC scores and get experts
    if os.path.exists("temp_files/experts_"+tag+".pik"):
        answerer_scores = loadfile("MECscores_"+tag+".pik")
        experts = loadfile("experts_"+tag+".pik")
    else:        
        experts, answerer_scores = get_MECscores(answers_att, this_answerers, tag)
    sys.exit(1)
    
    #users, experts = read_expert(tag)
    print "nr of experts: "+str(len(experts))
    #experts = sorted(experts)
    fa = open("answers_att_"+tag+".pik")
    answers_att = pickle.load(fa)
    get_motiv(experts, answers_att)
    #temp_cluster(users, experts, tag)
    #temp_analyze(users, experts, tag)
    #plot_illustration(users, experts, tag)
    #experts_explore(users, experts, tag)
    #comlook()
    


