import string
import psycopg2
import sys
import numpy
import pickle
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pylab as pl
import os
from scipy import interpolate
from scipy.interpolate import spline
from scipy.interpolate import interp1d
from sklearn import mixture
from sklearn import preprocessing
from datetime import datetime

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()
print "Connecting to DB: " + str(ver)

fe = open("experts_compc#"+".pik")
mec = pickle.load(fe)
'''fe2 = open("experts_c#"+".pik")
exs = pickle.load(fe)'''

fr = open("repu.pik")
repu_dict = pickle.load(fr)
fu = open("userranklists_c#"+".pik")
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

def wirte_clu_result(data):
    output = open('subgroups.csv', 'w')
    m,n = data.shape
    
    for j in range(n):
        for i in range(m):
            output.write(str(round(data[i,j],2)))
            if j==n-1:
                output.write("\n")
            else:
                output.write(",")

def get_stat(uset):
    mec_g = []
    repu_g = []
    regis_g = []
    nrans_g = []
    nrqst_g = []
    
    date_orig = datetime.strptime('2008-07-31', '%Y-%m-%d')
    for u in uset:
        mec_g.append(mec[u])
        repu_g.append(repu_dict[u])
        nrans_g.append(nrans[u])
        nrqst_g.append(nrqst[u])
        cur.execute("select TO_CHAR(creationdate,'YYYY-MM-DD') from users where id="+str(u))
        result = cur.fetchone()
        if result != None:
            date_regis = datetime.strptime(result[0], '%Y-%m-%d')
            diff = date_regis - date_orig
            regis_g.append(diff.days)
    return mec_g, repu_g, nrans_g, nrqst_g, regis_g

def clustering(data, clu_users):
    nr = 6
    colors = ['blue', 'red', 'green', 'yellow', 'black', 'cyan', 'magenta']
    labels = ['D1', 'I1', 'C1', 'D2', 'C2', 'D3']
    m, n = data.shape
    '''for k in range(nr+4):
        g = mixture.GMM(n_components=k+1)
        g.fit(data)
        print g.bic(data)'''
    
    g = mixture.GMM(n_components=nr)
    print g.fit(data)
    print np.round(g.means_, 2)
    fig, ax = plt.subplots()
    clus = g.predict(data)
    
    wirte_clu_result(g.means_)
    ustats = []
    mec_gs = []
    repu_gs = []
    nrans_gs = []
    nrqst_gs = []
    regis_gs = []
    for i in range(nr):
        sums = 0
        ax.plot(g.means_[i], color=colors[i])
        for j in range(m):
            if clus[j] == i:
                sums += 1
                ustats.append(clu_users[j])
        print "the nr of users in cluster "+colors[i] + " is: "+str(sums)
        mec_g, repu_g, nrans_g, nrqst_g, regis_g = get_stat(ustats)
        print "  mean MEC: "+str(numpy.mean(mec_g))+" median: "+str(numpy.median(mec_g))+", std:"+str(numpy.std(mec_g))
        print "  mean reputation: "+str(numpy.mean(repu_g))+" median: "+str(numpy.median(repu_g))+", std:"+str(numpy.std(repu_g))
        print "  mean #answers: "+str(numpy.mean(nrans_g))+" median: "+str(numpy.median(nrans_g))+", std:"+str(numpy.std(nrans_g))
        print "  mean #questions: "+str(numpy.mean(nrqst_g))+" median: "+str(numpy.median(nrqst_g))+", std:"+str(numpy.std(nrqst_g))
        print "  mean registration time: "+str(numpy.mean(regis_g))+" median: "+str(numpy.median(regis_g))+", std:"+str(numpy.std(regis_g))
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
    ax.set_xlabel("Days in a year", fontsize=20)
    ax.set_ylabel("#Answers posted", fontsize=20)
    ax.legend()
    '''ax2 = fig.add_axes()
    ax2.boxplot(mec_gs, colors)'''
    plt.show()
    sys.exit(1)

    
    '''clus = []
    for i in range(m):
        clu = g.predict(list(data[i, :]))
        clus.append(clu)
    print clus'''

def temp_cluster(users, experts, tag):
    date = '2012-01-01'
    ac_data = []
    
    fua = open("ua_"+str(tag)+".pik")
    ua = pickle.load(fua)
    
    clu_users = []
    '''for u in ua:
        #if u in experts:
            #continue
        if check_date(u, date) and os.path.exists("tempcsharp/ans"+str(u)+".csv"):
            days, impressions = numpy.loadtxt("tempcsharp/ans"+str(u)+".csv", unpack=True,
                                              converters={ 0: mdates.strpdate2num('%Y-%m-%d')})
            temp_ac = []
            temp_index = []
            #print days
            if isinstance(days, numpy.float64):
                temp_ac.append(impressions)
                for i in range(364):
                    temp_ac.append(0)
            else:
                temp_index = list(days)
                for i in range(365):
                    if (days[0] + i) in days:
                        temp_ac.append(impressions[temp_index.index(days[0] + i)])
                    else:
                        temp_ac.append(0)
            ac_data.append(temp_ac)
            clu_users.append(u)
    fac = open("ac_data_ua.pik", 'w')
    pickle.dump(ac_data, fac)
    fclu = open("clu_users_ua.pik", 'w')
    pickle.dump(clu_users, fclu)'''
    
    fac = open("ac_data.pik")
    ac_data = pickle.load(fac)
    fclu = open("clu_users.pik")
    clu_users = pickle.load(fclu)
    print len(clu_users)
    
    ac_data = np.array(ac_data)
    #ac_data = preprocessing.scale(ac_data)
    clustering(ac_data, clu_users)
    sys.exit(1)
    return 0

def check_date(id, date):
    cur.execute("select id from users where id="+str(id)+" and TO_CHAR(creationdate,'YYYY-MM-DD') <'" + date +"'")
    result = cur.fetchone()
    if result == None:
        return False
    else:
        return True

def check_date2(id, date1, date2):
    cur.execute("select id from users where id="+str(id)+" and TO_CHAR(creationdate,'YYYY-MM-DD') >'" + date1 +"' and TO_CHAR(creationdate,'YYYY-MM-DD') <='"+date2+"'")
    result = cur.fetchone()
    if result == None:
        return False
    else:
        return True

def shist(userscore):
    scores = dict([])
    for us in userscore:
        cur.execute("select TO_CHAR(creationdate,'YYYY-MM-DD') from users where id="+str(us))
        result = cur.fetchone()
        if result != None:
            if scores.has_key(result[0]):
                scores[result[0]] = scores[result[0]] + 1
            else:
                scores[result[0]] = 1

    
    #write_ac(scores, 'sparrow')
    #plot_ac('userscore')
    '''pl.hist(scores,bins=50)
    pl.show() #bbox_inches=0,'''

def temp_analyze(users, experts, tag):
    nr_exp = 0
    nr_ac = 0
    date = '2013-10-10'
    ac_qst = dict([])
    ac_ans = dict([])
    ac_cmt = dict([])
    
    fua = open("ua_"+str(tag)+".pik")
    ua = pickle.load(fua)
    #shist(experts)
    #shist(ua)
    #regis_plot()

    
    '''for u in experts:
        if check_date(u, date):
            nr_exp += 1
            ac_flag1, time_counts_qst = analyze_one(u, 'sim_questions')
            if ac_flag1:
                #print time_counts_qst
                #write_ac(time_counts_qst, 'qst'+str(u))
                #plot_ac('qst'+str(u))
            ac_flag2, time_counts_ans = analyze_one(u, 'sim_answers')
            if ac_flag2:
                print time_counts_ans
                #write_ac(time_counts_ans, 'ans'+str(u))
                #plot_ac('ans'+str(u))
            
            ac_flag3, time_counts_cmt = analyze_one(u, 'comments')
            if ac_flag3:
                write_ac(time_counts_cmt, 'cmt'+str(u))
                #plot_ac('cmt'+str(u))
            
            if ac_flag1 or ac_flag2 or ac_flag3:
                nr_ac += 1
                ac_qst = combine(ac_qst, time_counts_qst)
                ac_ans = combine(ac_ans, time_counts_ans)
                ac_cmt = combine(ac_cmt, time_counts_cmt)'''
    
    '''print "Nr experts registered before "+date+": "+str(nr_exp)
    print "Nr of elephants that are still alive:" +str(nr_ac)
    write_ac(ac_qst, 'qstua')
    write_ac(ac_ans, 'ansua')
    write_ac(ac_cmt, 'cmtua')'''
    
    '''plot_ac('qst')
    plot_ac('ans')
    plot_ac('cmt')'''
    plot_all_explore()
    sys.exit(1)

def analyze_one(u, tb):
    
    ac_flag = False
    
    time_counts = dict([])
    if tb!='comments':
        cur.execute("select TO_CHAR(creationdate,'YYYY-MM-DD') from "+tb+" where owneruserid="+str(u))
    else:
        cur.execute("select TO_CHAR(creationdate,'YYYY-MM-DD') from "+tb+" where userid="+str(u))
    results = cur.fetchall()
    for r in results:
        day = r[0]
        if day > '2012-01-01':
            ac_flag = True
        if day in time_counts:
            time_counts[day] += 1
        else:
            time_counts[day] = 1
        
    return ac_flag, time_counts

def combine(ac, time_counts):
    for d in time_counts:
        if d in ac:
            ac[d] += 1
        else:
            ac[d] = 1
    return ac

def write_ac(time_counts, type):
    sort_tc = []
    out = open("tempcsharp/"+type+".csv",'w')
    for d in time_counts:
        sort_tc.append(d)
    sort_tc = sorted(sort_tc)
    for d in sort_tc:
        out.write(str(d)+"    "+str(time_counts[d])+"\n")
    out.close()

def plot_ac(type):
    days, impressions = numpy.loadtxt("tempcsharp/"+type+".csv", unpack=True,
                                   converters={ 0: mdates.strpdate2num('%Y-%m-%d')})

    plt.plot_date(x=days, y=impressions, fmt="r-")
    plt.xlabel("time")
    plt.ylabel("#users")
    #plt.ylim([0,5])
    plt.grid(True)
    plt.show()

def analyze_one2(u, tb, date1, date2):
    
    time_counts = 0
    if tb!='comments':
        cur.execute("select count(TO_CHAR(creationdate,'YYYY-MM-DD')) from "+tb+" where owneruserid="+str(u)+" and TO_CHAR(creationdate,'YYYY-MM-DD') >'" + date1 +"' and TO_CHAR(creationdate,'YYYY-MM-DD') <='"+date2+"'")
    else:
        cur.execute("select count(TO_CHAR(creationdate,'YYYY-MM-DD')) from "+tb+" where userid="+str(u)+" and TO_CHAR(creationdate,'YYYY-MM-DD') >'" + date1 +"' and TO_CHAR(creationdate,'YYYY-MM-DD') <='"+date2+"'")
    results = cur.fetchall()
    for r in results:
        count = r[0]
        time_counts += count

    return time_counts

def get_ac_count(this_uset, k):
    if k==0:
        start = '2008-09-01'
        end = '2009-09-01'
    if k==1:
        start = '2009-09-01'
        end = '2010-09-01'
    if k==2:
        start = '2010-09-01'
        end = '2011-09-01'
    if k==3:
        start = '2011-09-01'
        end = '2012-09-01'
    if k==4:
        start = '2012-09-01'
        end = '2013-09-01'

    sum_ac = 0
    for u in this_uset:
        #sum_ac += analyze_one2(u, 'sim_answers', start, end)
        if os.path.exists("tempcsharp/ans"+str(u)+".csv"):
            days, impressions = numpy.loadtxt("tempcsharp/ans"+str(u)+".csv", unpack=True,dtype='str')
            #print days
            for i in range(len(days)):
                if days[i]>start and days[i]<=end:
                    sum_ac += int(impressions[i])
    return sum_ac

def plot_all():
    #nr = 14161
    #graphs = ['qstua', 'ansua','cmtua']
    nr = 11910
    graphs = ['ans', 'qst', 'cmt']
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']
    labels = [ '#Answer', '#Question', '#Comment']
    
    '''uset1 = []
    uset2 = []
    uset3 = []
    uset4 = []
    uset5 = []
    for u in mec:
        if check_date2(u, '2008-09-01', '2009-09-01'):
            uset1.append(u)
        elif check_date2(u, '2009-09-01', '2010-09-01'):
            uset2.append(u)
        elif check_date2(u, '2010-09-01', '2011-09-01'):
            uset3.append(u)
        elif check_date2(u, '2011-09-01', '2012-09-01'):
            uset4.append(u)
        elif check_date2(u, '2012-09-01', '2013-09-01'):
            uset5.append(u)

    uset = [uset1, uset2, uset3, uset4, uset5]
    fuset = open("uset.pik", 'w')
    pickle.dump(uset, fuset)'''
    fuset = open("usetua.pik")
    uset = pickle.load(fuset)
    

    stack_data = numpy.zeros((5,5))
    for i in range(5):
        print "the "+str(i)+"th year"
        print len(uset[i])
        for j in range(5):
            if j>i:
                continue
            stack_data[i,j] = get_ac_count(uset[j], i)

    print stack_data


def plot_all_explore():
    #nr = 11910
    #graphs = ['qst', 'ans','cmt']
    #nr = 14161
    graphs = ['qstua', 'ansua','cmtua']
    
    #graphs = ['qstall', 'ansall', 'cmtall'] #,
    
    colors = ['blue', 'red', 'green', 'yellow', 'cyan', 'magenta']
    labels = ['#Question', '#Answer', '#Comment']
 
    fig, ax = plt.subplots()
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
             item.set_fontsize(30)
    for i in range(len(graphs)):
        pre = 0
        days, impressions = numpy.loadtxt("tempcsharp/"+graphs[i]+".csv", unpack=True,
                                          converters={ 0: mdates.strpdate2num('%Y-%m-%d')})
        days_s1, impressions_t = numpy.loadtxt("tempcsharp/"+graphs[i]+".csv", unpack=True,dtype='str')
        impressions_log = []
        nr_users = []
        fnr = open("nr_users_"+graphs[i]+".pik")
        nr_users = pickle.load(fnr)
        fnr.close()
        for j in range(len(impressions)):
            #impressions_log.append(sum(impressions[0:j]))
            #print str(sum(impressions[0:j]))+"    "+str(nr_users[j])
            if nr_users[j]==0:
                impressions_log.append(sum(impressions[0:j]))  #av
                #impressions_log.append(impressions[j])      #avin
                continue
            impressions_log.append(float(sum(impressions[0:j]))/nr_users[j])
            #nr_users.append(avg_current(sum(impressions[0:j]), days[j]))
            #impressions_log.append(float(sum(impressions[0:j]))/nr_users[j])
        
        '''fnr = open("nr_users_"+graphs[i]+".pik", 'w')
        pickle.dump(nr_users, fnr)
        fnr.close()'''
        output = open("datas/fig7sparrow"+str(graphs[i])+".csv", 'w')
        for i in range(len(days_s1)):
            output.write(days_s1[i]+", "+str(impressions_log[i])+"\n")
        output.close()

        
        #ax.plot_date(x=days[0:-10], y=impressions_log[0:-10], fmt="r-", color=colors[i], label = labels[i], lw=2)
        #ax.set_ylabel(graphs[i]+" count")
        #plt.ylim([0,5])
        ax.grid(True)
        ax.hold(True)
        ax.set_xlabel("Time")
        ax.set_ylabel("Averaged activity count")#Instaneous
        ax.legend(loc=2,prop={'size':30})
        plt.show()

    '''nr = 14161
    graphs = ['qstua', 'ansua','cmtua']
    labels = ['#Question', '#Answer', '#Comment']

    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(30)
    for i in range(len(graphs)):
        pre = 0
        days, impressions = numpy.loadtxt("tempcsharp/"+graphs[i]+".csv", unpack=True,
                                          converters={ 0: mdates.strpdate2num('%Y-%m-%d')})
        impressions_log = []
        nr_users = []
        fnr = open("nr_users_"+graphs[i]+".pik")
        nr_users = pickle.load(fnr)
        fnr.close()
        for j in range(len(impressions)):
          #impressions_log.append(sum(impressions[0:j]))
          #print str(sum(impressions[0:j]))+"    "+str(nr_users[j])
          if nr_users[j]==0:
              impressions_log.append(sum(impressions[0:j]))
              continue
          impressions_log.append(float(sum(impressions[0:j]))/nr_users[j])


        ax.plot_date(x=days[0:-10], y=impressions_log[0:-10], fmt="r-", color=colors[i+3], label = labels[i]+" by sparrow", lw=2)

        #ax.set_ylabel(graphs[i]+" count")
        #plt.ylim([0,5])
        ax.grid(True)
        ax.hold(True)'''



    ax.set_xlabel("Time")
    ax.set_ylabel("Averaged activity count")#Instaneous
    ax.legend(loc=2,prop={'size':30})
    plt.show()
    plt.savefig('eac.pdf')




def regis_plot():
    fig,ax=plt.subplots()
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(30)
    days1, impressions = numpy.loadtxt("tempcsharp/owl.csv", unpack=True,
                                      converters={ 0: mdates.strpdate2num('%Y-%m-%d')})
    days_s1, impressions_t = numpy.loadtxt("tempcsharp/owl.csv", unpack=True,dtype='str')

    impressions_log1 = []
    print len(impressions)
    for i in range(len(impressions)):
        impressions_log1.append(sum(impressions[0:i]))
    ax.plot_date(x=days1, y=impressions_log1, fmt="r-", label='owls', color='blue', lw=2)



    days2, impressions = numpy.loadtxt("tempcsharp/sparrow.csv", unpack=True,
                                    converters={ 0: mdates.strpdate2num('%Y-%m-%d')})
    days_s2, impressions_t = numpy.loadtxt("tempcsharp/sparrow.csv", unpack=True,dtype='str')
    impressions_log2 = []
    print len(impressions)
    for i in range(len(impressions)):
        impressions_log2.append(sum(impressions[0:i]))
    ax.plot_date(x=days2, y=impressions_log2, fmt="r-", label='sparrows', color='red', lw=2)



    days3, impressions = numpy.loadtxt("tempcsharp/all.csv", unpack=True,
                                  converters={ 0: mdates.strpdate2num('%Y-%m-%d')})
    days_s3, impressions_t = numpy.loadtxt("tempcsharp/all.csv", unpack=True,dtype='str')
    impressions_log3 = []
    print len(impressions)
    for i in range(len(impressions)):
        impressions_log3.append(sum(impressions[0:i]))
    ax.plot_date(x=days3, y=impressions_log3, fmt="r-", label='all', color='green', lw=2)


    ax.set_xlabel("Registration time")
    ax.set_ylabel("#Users")

    days1 = list(days1)
    days2 = list(days2)
    days3 = list(days3)
    output = open("datas/register.csv", 'w')
    for i in range(len(days1)):
        if days1[i] in days2 and days1[i] in days3:
            j = days2.index(days1[i])
            k = days3.index(days3[i])
            output.write(days_s1[i]+", "+str(impressions_log1[i])+", "+str(impressions_log2[j])+", "+str(str(impressions_log3[k]))+"\n")
    output.close()


    #plt.ylim([0,5])
    plt.legend(prop={'size':30}, loc = 2)
    plt.grid(True)
    plt.show()
    sys.exit(1)

def avg_current(sums, current_day):
    days2, impressions2 = numpy.loadtxt("tempcsharp/all.csv", unpack=True,
                                        converters={ 0: mdates.strpdate2num('%Y-%m-%d')})
    impressions_log2 = []
    nr_users = 0
    for i in range(len(days2)):
        if days2[i]>=current_day:
            nr_users = sum(impressions2[0:i])
            #print nr_users
            break
    return nr_users


'''xnew = range(len(impressions))
ynew = impressions
xi = np.linspace(0, len(impressions), int(float(len(impressions)/15)))
for ind in range(len(xi)):
xi[ind] = int(xi[ind])
print len(xi)
print xi
power_smooth = spline(xnew,impressions,xi)
print len(power_smooth)

days2 = []
for k in range(len(days)):
if k in xi:
print k
days2.append(days[k])
print len(days2)
if len(power_smooth)!=len(days2):
power_smooth = power_smooth[0:-1]'''