'''
    all power-law distributed
    act - exp, slightly correlated
    act - par, linearly correlated, even remove the comments to the questions/answers by the same users
               so normalize par here by dividing to act, as there are very small amount of users who only comment while not answer
    
'''


import pandas as pd
import matplotlib.pyplot as plt
from numpy import corrcoef, sum, log, arange, mean, median, std
from numpy.random import rand
import scipy.stats  as stats
import math
import sys

#from pylab import pcolor, show, colorbar, xticks, yticks

location='/Users/jyang3/Projects/SODA/so_RQ/code/exploratory_analysis/'

t_test_set = ['c#', 'asp.net-mvc', 'windows', 'oop', 'regex', 'assembly']
#t_test_set = ['c#', 'asp.net-mvc']
    
def plot_distribution(property, fname, t):
    fig, ax = plt.subplots()
    #property.hist(ax=ax, bins=100, bottom=0.1)
    ax.hist(property, bins=100)
    ax.set_yscale('log')
    ax.set_xscale('log')
    plt.savefig('./exploratory_analysis/'+t+'_'+fname+'.png')
def plot_correlation(data, t):
    R = corrcoef(data)
    plt.figure()
    plt.xticks([0.5,1.5,2.5], ["act", "exp", "par"], rotation='vertical')
    plt.yticks([0.5,1.5,2.5], ["act", "exp", "par"])
    plt.pcolor(R)
    plt.tight_layout()
    plt.savefig('./exploratory_analysis/'+t+'_corr'+'.png')
    #yticks(arange(0.5,10.5),range(0,10))
    #xticks(arange(0.5,10.5),range(0,10))

def normalize_par(act, exp, par):
    act_new = []
    exp_new = []
    par_new = []
    print '.Overall analysis'
    print '..Before removing: ', len(act), 'users'
    ca = 0
    for i in xrange(len(act)):
        if act[i] != 0:
            act_new.append(act[i])
            exp_new.append(exp[i])
            par_new.append(float(par[i])/act[i])
    print '..After removing: ', len(act_new), 'users'
    print '..Percentage of only comment: ', 1-float(len(act_new))/len(act)
    k = 0
    for u in par:
        if par[u]!=0:
            k += 1
    print '..Percentage of only answer: ', 1-float(k)/len(act)
    print '..#Answers: ', len(act_new)
    print '..#Commenters: ', k
    print '..#C./#A.: ', float(k)/len(act_new)
    return act_new, exp_new, par_new

def user_analysis():
    for t in t_test_set:
        print 'Analyzing ', t
        u_data = pd.read_csv(location+'par_normed_exp_data_repu_norm_users_'+t+'.csv')
        act = u_data['act']
        exp = u_data['exp']
        par = u_data['par']
        # par = par-act
        act, exp, par = normalize_par(act, exp, par)
        ######### analyzig user side
        plot_distribution(act, 'act', t)
        plot_distribution(exp, 'exp', t)
        plot_distribution(par, 'par', t)
        print '.data description'
        #print '..', u_data.describe()
        print '..act', mean(act), median(act), std(act)
        print '..exp', mean(exp), median(exp), std(exp)
        print '..par', mean(par), median(par), std(par)
        
        all_data = []
        all_data.append(act)
        all_data.append(exp)
        all_data.append(par)
        plot_correlation(all_data, t)
        print '.correlation between features'
        print '..act-exp', stats.pearsonr(act, exp)
        print '..act-par', stats.pearsonr(act, par)
        print '..par-exp', stats.pearsonr(par, exp)
    return 0

def topic_analysis():
    for t in t_test_set:
        print 'Analyzing ', t
        u_data = pd.read_csv(location+'par_normed_exp_data_repu_norm_topics_'+t+'.csv')
        '''act = u_data['act']
        exp = u_data['exp']
        par = u_data['par']
        # par = par-act
        act, exp, par = normalize_par(act, exp, par)
        print '.data description 1'
        #print '..', u_data.describe()
        print '..act', mean(act), median(act), std(act)
        print '..exp', mean(exp), median(exp), std(exp)
        print '..par', mean(par), median(par), std(par)'''
        
        act_rank = u_data['r_act']
        exp_rank = u_data['r_exp']
        par_rank = u_data['r_par']
        par_rank = [float(item) for item in par_rank if item!='None']
        #print par_rank
        
        # par = par-act
        #act_rank, exp_rank, par_rank = normalize_par(act_rank, exp_rank, par_rank)
        print '.data description 2'
        #print '..', u_data.describe()
        print '..act_rank', mean(act_rank), median(act_rank), std(act_rank)
        print '..exp_rank', mean(exp_rank), median(exp_rank), std(exp_rank)
        print '..par_rank', mean(par_rank), median(par_rank), std(par_rank)
        
        act_pear = u_data['pear_act']
        exp_pear = u_data['pear_exp']
        par_pear = u_data['pear_par']
        par_pear = [float(item) for item in par_pear if not math.isnan(item)]
        
        # par = par-act
        #act_rank, exp_rank, par_rank = normalize_par(act_rank, exp_rank, par_rank)
        print '.data description 3'
        #print '..', u_data.describe()
        print '..act_pear', mean(act_pear), median(act_pear), std(act_pear)
        print '..exp_pear', mean(exp_pear), median(exp_pear), std(exp_pear)
        print '..par_pear', mean(par_pear), median(par_pear), std(par_pear)
        
    return 0

if __name__ == '__main__':
    #user_analysis()
    topic_analysis()
    
    