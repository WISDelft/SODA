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
from util_mec import *

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='wistudelft')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()

def post_qst_properties(experts, sparrows, allusers, tag):
    ex_pop = dict([])
    au_pop = dict([])
    ov_pop = dict([])
    f=open('data/post_pop_median.csv', 'w')
    for u in allusers:
        cur.execute('select viewcount, tags from questions where owneruserid='+str(u))
        qres = cur.fetchall()
        if qres==None or len(qres)==0:
            continue
        qres = [qr[0] for qr in qres if qr[0] is not None and is_tag_in(tag, qr[1])]
        if len(qres)==0:
            continue
        this_pop = numpy.median(qres)
        
        ov_pop[u] = this_pop
        if u in experts:
            ex_pop[u] = this_pop
        if u in sparrows:
            au_pop[u] = this_pop
    for u in au_pop:
        f.write("sparrow, "+str(au_pop[u])+"\n")
    for u in ex_pop:
        f.write("owl, "+str(ex_pop[u])+"\n")
    for u in ov_pop:
        f.write("overall, "+str(ov_pop[u])+"\n")
    f.close()        
    print 'data/post_pop_median.csv done!'
    
    f=open('data/post_diff_median.csv', 'w')
    ex_diff = dict([])
    au_diff = dict([])
    ov_diff = dict([])
    for u in allusers:    
        cur.execute('select duration, tags from questions where owneruserid='+str(u))
        qres = cur.fetchall()
        if qres==None or len(qres)==0:
            continue
        qres = [qr[0] for qr in qres if qr[0] is not None and is_tag_in(tag, qr[1])]
        if len(qres)==0:
            continue
        this_diff = numpy.median(qres)
        
        ov_diff[u] = this_diff
        if u in experts:
            ex_diff[u] = this_diff
        if u in sparrows:
            au_diff[u] = this_diff
    for u in au_diff:
        f.write("sparrow, "+str(au_diff[u])+"\n")
    for u in ex_diff:
        f.write("owl, "+str(ex_diff[u])+"\n")
    for u in ov_diff:
        f.write("overall, "+str(ov_diff[u])+"\n")
    f.close()      
    print 'data/post_diff_median.csv done!'
    return 0
def ansd_qst_properties(experts, sparrows, allusers, tag):
    ex_pop = dict([])
    au_pop = dict([])
    ov_pop = dict([])
    f=open('data/ansd_pop_median.csv', 'w')
    for u in allusers:
        cur.execute('select qviewcount, qtags from sim_qa where aowneruserid='+str(u))
        qres = cur.fetchall()
        if qres==None or len(qres)==0:
            continue
        qres = [qr[0] for qr in qres if qr[0] is not None and is_tag_in(tag, qr[1])]
        if len(qres)==0:
            continue
        this_pop = numpy.median(qres)
        
        ov_pop[u] = this_pop
        if u in experts:
            ex_pop[u] = this_pop
        if u in sparrows:
            au_pop[u] = this_pop
    for u in au_pop:
        f.write("sparrow, "+str(au_pop[u])+"\n")
    for u in ex_pop:
        f.write("owl, "+str(ex_pop[u])+"\n")
    for u in ov_pop:
        f.write("overall, "+str(ov_pop[u])+"\n")
    f.close()
    print 'data/ansd_pop_median.csv done!'
    
    f=open('data/ansd_diff_median.csv', 'w')
    ex_diff = dict([])
    au_diff = dict([])
    ov_diff = dict([])
    for u in allusers:    
        cur.execute('select qduration, qtags from sim_qa where aowneruserid='+str(u))
        qres = cur.fetchall()
        if qres==None or len(qres)==0:
            continue
        qres = [qr[0] for qr in qres if qr[0] is not None and is_tag_in(tag, qr[1])]
        if len(qres)==0:
            continue
        this_diff = numpy.median(qres)
        
        ov_diff[u] = this_diff
        if u in experts:
            ex_diff[u] = this_diff
        if u in sparrows:
            au_diff[u] = this_diff
    for u in au_diff:
        f.write("sparrow, "+str(au_diff[u])+"\n")
    for u in ex_diff:
        f.write("owl, "+str(ex_diff[u])+"\n")
    for u in ov_diff:
        f.write("overall, "+str(ov_diff[u])+"\n")
    f.close()      
    print 'data/ansd_diff_median.csv done!'
    return 0

def post_qst_properties_all(experts, sparrows, allusers, tag):
    f=open('data/post_pop_all.csv', 'w')
    for u in allusers:
        cur.execute('select viewcount, tags from questions where owneruserid='+str(u))
        qres = cur.fetchall()
        if qres==None or len(qres)==0:
            continue
        qres = [qr[0] for qr in qres if qr[0] is not None and is_tag_in(tag, qr[1])]
        if len(qres)==0:
            continue
        for q in qres:
            f.write("overall, "+str(q)+"\n")
            if u in experts:
                f.write("owl, "+str(q)+"\n")
            if u in sparrows:
                f.write("sparrow, "+str(q)+"\n")
    f.close()        
    print 'data/post_pop_all.csv done!'
    
    f=open('data/post_diff_all.csv', 'w')
    for u in allusers:    
        cur.execute('select duration, tags from questions where owneruserid='+str(u))
        qres = cur.fetchall()
        if qres==None or len(qres)==0:
            continue
        qres = [qr[0] for qr in qres if qr[0] is not None and is_tag_in(tag, qr[1])]
        if len(qres)==0:
            continue
        for q in qres:
            f.write("overall, "+str(q)+"\n")
            if u in experts:
                f.write("owl, "+str(q)+"\n")
            if u in sparrows:
                f.write("sparrow, "+str(q)+"\n")
    f.close()      
    print 'data/post_diff_all.csv done!'
    return 0
def ansd_qst_properties_all(experts, sparrows, allusers, tag):
    f=open('data/ansd_pop_all.csv', 'w')
    for u in allusers:
        cur.execute('select qviewcount, qtags from sim_qa where aowneruserid='+str(u))
        qres = cur.fetchall()
        if qres==None or len(qres)==0:
            continue
        qres = [qr[0] for qr in qres if qr[0] is not None and is_tag_in(tag, qr[1])]
        if len(qres)==0:
            continue
        for q in qres:
            f.write("overall, "+str(q)+"\n")
            if u in experts:
                f.write("owl, "+str(q)+"\n")
            if u in sparrows:
                f.write("sparrow, "+str(q)+"\n")
    f.close()
    print 'data/ansd_pop_all.csv done!'
    
    f=open('data/ansd_diff_all.csv', 'w')
    for u in allusers:    
        cur.execute('select qduration, qtags from sim_qa where aowneruserid='+str(u))
        qres = cur.fetchall()
        if qres==None or len(qres)==0:
            continue
        qres = [qr[0] for qr in qres if qr[0] is not None and is_tag_in(tag, qr[1])]
        if len(qres)==0:
            continue
        for q in qres:
            f.write("overall, "+str(q)+"\n")
            if u in experts:
                f.write("owl, "+str(q)+"\n")
            if u in sparrows:
                f.write("sparrow, "+str(q)+"\n")
    f.close()      
    print 'data/ansd_diff_all.csv done!'
    return 0

