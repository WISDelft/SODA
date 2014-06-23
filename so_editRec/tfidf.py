import string
import os
import sys
import pickle
from util import *
import math
import re
import numpy

df = dict([])
porter = nltk.PorterStemmer()

def addup_df(qid, type):
    if type==0:
        cur.execute("select text, creationdate, userid from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=2")
        result = cur.fetchall()
        if result==None:
            return 0
        if len(result)==0:
            return 0
        orign = result[0]
        text = orign[0]
    else:
        cur.execute("select text, creationdate, userid from closedquestionhistory_edit where postid="+str(qid)+" and posthistorytypeid=5")
        edit_texts = cur.fetchall()
        texts = sorted(edit_texts, key=lambda edit_texts : edit_texts[0])
        text = texts[-1][0]
        
    t1 = get_text(text)
    t1 = nltk.sent_tokenize(t1)
    tot = [nltk.word_tokenize(sent) for sent in t1]
    tot = [item for sublist in tot for item in sublist] 
    tot = [w.lower() for w in tot]
    tot = [porter.stem(t) for t in tot]
    #print df
    for t in tot:
        if t in df:
            df[t] += 1
        else:
            df[t] = 1
    #print df
    return df
def get_text(text):
    p_code_line = re.compile('\ \ \ \ .+\n')
    code_lines = p_code_line.findall(text)
    r_text = text
    for cl in code_lines:
        r_text = string.replace(r_text, cl, '')
    
    p_link = re.compile('\(http://.+\)|\[http://.+\]|\<http://.+\>|\"http://.+\"|\n\ \ \[\d\]: http://.+|http://.+\ ')
    link_lines = p_link.findall(r_text)
    for ll in link_lines:
        r_text = string.replace(r_text, ll, '')
        
    return r_text

def gene_df(qids):
    allqids = qids
    inds = 0
    for qid in allqids:
        inds += 1
        #print inds
        if inds%1000==0:
            print '..[df] processing the '+str(inds)+"th question"
        df = addup_df(qid, 0)
    # following code clean df
    df1 = dict([])
    for d in df:
        df1[d] = df[d]
    for d in df:
        regx = '`\w+`|`\w+|\*\*\w+\*\*|\*\w+'
        form = re.compile(regx)
        newd = form.findall(d)
        if len(newd)==0:
            continue
        newd = re.sub('`|\*', '', newd[0])
        
        newd = porter.stem(newd)
        if newd in df1:
            df1[newd] += df1[d]
            df1.pop(d, None)
    df2 = dict([])
    for d in df1:
        if df1[d]>10 and df1[d]<len(qids):
            df2[d] = df1[d]
    #print len(df2)
    df = df2
    dumpfile(df2, 'df_conf')
    
def gene_tps(qids):
    tps = set()
    for qid in qids:
        cur.execute("select qtags from sim_qa where qid="+str(qid))
        tags = cur.fetchall()[0][0]
        if tags == None:
            continue
        tags = tags.split('|')
        for t in tags:
            tps.add(t)
    dumpfile(list(tps), 'tps_conf')