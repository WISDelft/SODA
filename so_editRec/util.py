import string
import sys
import psycopg2
import os
import pickle
import numpy as np
import re
import nltk
from bisect import bisect_left
bashCommand = "export PGHOST=localhost"
os.system(bashCommand)

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()
def bi_contains(lst, item):
    """ efficient `item in lst` for sorted lists """
    # if item is larger than the last its not in the list, but the bisect would
    # find `len(lst)` as the index to insert, so check that first. Else, if the
    # item is in the list then it has to be at index bisect_left(lst, item)
    return (item <= lst[-1]) and (lst[bisect_left(lst, item)] == item)

def dict2list(d):
    ds = []
    for dd in d:
        ds.append([dd, d[dd]])
    return ds
def loadfile(fname):
    f = open("temp_files/"+fname+".pik")
    data = pickle.load(f)
    f.close()
    data = polish(data)
    return data

def loadfile_flat(fname):
    f = open("temp_files/"+fname+".pik")
    data = pickle.load(f)
    f.close()
    return data

def loadfile_all(fname, vec):
    data = []
    for i in vec:
        f = open("temp_files/"+fname+str(i)+".pik")
        data_tmp = pickle.load(f)
        f.close()
        for dt in data_tmp:
            data.append(dt)
    return data

def dumpfile(data, fname):
    f = open("temp_files/"+fname+".pik", 'w')
    pickle.dump(data, f)
    f.close()

def polish(data):
    data2 = []
    for d in data:
        data2.append(d[0])
    return data2

def get_random_ones(vector, num, tag=''):
    v2 = set([])
    i = 0
    while i<num:
        ind = int(np.random.rand()*len(vector))
        elem = vector[ind]
        if i%1000==0:
            print '..[select] processing the '+str(i)+"th question"
        if not(elem in v2): #and is_tag_in(tag, elem):# and good_qst(elem):
            v2.add(elem)
            i += 1

    return list(v2)

def have_vote(qid):
    cur.execute("select addvote from closedquestionhistory_edit where postid="+str(qid)+" and posthistorytypeid=5 and addvote is not null")
    edit_texts = cur.fetchall()
    if edit_texts == None:
        return True
    else:
        return False

def is_tag_in(tag, qid):
    cur.execute("select text from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=3")
    fetched = cur.fetchone()
    #print tags
    
    if fetched == None:
        return False
    else:
        tags = fetched[0]
    if tags == None:
        return False

    p_tag = re.compile('<'+tag+'>')
    if len(p_tag.findall(tags)) > 0:
        return True
    
    return False

def good_qst(qid):
    cur.execute("select qscore from sim_qa where qid="+str(qid))
    result = cur.fetchone()[0]
    if result>=10:
        return True
    else:
        return False



def output_samples(qids):
    i = 0
    
    '''while i<num:
        randx = int(np.random.rand()*len(qids))
        qid = qids[randx]
        if qlabels[qid] == 1 and not (qid in ids) and is_tag_in('java', qid):
            i += 1
            ids.append(qid)'''
    
    output = open("/Users/jyang3//Documents/workspace/SoEdit/java_stats/sample_edit", 'w')
    for id in qids:
        cur.execute("select text from closedquestionhistory_orign where postid="+str(id)+" and posthistorytypeid=2")
        orign_text = cur.fetchall()[0][0]
        cur.execute("select creationdate, text, comment from closedquestionhistory_edit where postid="+str(id)+" and posthistorytypeid=5")
        edit_texts = cur.fetchall()
        
        '''if nltk.metrics.edit_distance(orign_text, edit_texts[-1][1])<20:
            continue'''
        
        edit_texts = sorted(edit_texts, key=lambda edit_texts : edit_texts[0])
        output.write(orign_text)
        output.write('\n-------Edited version(s)----------\n')
        for et in edit_texts:
            output.write(et[1])
            #output.write("\n!!!Comment: ")
            #output.write(et[2])
            if len(edit_texts)>1 and et!=edit_texts[-1]:
                output.write('\n-------Later Edit----------\n')
        output.write('\n====================================\n')
    output.close
    
    
def check_date(id, date):
    cur.execute("select qid from sim_qa where qid="+str(id)+" and TO_CHAR(qcreationdate,'YYYY-MM-DD') <'" + date +"'")
    result = cur.fetchone()
    if result == None:
        return False
    else:
        return True