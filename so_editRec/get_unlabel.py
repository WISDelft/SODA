from util import *
import sys
import random

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()

'''qlabels = loadfile_flat('ed_type_label_extreme_'+'code')
i = 0
for qid in qlabels:
    if qlabels[qid]==0:
        continue
    cur.execute("select text, creationdate, userid from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=2")
    orign = cur.fetchall()[0]
    orign_text = orign[0]
        
    cur.execute("select text, creationdate, id, userid, addvote, comment from closedquestionhistory_edit where postid="+str(qid)+" and posthistorytypeid=5")
    edit_texts = cur.fetchall()
    edit_texts = sorted(edit_texts, key=lambda edit_texts : edit_texts[1])
        
    tot = nltk.word_tokenize(orign_text)
    toe = nltk.word_tokenize(edit_texts[-1][0])
    
    if 'code' in toe and 'code' not in tot:
        i += 1
print i  # 'attempt': 71/336; 'code': 235/612
sys.exit(1)'''

pos = loadfile_flat('extreme_set')
#random.shuffle(pos)
i = 0
qids = []
k = 0
for qid in pos:
    '''if pos[qid]==0:
        continue'''
    i += 1
    if i%1000 ==0:
        print 'the '+str(i)+'th extreme question'
    cur.execute("select text, creationdate, userid from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=2")
    orign = cur.fetchall()[0]
    orign_text = orign[0]
        
    cur.execute("select text, creationdate, id, userid, addvote, comment from closedquestionhistory_edit where postid="+str(qid)+" and posthistorytypeid=5")
    edit_texts = cur.fetchall()
    edit_texts = sorted(edit_texts, key=lambda edit_texts : edit_texts[1])
        
    tot = nltk.word_tokenize(orign_text)
    toe = nltk.word_tokenize(edit_texts[-1][0])
    
    if 'code' in toe and 'code' not in tot:
        k += 1
        #print '-----------------'
        #print qid
        qids.append(qid)
        #print orign_text
        #print edit_texts[-1][0]
    
print k
dumpfile(qids, 'augmented_code')
    