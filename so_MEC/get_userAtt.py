'''
Created on May 1, 2014

@author: jyang3
'''
import psycopg2
from util_mec import *

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()

def get_answerers():
    cur.execute('select id from users')
    ids = cur.fetchall()
    ids = [x[0] for x in ids]
    dumpfile(ids, "all_answerers")
    return ids
    
def get_answers_att(users, tag):
    answerers = set()
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
                    answerers.add(u)
    print 'nr answerers: '+str(len(answerers))
    print 'nr answers: '+str(len(answers_att))
    dumpfile(answerers, "answerers_"+tag)
    dumpfile(answers_att, "answers_att_"+tag)
    return answerers, answers_att

if __name__ == '__main__':
    get_answerers()
    