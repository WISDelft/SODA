import string
import sys
import psycopg2
import os
import pickle
import numpy as np
import nltk
import re
from util import *
import time
import math
import json
'''big_E = dict([])
big_NE = dict([])
big_EA = dict([])'''
porter = nltk.PorterStemmer()
con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()
print "Connecting to DB: " + str(ver)

add_rela_votes = []
output = open('/Users/jyang3//Documents/workspace/SoEdit/java_stats/good_edit_test', 'w')
output2 = open('/Users/jyang3//Documents/workspace/SoEdit/java_stats/neg_edit_test', 'w')
jfile = open('/Users/jyang3//Documents/workspace/SoEdit/java_stats/jfile', 'w')
jcontent = []

def get_most_edit(qids, n):
    qlen = []
    most_edit_q = []
    for qid in qids:
        cur.execute("select text, creationdate, userid from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=2")
        orign = cur.fetchall()[0]
        orign_text = orign[0]
        
        cur.execute("select text, creationdate, id, userid, addvote, comment from closedquestionhistory_edit where postid="+str(qid)+" and posthistorytypeid=5")
        edit_texts = cur.fetchall()
        edit_texts = sorted(edit_texts, key=lambda edit_texts : edit_texts[1])
        
        tot = nltk.word_tokenize(orign_text)
        toe = nltk.word_tokenize(edit_texts[-1][0])
        
        #qlen.append([qid, float(len(toe))/len(tot)])
        qlen.append([qid, len(toe)-len(tot)])
    qlen = sorted(qlen, key=lambda qlen : qlen[1], reverse=True)
    for i in range(n):
        #print qlen[i]
        most_edit_q.append(qlen[i][0])
    return most_edit_q
def select_pos_cmt(qid, type):
    cur.execute("select text, creationdate, userid from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=2")
    orign = cur.fetchall()[0]
    orign_text = orign[0]
    
    cur.execute("select text, creationdate, id, userid, addvote, comment from closedquestionhistory_edit where postid="+str(qid)+" and posthistorytypeid=5")
    edit_texts = cur.fetchall()
    edit_texts = sorted(edit_texts, key=lambda edit_texts : edit_texts[1])
    
    cur.execute("select text, creationdate from comments where postid="+str(qid))
    cmts = cur.fetchall()
    
    t1 = get_text(orign_text)
    t2 = get_text(edit_texts[-1][0])
    tot = nltk.word_tokenize(t1)
    toe = nltk.word_tokenize(t2)
    tot = nltk.pos_tag(tot)
    toe = nltk.pos_tag(toe)
    
    nrc = 0
    flag = False
    for c in cmts:
        if c[1]<edit_texts[-1][1]:
            nrc += 1
            toc = nltk.word_tokenize(c[0])
            toc = nltk.pos_tag(toc)
            tsec1 = set(toc).intersection(set(tot))
            tsec2 = set(toc).intersection(set(toe))
            tdiff = tsec2.difference(tsec1)
            for a in tdiff:
                if a[1]=='NN' or a[1]=='NNP':
                    flag=True
                    break
            
            if flag:            
                print orign_text
                print '---------'
                print edit_texts[-1][0]
                print '---------'
                print c[0]
                print '---------'
                print tdiff
                print '=========================='
                break
                return True
    
'''================================================= Below is old filtering mechanism based on answering time ==============================================='''

def select_pos(qid):
    
    cur.execute("select text, creationdate, userid from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=2")
    result = cur.fetchall()
    if result==None:
        return False,-1
    if len(result)==0:
        return False,-1
    orign = result[0]
    orign_text = orign[0]
    cur.execute("select text, creationdate, userid from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=1")
    title = cur.fetchall()[0]
    title_text = title[0]
    
    cur.execute("select creationdate, text, id, userid, addvote, comment from closedquestionhistory_edit where postid="+str(qid)+" and posthistorytypeid=5")
    edit_texts = cur.fetchall()
    edit_texts = sorted(edit_texts, key=lambda edit_texts : edit_texts[0])
    

    '''if type == 0:
        flag, ind = good_edit(orign[1], edit_texts, qid)
    elif type == 1:
        flag, ind = get_answer(orign[1], edit_texts, qid, 0)'''


    ind = -1
    
    e_time = edit_texts[ind][0]-orign[1]
    if e_time.days>7 or e_time.seconds<600:
        return False, -1


    t1 = get_text(orign_text)
    t2 = get_text(edit_texts[ind][1])
    all_diff = math.fabs( len(nltk.word_tokenize(t1)) - len(nltk.word_tokenize(t2)) )
    if all_diff<5: #only code formatting
        return False, -1

    edit_dist = nltk.metrics.edit_distance(t1, t2)
    '''print edit_dist
    print code_line1
    print code_line_diff'''
    code_line1 = code_line(orign_text)
    code_line2 = code_line(edit_texts[ind][1])
    code_line_diff =  math.fabs( code_line1 - code_line2)
    if edit_dist<100: #only text difference
        if code_line1 != 0:
            if float(code_line_diff)/code_line1<0.2:
                return False, -1
        return False, -1




    #t1 = [w.lower() for w in t1]
    tot = nltk.sent_tokenize(t1)
    tot = [nltk.word_tokenize(sent) for sent in tot]
    
    #t2 = [w.lower() for w in t2]
    toe = nltk.sent_tokenize(t2)
    toe = [nltk.word_tokenize(sent) for sent in toe]
    

    '''sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sot = set(sent_detector.tokenize(get_text(orign_text)))
    soe = set(sent_detector.tokenize(get_text(edit_texts[ind][1])))
    print soe.difference(sot)'''
    '''tot_all = set(nltk.word_tokenize(orign_text))
    toe_all = set(nltk.word_tokenize(edit_texts[ind][1]))
    
    diff1 = toe_all.difference(tot_all)
    diff2 = tot_all.difference(toe_all)
    
    print diff1
    print diff2'''
    
    

    Solu_indic = ['answer', 'Answer', 'reference', 'Reference', 'duplication', 'Duplication', 'duplicate', 'Duplicate', '**Possible Duplicate:**','**possible duplicate**', 'related', 'Related'] #only new references
    for si in Solu_indic:
        if (si in toe) and not (si in tot):
            return False, -1
    
    Solu_sents = ['I finally got it to work', 'is what I was looking for', 'I have solved', 'I solved']
    for regxs in Solu_sents:
        Solu_sent = re.compile(regxs)
        if Solu_sent.findall(edit_texts[ind][1]):
            return False, -1
        
    flag2 = False
    #print tot
    tot_tag = [nltk.pos_tag(t) for t in tot]
    toe_tag = [nltk.pos_tag(t) for t in toe]
    #print tot_tag
    tot_tag = [item for sublist in tot_tag for item in sublist]
    toe_tag = [item for sublist in toe_tag for item in sublist]
    #print tot_tag
    cur.execute("select text, creationdate from comments where postid="+str(qid))
    cmts = cur.fetchall()
    for c in cmts:
        if c[1]<edit_texts[ind][0]:            
            toc = nltk.word_tokenize(c[0])
            toc = nltk.pos_tag(toc)
            tsec1 = set(toc).intersection(set(tot_tag))
            tsec2 = set(toc).intersection(set(toe_tag))
            tdiff = tsec2.difference(tsec1)
            for a in tdiff:
                if (a[1]=='NN' or a[1]=='NNP') and a[0]!='http' and a[0]!='I':
                    flag2=True
                    '''print tsec1
                    print tsec2
                    print tdiff
                    print '------' '''
                    break
    if not flag2:
        return False, -1
    #explore_motiv()
    '''output.write( '\n\n==================================================\n\n')
    output.write( "qid: "+str(qid)+", edit id: "+str(ind)+"\n")
    output.write("time past: "+str(edit_texts[ind][0]-orign[1])+"\n")
    output.write( "edit user self? "+str(edit_texts[ind][3]==orign[2])+"\n")
    output.write( "Edit comment: "+str(edit_texts[ind][5])+"\n")
    output.write( "Comments in between: "+get_comment_inbetween(edit_texts[ind][0], qid)+"\n")
    output.write( "\n Answers in between: "+get_answer_inbetween(edit_texts[ind][0], qid)+"\n\n")
    
    output.write( orign_text)
    output.write( '\n\n----------------------------------------------------\n\n')
    output.write( edit_texts[ind][1])
    output.write( '\n\n----------------------------------------------------\n\n')
    
    d = dict([])
    d['qid']=qid
    d['title']=title_text
    d['body']=[]
    
    version0 = dict([])
    version0['eid'] = 0
    version0['ebody'] = orign_text
    d['body'].append(version0)
    version1 = dict([])
    version1['eid'] = 1
    version1['ebody'] = edit_texts[ind][1]
    d['body'].append(version1)
    
    jcontent.append(d)
    json.dump(jcontent, jfile)
    sys.exit(1)'''
    return True, len(tot)

def select_neg(qid, type):
    if type == 0:
        cur.execute("select id, votetypeid, creationdate from votes where postid="+str(qid)+" and (votetypeid=2 or votetypeid=3)")
        vote_versions = cur.fetchall()
        if vote_versions==None:
            print "..warning, no vote found"
            return False,-1
        av = 0
        for vv in vote_versions:
            if vv[1]==2:
                av += 1
            elif vv[1]==3:
                av -= 1
        if av>=5:
            return True,-1
        else:
            return False,-1
    elif type == 1:
        cur.execute("select text, creationdate from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=2")
        orign = cur.fetchall()[0]
        orign_time = orign[1]

        '''cur.execute("select acreationdate, aid from sim_qa where qid="+str(qid)+" and aid is not null")
        acreationdate = cur.fetchall()
        if acreationdate == None:    #1
            return False,-1
        #print acreationdate
        acreationdate = sorted(acreationdate, key=lambda acreationdate : acreationdate[0])
        good_i = -1
        for i in range(len(acreationdate)):
            aid = acreationdate[i][1]
            cur.execute("select id, votetypeid, creationdate from votes where postid="+str(aid)+" and (votetypeid=2 or votetypeid=3)")
            vote_versions = cur.fetchall()
            av = 0
            for vv in vote_versions:
                if vv[1]==2:
                    av += 1
                elif vv[1]==3:
                    av -= 1
            if av >= 5:
                good_i = i
                break
        if good_i == -1:    #2
            return False,-1
        earlist_goodanstime = acreationdate[good_i][0]
        
        time_elasped = earlist_goodanstime - orign_time '''
        
        
        '''cur.execute("select U.reputation from closedquestionhistory_orign as C, users as U where C.postid="+str(qid)+" and C.posthistorytypeid=2 and C.userid=U.id")
        result = cur.fetchone()
        if result!=None and result[0]!=None:   
            if result[0]>5000: '''
                #if time_elasped.seconds > 600:
            
        orign_text = orign[0]
        t1 = get_text(orign_text)
    
        #print t1
        #t1 = [w.lower() for w in t1]
        too = nltk.sent_tokenize(t1)
        toow = [nltk.word_tokenize(sent) for sent in too]
        
        '''bitoo = nltk.bigrams(too)
        for t in bitoo:
            if t in big_NE:
                big_NE[t] += 1
            else:
                big_NE[t] = 1'''
        #print "[neg] length of text: "+str(len(too))
        #print '==========================================================='
        #print orign_text
        #print '-----------'
        
        if len(too)>3:
            cur.execute("select text, creationdate from comments where postid="+str(qid))
            cmts = cur.fetchall()
            if len(cmts)>0: 
                for c in cmts:
                    output2.write(c[0])
                output2.write('\n---------------------------\n')
                output2.write(orign_text)
                output2.write('\n==========================================\n')
                return True, len(toow)
            else: 
                return False, -1
        else:
            return False, -1
            '''else:
                return False, -1'''
        #return False, -1

def get_answer(orign_time, edit_texts, qid, explore=0):
    ind = -1
    
    cur.execute("select acreationdate, aid from sim_qa where qid="+str(qid)+" and aid is not null")
    acreationdate = cur.fetchall()
    if acreationdate == None:    #1
        return False, ind
    #print acreationdate
    acreationdate = sorted(acreationdate, key=lambda acreationdate : acreationdate[0])
    good_i = -1
    av_pre = 0
    nr_ans_pre = 0
    av_max = []
    for i in range(len(acreationdate)):
        aid = acreationdate[i][1]
        cur.execute("select id, votetypeid, creationdate from votes where postid="+str(aid)+" and (votetypeid=2 or votetypeid=3)")
        vote_versions = cur.fetchall()
        av = 0
        for vv in vote_versions:
            if vv[1]==2:
                av += 1
            elif vv[1]==3:
                av -= 1
        if av >= 5:
            good_i = i
            break
        av_pre += av
        nr_ans_pre += 1
        av_max.append(av)
    if good_i == -1:    #2
        return False, ind

    #print edit_texts
    earlist_goodanstime = acreationdate[good_i][0]
    #print earlist_goodanstime
    if earlist_goodanstime < edit_texts[0][0]:
        return False, ind   #3

    time_elasped = earlist_goodanstime - orign_time
    #if explore == 0 and int(time_elasped.total_seconds()) < 1800:     #4
        #return False, ind
    if  len(av_max)>0 and max(av_max)>3:
        return False, ind
    if nr_ans_pre!=0 and float(av_pre)/nr_ans_pre > 2:
        return False, ind
    
    for et in edit_texts:
        
        ind += 1
        last_i = len(edit_texts)-1
        if et==edit_texts[last_i]:
            return True, ind
        
        if earlist_goodanstime < et[0]:
            return True, ind-1

    return False, ind

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

def code_line(text):
    p_code_line = re.compile('\ \ \ \ .+\n')
    code_lines = p_code_line.findall(text)
    return len(code_lines)

def get_answer_inbetween(time, qid):
    cur.execute("select body, creationdate from answers where parentid="+str(qid))
    anses = cur.fetchall()
    text = ''
    for a in anses:
        if a[1]<time:
            text += 'date: '+str(a[1])+'; text: '+a[0]+'\n'
    return text

def get_comment_inbetween(time, qid):
    cur.execute("select text, creationdate from comments where postid="+str(qid))
    cmts = cur.fetchall()
    text = ''
    for c in cmts:
        if c[1]<time:
            text += 'date: '+str(c[1])+'; text: '+c[0]+'\n'
    return text
"""================================== ===========================================================================
   ================================== following are deprecated codes=============================================
   ================================== ===========================================================================""" 



def good_edit(orign_time, edit_texts, qid):
    t_start = orign_time
    t_end = edit_texts[0][0]
    time_elasped = t_end-t_start
    #if time_elasped.seconds < 2*3600:
    #return False, -1
    
    cur.execute("select id, votetypeid, creationdate from votes where postid="+str(qid)+" and (votetypeid=2 or votetypeid=3)")
    vote_versions = cur.fetchall()
    if vote_versions==None or len(vote_versions)==0:
        #print "..warning, no vote found"
        return False, -1
    vote_versions = sorted(vote_versions, key=lambda vote_versions : vote_versions[2])
    av = 0
    for vv in vote_versions:
        #print vv[2]
        if vv[2]>=t_start and vv[2]<t_end:
            if vv[1]==2:
                av += 1
            elif vv[1]==3:
                av -= 1
        elif vv[2]>=t_end:
            break
    
    #print vote_versions
    #print av
    #print orign_time
    #print t_end
    
    normalize_factor = float(av*60)/int(time_elasped.total_seconds())

    #print normalize_factor

    ind = -1
    #print edit_texts
    for et in edit_texts:
        ind += 1
        t_start = t_end
        if et != edit_texts[-1]:
            next_et = edit_texts.index(et)+1
            t_end = edit_texts[next_et][0]
            time_elasped = t_end-t_start
        else:
            #print t_start
            this_i = len(vote_versions)-1
            v_end = vote_versions[this_i]
            while this_i>=0:
                #print this_i
                v_end = vote_versions[this_i]
                time_elasped = v_end[2] - t_start
                if time_elasped.days>7:
                    this_i = this_i - 1
                else:
                    break
            if v_end[2]<t_start or time_elasped.days>7:
                return False, ind
            #print time_elasped
        #print time_elasped
        if et[4] == None:
            continue
        add_rela_vote = et[4] - (time_elasped.total_seconds()/60)*normalize_factor
        #print 'absolute add vote: '+ str(et[4])+ ' relative add vote: '+str(add_rela_vote)
        if add_rela_vote>=3:
            ettt = et[0]-orign_time
            if int(ettt.total_seconds())/(3600*24) > 7:
                return False, ind
            return True, ind
        add_rela_votes.append(add_rela_vote)
        #print add_rela_votes
    return False, ind

def spell_error(text1, text2):
    p_em=re.compile('_\w+_|\*\*\w+\*\*|`\w+`')
    #tokens_orign_alpha = p.findall(text1)
    #tokens_edit_alpha = p.findall(text2)
    
    tokens_orign = nltk.word_tokenize(text1)
    tokens_edit = nltk.word_tokenize(text2)
    
    if nltk.metrics.edit_distance(tokens_orign, tokens_edit) == 0: #only whitespace difference
        return False
    
    
    for to in tokens_orign:  #exists special character difference 1
        if '_'+to+'_' in tokens_edit:
            return False
        if '**'+to+'**' in tokens_edit:
            return False
        if '`'+to+'`' in tokens_edit:
            return False
        if '<'+to+'>' in tokens_edit:
            return False

    if not('>' in tokens_orign) and ('>' in tokens_edit):
        return False

    if not('*' in tokens_orign) and ('*' in tokens_edit):
        return False

    if not('-' in tokens_orign) and ('-' in tokens_edit):
        return False

    #1. 2. >=4

    for te in tokens_edit:  #exists special character difference 2
        if '_'+te+'_' in tokens_orign:
            return False
        if '**'+te+'**' in tokens_orign:
            return False
        if '`'+te+'`' in tokens_orign:
            return False
        if '<'+te+'>' in tokens_orign:
            return False

    if ('>' in tokens_orign) and not('>' in tokens_edit):
        return False

    if ('*' in tokens_orign) and not('*' in tokens_edit):
        return False

    if ('-' in tokens_orign) and not('-' in tokens_edit):
        return False


    a1 = p_em.findall(text1)  #exists special character difference 3
    a2 = p_em.findall(text2)
    if nltk.metrics.edit_distance(a1,a2)==2:
        return False

    return True #spelling problem

#<pre>, ->



