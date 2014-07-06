import string
import sys
import psycopg2
import os
import pickle
import numpy as np
from util import *
from tfidf import *
import re
import nltk
from scipy import sparse
import gensim
from datetime import *  
import time  

bashCommand = "export PGHOST=localhost"
os.system(bashCommand)
con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()

meta_tag = {'c#': 0, 'asp.net': 0, '.net': 0, 'vb.net': 0, 'wcf': 0, 'android': 1, 'java': 1, 'eclipse': 1, 'javascript': 2, 'jquery': 2, 'html': 2, 'css': 2, 'php': 3, 'mysql': 3, 'arrays': 3, 'apache': 3, 'c': 4, 'c++': 4, 'windows': 4, 'qt': 4, 'oop': 5, 'image': 5, 'performance': 5, 'delphi': 5, 'iphone': 6, 'ios': 6, 'objective-c': 6, 'sql': 7, 'sql-server': 7, 'database': 7, 'python': 8, 'django': 8, 'list': 8, 'ruby': 9, 'ruby-on-rails': 9, 'regex': 10, 'string': 10, 'perl': 10, 'asp.net-mvc': 11, 'mvc': 11, 'flex': 12, 'flash': 12, 'actionscript': 12, 'git': 13, 'svn': 13}
porter = nltk.PorterStemmer()
'''====================================== syntactical and semantical features =============================='''
dis_bigram_schema = [('to', 'gener'), ('the', 'request'), ('of', 'object'), ('to', 'pars'), ('like', 'a'), ('use', 'an'), ('java', 'class'), ('the', 'function'), ('whi', 'is'), ('doe', 'the'), ('sourc', 'code'), ('the', 'array'), ('the', 'command'), ('becaus', 'the'), ('must', 'be'), ('the', 'two'), ("'s", 'a'), ('to', 'solv'), ("'ve", 'got'), ('the', 'result')]




def clean_df(odf):
    ndf = dict([])
    for d in odf:
        ndf[d] = odf[d]
    for d in odf:
        regx = '`\w+`|`\w+|\*\*\w+\*\*|\*\w+'
        form = re.compile(regx)
        newd = form.findall(d)
        if len(newd)==0:
            continue
        newd = re.sub('`|\*', '', newd[0])
        
        newd = porter.stem(newd)
        #print '----'
        #print d
        #print newd
        if newd in odf:
            #print 'old: '+d+', oldfre: '+str(df[d])
            ndf[newd] += odf[d]
            #print 'new: '+newd+', fre: '+str(df[newd])
            ndf.pop(d, None)
    return ndf

def get_dis_bigram(text, nr):
    t1 = text
    t1 = nltk.sent_tokenize(t1)
    tot = [nltk.word_tokenize(sent) for sent in t1]
    bitot = []
    for sent in tot:
        sent = [w.lower() for w in sent]
        sent = [porter.stem(t) for t in sent] 
        bitot.append(nltk.bigrams(sent) )   
    bitot = [item for sublist in bitot for item in sublist]   
    
    bi_this = dict([])
    for t in bitot:
        tt = t[1]
        if (t[0]=='I' or t[0]=='i') and t[1]=="'m":
            tt='am'
        if (t[0]=='I' or t[0]=='i') and t[1]=="'ve":
            tt='have'
        if t[0] =='public':
            continue
        tr = (t[0], tt)
        if tr in bi_this:
            bi_this[tr] += 1
        else:
            bi_this[tr] = 1
            
    vec = []
    for tr in dis_bigram_schema:
        if tr in bi_this:
            vec.append(float(bi_this[tr])/big_ALL2[tr])
        else:
            vec.append(0)
    return vec

def get_unigram(text, df):
    vec = []
    this_tf = dict([])
    t1 = nltk.sent_tokenize(text)
    tot = [nltk.word_tokenize(sent) for sent in t1]
    tot = [item for sublist in tot for item in sublist] 
    tot = [w.lower() for w in tot]
    tot = [porter.stem(t) for t in tot]
    for t in tot:
        if t in this_tf:
            this_tf[t] += 1
        else:
            this_tf[t] = 1
    this_tf = clean_df(this_tf)  
     
    for t in df:
        if t in this_tf:
            vec.append(float(this_tf[t])/df[t])
        else:
            vec.append(0)
        
    return vec
'''====================================== below is some basic features =============================='''

def remove_code(text):
    p_code_line = re.compile('\ \ \ \ .+\n')
    code_lines = p_code_line.findall(text)
    for cl in code_lines:
        text = string.replace(text, cl, '')
    return text

def match_list(text, symbol):
    list_len = 0
    regx = '\n'+symbol+' .+'
    p_list = re.compile(regx)
    list_lines = p_list.findall(text)
    text = string.replace(text, symbol, '')
    list_len = len(list_lines)
    return list_len, text

def match_em(text, symbol):
    em_len = 0
    regx = symbol+'.+'+symbol
    p_em = re.compile(regx)
    em_words = p_em.findall(text)
    text = string.replace(text, symbol, '')
    em_len = len(em_words)
    return em_len, text

def match_remove_link(text):
    link_len = 0
    p_link = re.compile('\(http://.+\)|\[http://.+\]|\<http://.+\>|\"http://.+\"|\n\ \ \[\d\]: http://.+|http://.+\ ')
    link_lines = p_link.findall(text)
    for ll in link_lines:
        text = string.replace(text, ll, '')
    link_len = len(link_lines)
    return link_len, text

'''====================================== below is the feature aggregation function =============================='''

def rm_feature_term(qfeatures, term, df):
    i = 0
    for d in df:
        if d==term:
            print d
            break
        i += 1
    
    return numpy.delete(qfeatures,(i), axis=1)

def add_feature4type(qid, qfeature):
    ##user past activities
    cur.execute("select qowneruserid from sim_qa where qid="+str(qid))
    u = cur.fetchone()
    if u==None or len(u)==0:
        qfeature.append(None)
        u = None
    elif u[0]==None:
        qfeature.append(None)
        u = None
    else:
        u=u[0]
        cur.execute("select count(distinct qid) from sim_qa where qowneruserid="+str(u))
        nrq = cur.fetchone()[0]
        cur.execute("select count(distinct aid) from sim_qa where qowneruserid="+str(u))
        nra = cur.fetchone()[0]
        qfeature.append(nrq+nra)
        
    '''#knowledge
    if u==None:
        qfeature.append(None)
    else:
        cur.execute("select EXTRACT(day FROM sim_qa.qcreationdate-users.creationdate) from sim_qa, users where sim_qa.qid="+str(qid)+" and users.id=sim_qa.qowneruserid and users.id="+str(u))
        result = cur.fetchone()
        if result==None or len(result)==0:       
            qfeautres.append(None)
        day = result[0]
        if day==None:
            qfeature.append(None)
        else:
            qfeature.append(day)'''
                
 
    return qfeature

    
def add_feature(qid, qfeature, tps):
    #topics
    cur.execute("select qtags from sim_qa where qid="+str(qid))
    tags = cur.fetchone()
    if tags == None or len(tags)==0:
        print '[warning] tags'
        tags = []
    elif tags[0] == None:
        print '[warning] tags'
        tags = []
    else:
        tags = tags[0].split('|')
    
    for t in tps:
        if t in tags:
            qfeature.append(1)
        else:
            qfeature.append(0)
    
    ##user past activities
    cur.execute("select qowneruserid from sim_qa where qid="+str(qid))
    u = cur.fetchone()
    if u==None or len(u)==0:
        qfeature.append(None)
        u = None
    elif u[0]==None:
        qfeature.append(None)
        u = None
    else:
        u=u[0]
        cur.execute("select count(distinct qid) from sim_qa where qowneruserid="+str(u))
        nrq = cur.fetchone()[0]
        cur.execute("select count(distinct aid) from sim_qa where qowneruserid="+str(u))
        nra = cur.fetchone()[0]
        qfeature.append(nrq+nra)
        
    #knowledge
    if u==None:
        qfeature.append(None)
    else:
        cur.execute("select EXTRACT(day FROM sim_qa.qcreationdate-users.creationdate) from sim_qa, users where sim_qa.qid="+str(qid)+" and users.id=sim_qa.qowneruserid and users.id="+str(u))
        result = cur.fetchone()
        if result==None or len(result)==0:       
            qfeautres.append(None)
        day = result[0]
        if day==None:
            qfeature.append(None)
        else:
            qfeature.append(day)
                
    #temporal
    cur.execute("select TO_CHAR(qcreationdate,'YYYY-MM-DD') from sim_qa where qid="+str(qid))
    result = cur.fetchone()
    if result==None or len(result)==0:
        qfeature.append(None)
    elif result[0]==None:
        qfeature.append(None)
    else:
        year, month, day = result[0].split('-')
        postdate = date(int(year), int(month), int(day))
        basedate = date(2008,07,31)
        timedelta = postdate-basedate
        qfeature.append(timedelta.days)
    
    return qfeature

def extract_one(qid, df):
    cur.execute("select text from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=2")
    result = cur.fetchall()
    if result == None or len(result)==0:
        return -1
    if result[0] == None:
        return -1
    orign_text = result[0][0]
    #print orign_text
    
    ''' orign_text -> code_free_text -> format_free_text: -*> -> em_free_text: _a_ **a** `a` -> tag_free_text <>a<> -> link_free_text http:// -> explan_text'''
    
    # code_free_text: code_len
    p_code = re.compile('\ \ \ \ ')
    code_len = len(p_code.findall(orign_text))
    code_free_text = remove_code(orign_text)
    
    # format_free_text -*> 1.
    list_len1, format_free_text = match_list(code_free_text, '-')
    list_len2, format_free_text = match_list(code_free_text, '\d\.')
    list_len3, format_free_text = match_list(format_free_text, '*')
    list_len4, format_free_text = match_list(format_free_text, '>')
    
    # em_free_text: _a_ **a** `a`
    em_len1, em_free_text = match_em(format_free_text, '_')
    em_len2, em_free_text = match_em(em_free_text, '\*\*')
    em_len3, em_free_text = match_em(em_free_text, '`')
    
    # tag_free_text, <></>, < />
    tag_free_text = em_free_text
    
    # link_free_text, http://
    link_len, link_free_text = match_remove_link(tag_free_text)
    
    # ask_len: what, how
    other_tokens = nltk.word_tokenize(code_free_text)
    ask_words = ['what', 'how', 'which']
    ask_len = 0
    for aw in ask_words:
        if aw in other_tokens:
            ask_len = 1
    
    # explan_len
    explain_len = len(other_tokens)
    
    sents = nltk.sent_tokenize(code_free_text)
    sent_len = len(sents)
    '''------ Basic feature set -------'''
    #qfeature = [code_len, list_len1+list_len2+list_len3, list_len4, em_len1+em_len2, em_len3, link_len, ask_len, explain_len, sent_len]
    '''------ Semantic feature set ---'''
    
    qfeature = []
    unigram = get_unigram(code_free_text, df)
    for u in unigram:
        qfeature.append(u)
    '''dis_bigrams = get_dis_bigram(code_free_text, sent_len)
    
    for d in dis_bigrams:
        qfeature.append(d)'''
    #sys.exit(1)
    return qfeature

def extract_one2(qid, df, type, ev):
    cur.execute("select text from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=2")
    result = cur.fetchall()
    if result == None or len(result)==0:
        return -1
    if result[0] == None:
        return -1
    orign_text = result[0][0]
    #print orign_text
    
    '''if ev!=0:
        cur.execute("select creationdate, text, id, userid, addvote, comment from closedquestionhistory_edit where postid="+str(qid)+" and posthistorytypeid=5")
        edit_texts = cur.fetchall()
        edit_texts = sorted(edit_texts, key=lambda edit_texts : edit_texts[0])
    orign_text = edit_texts[ev-1][1]'''
    ''' orign_text -> code_free_text -> format_free_text: -*> -> em_free_text: _a_ **a** `a` -> tag_free_text <>a<> -> link_free_text http:// -> explan_text'''
    
    # code_free_text: code_len
    p_code = re.compile('\ \ \ \ ')
    code_len = len(p_code.findall(orign_text))
    code_free_text = remove_code(orign_text)
    
    # format_free_text -*> 1.
    list_len1, format_free_text = match_list(code_free_text, '-')
    list_len2, format_free_text = match_list(code_free_text, '\d\.')
    list_len3, format_free_text = match_list(format_free_text, '*')
    list_len4, format_free_text = match_list(format_free_text, '>')
    
    # em_free_text: _a_ **a** `a`
    em_len1, em_free_text = match_em(format_free_text, '_')
    em_len2, em_free_text = match_em(em_free_text, '\*\*')
    em_len3, em_free_text = match_em(em_free_text, '`')
    
    # tag_free_text, <></>, < />
    tag_free_text = em_free_text
    
    # link_free_text, http://
    link_len, link_free_text = match_remove_link(tag_free_text)
    
    # ask_len: what, how
    other_tokens = nltk.word_tokenize(code_free_text)
    ask_words = ['what', 'how', 'which']
    ask_len = 0
    for aw in ask_words:
        if aw in other_tokens:
            ask_len = 1
    
    # explan_len
    explain_len = len(other_tokens)
    
    sents = nltk.sent_tokenize(code_free_text)
    sent_len = len(sents)
    '''------ Basic feature set -------'''
    #qfeature = [code_len, list_len1+list_len2+list_len3, list_len4, em_len1+em_len2, em_len3, link_len, ask_len, explain_len, sent_len]
    '''------ Semantic feature set ---'''
    
    qfeature = []
    unigram = get_unigram(code_free_text, df)
    for u in unigram:
        qfeature.append(u)
    # if we consider the bigrams
    '''dis_bigrams = get_dis_bigram(code_free_text, sent_len)
    
    for d in dis_bigrams:
        qfeature.append(d)'''
    # if we consider the edit version effect
    '''if ev!=0:
        qlabels = loadfile_flat('ed_type_qids_extreme_'+type+str(ev-1))
        if qid in qlabels:
            qfeature.append(qlabels[qid])
        else:
            qfeature.append(0)'''
    return qfeature
'''====================================== topic-wise =============================='''
def add_topic(qfeature, qid):
    cur.execute("select text from closedquestionhistory_orign where postid="+str(qid)+" and posthistorytypeid=3")
    fetched = cur.fetchone()

    if fetched == None:
        return False
    else:
        tags = fetched[0]
    #print tags
    if tags == None or not is_in_metatag(tags):
        for i in range(14):
            qfeature.append(0)
    else:
        inds = inds_in_metatag(tags)
        for i in range(14):
            if i in inds:
                qfeature.append(1)
            else:
                qfeature.append(0)
    #print qfeature
    return qfeature

def is_in_metatag(tags):
    result = False
    for mt in meta_tag:
        mt2 = mt
        if mt == 'c++':
            mt2 = 'c\+\+'
        regx = '<'+mt2+'\>'
        p_tag1 = re.compile(regx)
        if len(p_tag1.findall(tags)) > 0:
            result = True
            break
    return result

def inds_in_metatag(tags):
    inds = []
    result = False
    for mt in meta_tag:
        mt2 = mt
        if mt == 'c++':
            mt2 = 'c\+\+'
        regx = '<'+mt2+'\>'
        p_tag2 = re.compile(regx)
        if len(p_tag2.findall(tags)) > 0 and not(meta_tag[mt] in inds):
            inds.append(meta_tag[mt])
    return inds

def remove_feature(qfeatures, df, tps):
    (m,n) = qfeatures.shape
    print '-- before remove --'
    print m
    print n
    #print len(allu)
    qfeature_sl = []
    worddict = dict2list(df)
    effective_df = []
    effective_tps = []
    for j in range(n):
        if j%1000==0:
            print "processing the "+str(j)+"th column"
    
        if np.count_nonzero(qfeatures[:,j])>=10:
            this=list(qfeatures[:,j])
            qfeature_sl.append(this)
            if j<len(df):
                effective_df.append(worddict[j][0])
            if j>=len(df) and j<len(df)+len(tps):
                effective_tps.append(tps[j-len(df)])
            #worddict[allu[j]] = df2[allu[j]]
            #print qfeature_sl
    print np.array(qfeature_sl).shape
    dumpfile(effective_df, 'effective_df')
    dumpfile(effective_tps, 'effective_tps')
    #print len(worddict)
    print '-- end remove --'
    '''dictlist = dict2list(worddict)
    dictlist = sorted(dictlist, key=lambda dictlist : dictlist[1], reverse=True)
    for d in dictlist:
        dictionary.write(d[0]+','+str(d[1])+'\n')'''
    return np.array(qfeature_sl).T

def remove_feature_type(qfeatures, term=''):
    (m,n) = qfeatures.shape
    print '-- before remove --'
    print m
    print n
    #print len(allu)
    qfeature_sl = []
    terms = term.split('|')
    if term != '':
        df = loadfile_flat('df_type')
        i = 0
        for t in df:
            if t in terms:
                break
            i += 1
        if i!=len(df):
            qfeatures = np.delete(qfeatures, i, axis=1)
            n -= 1
    
    for j in range(n):
        if j%1000==0:
            print "processing the "+str(j)+"th column"
    
        if np.count_nonzero(qfeatures[:,j])>=10:
            this=list(qfeatures[:,j])
            qfeature_sl.append(this)

    print np.array(qfeature_sl).shape

    print '-- end remove --'
    '''dictlist = dict2list(worddict)
    dictlist = sorted(dictlist, key=lambda dictlist : dictlist[1], reverse=True)
    for d in dictlist:
        dictionary.write(d[0]+','+str(d[1])+'\n')'''
    return np.array(qfeature_sl).T