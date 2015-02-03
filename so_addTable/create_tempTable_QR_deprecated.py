import psycopg2

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='wistudelft')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()

MAXINQUERY = 10000

def check_date(aid, date):
    cur.execute("select aid from sim_qa where aid="+str(id)+" and TO_CHAR(acreationdate,'YYYY-MM-DD') <'" + date +"'")
    result = cur.fetchone()
    if result == None:
        return False
    else:
        return True

def get_answer_tag():
    at_NoAnswer = dict([])
    at_NoQuestion = dict([])
    at_NoVote = dict([])
    cur.execute('''Select aowneruserid, qowneruserid, qtags, ascore, aid, from sim_qa''')
    ats = cur.fetchall()
    for at in ats:
        aowneruserid = at[0]
        
        ascore = at[3]
        qowneruserid = at[1]
        qtags = []
        for t in at[2].split('|'): 
            qtags.append(t)
        if len(qtags)!= 0:
            for t in qtags:
                if (aowneruserid, t) in at_NoAnswer:
                    at_NoAnswer[(aowneruserid, t)] += 1
                    at_NoVote[(aowneruserid, t)] += ascore
                else:
                    at_NoAnswer[(aowneruserid, t)] = 1
                    at_NoVote[(aowneruserid, t)] = ascore
                    
                if (qowneruserid, t) in at_NoQuestion:
                    at_NoQuestion[(qowneruserid, t)] += 1
                else:
                    at_NoQuestion[(qowneruserid, t)] = 1
            
    return at_NoAnswer, at_NoQuestion, at_NoVote



if __name__ == '__main__':
    #cur.execute('''CREATE TABLE qr_UT_matrix (uid INT, tag TEXT , NoAnswers  INT, NoQuestions  INT, NoVotes INT, CONSTRAINT qr_UT_key PRIMARY KEY (uid,tag));''')
    #con.commit()
    
    at_NoAnswer, at_NoQuestion, at_NoVotes = get_answer_tag()
    #print at_NoAnswer
    i = 0
    for ut in at_NoAnswer:
        if ut[0] == None:
            continue
        i += 1
        nv = at_NoVotes[(ut[0],ut[1])]
        if (ut[0],ut[1]) in at_NoQuestion:
            nq = at_NoQuestion[(ut[0],ut[1])]
        else:
            nq = 0
        cur.execute("INSERT INTO qr_UT_matrix (uid, tag, noanswers, noquestions, novotes) VALUES (%s, %s, %s, %s, %s)", (str(ut[0]),ut[1],str(at_NoAnswer[(ut[0],ut[1])]),str(nq),str(nv)))
        
        if i%MAXINQUERY == 0:
            print '... the '+str(i/MAXINQUERY) +'th batch inserted.'
            con.commit()
            
    i = 0
    for ut in at_NoQuestion:
        if ut[0] == None:
            continue
        if (ut[0],ut[1]) in at_NoAnswer:
            continue
        i += 1

        cur.execute("INSERT INTO qr_UT_matrix (uid, tag, noanswers, noquestions, novotes) VALUES (%s, %s, %s, %s, %s)", (str(ut[0]),ut[1],str(0),str(at_NoQuestion[(ut[0],ut[1])]),str(0)))

        if i%MAXINQUERY == 0:
            print '... the '+str(i/MAXINQUERY) +'th batch inserted.'
            con.commit()
    con.commit()
    print 'table constructed!!'
    #cur.execute("CREATE INDEX qr_UT_matrix_ut ON qr_UT_matrix (uid,tag);")
    #con.commit()