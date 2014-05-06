import psycopg2

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()
cur.execute('SELECT version()')
ver = cur.fetchone()

def create_index():
    cur.execute('''CREATE INDEX questions_acceptedanswer_index
                      ON questions
                      USING btree
                      (acceptedanswer);
                   CREATE INDEX answers_parentid_index
                      ON answers
                      USING btree
                      (parentid);''')
    con.commit()
def qdiff():
    cur.execute("ALTER TABLE questions ADD duration integer")
    con.commit()
    index = 0
    cur.execute("select questions.id, EXTRACT(epoch FROM answers.creationdate-questions.creationdate) from  answers, questions where questions.acceptedanswer=answers.id")
    results1 = cur.fetchall()
    for r in results1:
        index += 1
        if index%10000 == 0:
            print "the "+str(index/10000)+"th 10 thousand record"
            con.commit()
        qid = r[0]
        minutes = int(r[1])
        if qid==None:
            continue
        cur.execute("update questions set duration = "+str(minutes)+" where id = "+str(qid))
        #print "user id "+ str(uid) + " minutes "+str(minutes)
    con.commit()

if __name__ == '__main__':
    create_index()
    qdiff()
    cur.execute('''CREATE TABLE sim_qa AS
                   SELECT 
                      questions.id as qid,
                      questions.acceptedanswer as qacceptedanswer,
                      questions.creationdate as qcreationdate,
                      questions.score as qscore,
                      questions.viewcount as qviewcount,
                      questions.tags as qtags,
                      questions.owneruserid as qowneruserid,
                      questions.lasteditoruserid as qlasteditoruserid,
                      questions.duration as qduration,
                      answers.id as aid,
                      answers.parentid as aparentid,
                      answers.creationdate as acreationdate,
                      answers.score as ascore,
                      answers.viewcount as aviewcount,
                      answers.owneruserid as aowneruserid,
                      answers.lasteditoruserid as alasteditoruserid
                    FROM questions, answers
                    WHERE questions.id = answers.parentid;
                    
                    ALTER TABLE sim_qa
                      OWNER TO postgres;
                    
                    CREATE INDEX sim_qa_aowneruserid_index
                      ON sim_qa
                      USING btree
                      (aowneruserid);''')
    con.commit()