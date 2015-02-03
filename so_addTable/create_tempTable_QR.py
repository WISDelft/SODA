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

if __name__ == '__main__':
    cur.execute('''CREATE TABLE sim_ca AS
                    SELECT
                    comments.id as cid,
                    comments.postid as cpostid,
                    comments.score as cscore,
                    comments.creationdate as ccreationdate,
                    comments.userdisplayname as cuserdisplayname,
                    comments.userid as cuserid,
                    answers.id as aid,
                    answers.parentid as aparentid,
                    answers.creationdate as acreationdate,
                    answers.score as ascore,
                    answers.viewcount as aviewcount,
                    answers.owneruserid as aowneruserid,
                    answers.lasteditoruserid as alasteditoruserid
                    FROM comments, answers 
                    WHERE comments.postid = answers.id;
                    
                    ALTER TABLE sim_ca
                      OWNER TO postgres;
                    
                    CREATE INDEX sim_ca_cuserid_index
                      ON sim_ca
                      USING btree
                      (cuserid);
                    CREATE INDEX sim_ca_aowneruserid_index
                      ON sim_ca
                      USING btree
                      (aowneruserid);''')
    
    cur.execute('''CREATE TABLE sim_cq AS
                    SELECT
                    comments.id as cid,
                    comments.postid as cpostid,
                    comments.score as cscore,
                    comments.creationdate as ccreationdate,
                    comments.userdisplayname as cuserdisplayname,
                    comments.userid as cuserid,
                    questions.id as qid,
                    questions.acceptedanswer as qacceptedanswer,
                    questions.creationdate as qcreationdate,
                    questions.score as qscore,
                    questions.viewcount as qviewcount,
                    questions.tags as qtags,
                    questions.owneruserid as qowneruserid,
                    questions.lasteditoruserid as qlasteditoruserid,
                    questions.duration as qduration
                    FROM comments, questions 
                    WHERE comments.postid = questions.id;
                    
                    ALTER TABLE sim_cq
                      OWNER TO postgres;
                    
                    CREATE INDEX sim_cq_cuserid_index
                      ON sim_cq
                      USING btree
                      (cuserid);
                    CREATE INDEX sim_cq_qowneruserid_index
                      ON sim_cq
                      USING btree
                      (qowneruserid);
                    CREATE INDEX sim_cq_qid_index
                      ON sim_cq
                      USING btree
                      (qid);''')
    
    cur.execute('''CREATE TABLE sim_cqa AS
                    SELECT
                    sim_ca.cid as cid,
                    sim_ca.cpostid as cpostid,
                    sim_ca.cscore as cscore,
                    sim_ca.ccreationdate as ccreationdate,
                    sim_ca.cuserdisplayname as cuserdisplayname,
                    sim_ca.cuserid as cuserid,
                    sim_ca.aid as aid,
                    sim_ca.aparentid as aparentid,
                    sim_ca.acreationdate as acreationdate,
                    sim_ca.ascore as ascore,
                    sim_ca.aviewcount as aviewcount,
                    sim_ca.aowneruserid as aowneruserid,
                    sim_ca.alasteditoruserid as alasteditoruserid,
                    questions.id as qid,
                    questions.acceptedanswer as qacceptedanswer,
                    questions.creationdate as qcreationdate,
                    questions.score as qscore,
                    questions.viewcount as qviewcount,
                    questions.tags as qtags,
                    questions.owneruserid as qowneruserid,
                    questions.lasteditoruserid as qlasteditoruserid,
                    questions.duration as qduration
                    FROM sim_ca, questions 
                    WHERE sim_ca.aparentid = questions.id;
                    
                    ALTER TABLE sim_cqa
                      OWNER TO postgres;
                    
                    CREATE INDEX sim_cqa_cowneruserid_index
                      ON sim_cqa
                      USING btree
                      (cuserid);
                    CREATE INDEX sim_cqa_qowneruserid_index
                      ON sim_cqa
                      USING btree
                      (qowneruserid);
                    CREATE INDEX sim_cqa_cowneruserid_index
                      ON sim_cqa
                      USING btree
                      (cuserid);
                    CREATE INDEX sim_cqa_qid_index
                      ON sim_cqa
                      USING btree
                      (qid);''')

    con.commit()