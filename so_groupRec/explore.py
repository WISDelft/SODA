
from util_groupRec import *

con = None
con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj')
cur = con.cursor()

if __name__ == "__main__":
    qids = loadfile('allqids')
    diffQ = []
    