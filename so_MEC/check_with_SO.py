'''jieresult = open('temp_files/jieresult.csv', 'r')
jr = []
for j in jieresult:
    j=j[0:-1]
    jr.append(j)
jieresult.close()

jieresult2 = open('temp_files/JieResult2.csv', 'w')
l = len(jr)-1
i=l
while i>=0:
    print jr[i]
    jieresult2.write(jr[i])
    jieresult2.write('\n')
    i -= 1
    
jieresult2.close()'''

''' Their query

-- http://yangjiera.github.io/home/works/umap2014_experts.pdf

declare @tag_debate float = (
  select avg(1.0*AnswerCount)
  from Posts q
       join PostTags on q.Id = PostId
       join Tags t on TagId = t.Id
  where TagName = '##tag##'
  and q.AnswerCount >= 1
);

/* The paper reported 2.27+/-1.74 as the average number of answers to 
   C# questions. I tried using that number, but it didn't radically 
   change the results. */
--declare @tag_debate float = 2.27;

with answers as (
  select a.OwnerUserId UserId,
         1.0/rank() over (partition by q.Id order by a.Score desc) utility,
         1.0*q.AnswerCount debatableness,
         1.0/rank() over (partition by q.Id order by a.Score desc) *
         q.AnswerCount/@tag_debate EC
  from Posts q
       join Posts a on q.Id = a.ParentId
       join PostTags on q.Id = PostId
       join Tags t on TagId = t.Id
  where TagName = '##tag##' --and q.Id = 9929585
    and q.AnswerCount >= 1
),

user_MEC as (
  select UserId,
         sum(EC)/count(*) MEC,
         avg(utility) AU, 
         avg(debatableness) D,
         @tag_debate D_avg_t,
         count(*) Q_u_t
  from answers
  group by UserId  
)

select *
from user_MEC
where MEC >= 1
order by MEC desc
'''


'''import sys

f1 = open('temp_files/JieResults.csv', 'r')
f2 = open('temp_files/QueryResults.csv', 'r')

jieu = []
sou = []
i = 0
for r in f1:
    i += 1
    if i==1:
        continue
    a = r.split(',')
    u = int(float(a[0]))
    jieu.append(u)

i = 0
for r in f2:
    i += 1
    if i==1:
        continue
    a = r.split(',')
    u = int(float(a[0][1:-1]))
    sou.append(u)
    
print len(jieu)
print len(sou)

resultfile = open('temp_files/intersect_at_p.csv','w')
for i in range(len(sou)):
    jier = jieu[0:i+1]
    sor = sou[0:i+1]
    intersct = len(set(jier).intersection(set(sor)))
    resultfile.write(str(i+1)+','+str(intersct)+','+str(float(intersct)/(i+1))+'\n')
        
resultfile.close()'''


import sys

f1 = open('temp_files/JieResults.csv', 'r')
f2 = open('temp_files/SOResults.csv', 'r')

jieu = []
jieu_all = []
sou_all = []
sou = []
i = 0
for r in f1:
    i += 1
    if i==1:
        continue
    jieu_all.append(r)
    a = r.split(',')
    u = int(float(a[0]))
    jieu.append(u)

i = 0
for r in f2:
    i += 1
    if i==1:
        continue
    sou_all.append(r)
    a = r.split(',')
    u = int(float(a[0][1:-1]))
    sou.append(u)

resultfile = open('temp_files/SO_yes_Jie_no_15.csv','w')
jier = jieu[0:13595]
sor = sou
diff = set(sor).difference(set(jier))
print len(diff)
notinjie = 0
for u in diff:
    if u in jieu:
        r=jieu_all[jieu.index(u)]
        resultfile.write(r)
    else:
        notinjie += 1
print notinjie
'''resultfile = open('temp_files/SO_no_Jie_yes.csv','w')
jier = jieu[0:len(sou)+1]
sor = sou
diff = set(jier).difference(set(sor))
print len(diff)
notinjie = 0
for u in diff:
    if u in jieu:
        r=jieu_all[jier.index(u)]
        a = r.split(',')
        if float(a[1])>=1:
            print a[1]
            resultfile.write(r)
    else:
        notinjie += 1
print notinjie'''
resultfile.close()