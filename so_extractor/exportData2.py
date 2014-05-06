'''
    This should be the first script to work with: it extract raw stackoverflow xml data into postgres database.

 The data is put in 'raw_data/', however due to disk storage limit, the original xml files are removed. To re-import, you need to put xml file under that directory, then run this script.

 Of course, you need to have postgres, python, related packages such as easy_install (for lxml), pip (for psycogp2) installed to make this script run successfully.
 
 The input parameter is either 0 or 1, where
    0: create a new table.
    1: do not create a new table. This can be used, for instance, when there are multiple files integrated to create a new table: all single files should be with parameter 1 except the first file.
'''

from lxml import etree
import psycopg2
import sys
import os
from HTMLParser import HTMLParser
from happyfuntokenizing import *
import re
import json
import string
import time
from functools import wraps

'''API_KEY = "d4jupjtwj98u9ntbh4umvd69"
calais = Calais(API_KEY, submitter="aleboz-test")'''
happyemoticons = [":-)", ":)", ":o)", ":]", ":3", ":c)", ":>", "=]", "8)", "=)", ":}", ":^)", ":-D", ":D", "8-D", "8D", "x-D", "xD", "X-D", "XD", "=-D", "=D","=-3", "=3", "B^D", ":-)", ";-)", ";)", "*-)", "*)", ";-]", ";]", ";D", ";^)", ":-,"]
sademoticons = [">:[", ":-(", ":(", ":-c", ":c", ":-<", ":<", ":-[", ":[", ":{", ":-||", ":@", ">:(", "'-(", ":'(", ">:O", ":-O", ":O"]	
codetext = ["<code>"]

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)


def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def fast_iterUser(context, cur,con):
	numbatch = 0
	counter = 0
	MAXINQUERY = 10000
	start_time = time.time()
	print "... START TIME at:" + str(start_time)
	
	query = "INSERT INTO users (id, reputation, creationdate, displayname, lastaccessdate, location, aboutme, views, upvotes, downvotes, emailhash, lenText, hemo, semo, upperc, punctu, code, hasurl, uurls, numurls, numgooglecode, numgithub, numtwitter, numlinkedin, numgoogleplus, numfacebook) VALUES (%s, %s, %s,%s, %s, %s, %s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s, %s,%s, %s)"
	

	urlsreg = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
	
	tok = Tokenizer(preserve_case=False)

	for event, elem in context:
		counter+=1

		stripped = None
		lenText = 0
		hemo = 0
		semo = 0
		upper = 0
		punctu = 0
		code = False
		hasURL = False
		uurls = None
		numurls = 0
		numGoogleCode = 0
		numGitHub = 0
		numTwitter = 0
		numLinkedIn = 0
		numGooglePlus = 0
		numFacebook = 0

		try:
			if(elem.get("AboutMe")):
				stripped = elem.get("AboutMe").encode('utf-8','ignore')
				lenText = len(stripped)	
				tokenized = tok.tokenize(stripped)
				hemo = sum(p in happyemoticons for p in tokenized)
				semo =  sum(p in sademoticons for p in tokenized)
				upper =  float(sum(x.isupper() for x in stripped)) / float(len(stripped)) * 100
				punctu =  sum(o in string.punctuation for o in stripped)
				code = True if sum(o in codetext for o in tokenized) > 0 else False
				result = re.findall(urlsreg, stripped)
				if(result):					
					uurls = ""
					for u in result:
						uurls = str(uurls) + str(u) + "|"
						if "code.google" in u:
							numGoogleCode += 1
						if "plus.google" in u:
							numGooglePlus += 1

						if "twitter" in u:
							numTwitter += 1

						if "github" in u:
							numGitHub += 1
						if "linkedin" in u:
							numLinkedIn += 1

						if "facebook" in u:
							numFacebook += 1

					if(len(uurls) > 1):						
						uurls = uurls[:-1]
				
					numurls = len(result)
					if(numurls > 0):
						hasURL = True

				del result
				del tokenized		


		except UnicodeDecodeError, e:
			print 'Error %s' % e    
		

		user = ( elem.get("Id"), #"Id": 
       			elem.get("Reputation"), #"Reputation": 
       			elem.get("CreationDate"), #"CreationDate": 
       			elem.get("DisplayName"),	#"DisplayName": 
       			elem.get("LastAccessDate"), #"LastAccessDate": 
       			elem.get("Location"), #"Location": 
       			stripped, #"AboutMe": 
       			elem.get("Views"), #"Views": 
				elem.get("UpVotes"), #"UpVotes": 
				elem.get("DownVotes"), #"DownVotes": 
				elem.get("EmailHash"), #"EmailHash": 
				lenText,
				hemo,
				semo,
				upper,
				punctu,
				code,
				hasURL,
				uurls,
				numurls,
				numGoogleCode,
				numGitHub,
				numTwitter,
				numLinkedIn,
				numGooglePlus,
				numFacebook
		)
		
		cur.execute(query,user)
		
		del user

		if(counter == MAXINQUERY or elem.getnext() == False):
			numbatch+=1
			counter = 0
			#print counter
			print "... commiting batch number " + str(numbatch) + ". Elapsed time: " + str(time.time() - start_time)
			con.commit();	
			
		
		elem.clear()
		while elem.getprevious() is not None:
			del elem.getparent()[0]
		#print user["Id"]

def fast_iterQuestions(context, cur,con):
	numbatch = 0
	counter = 0
	MAXINQUERY = 10000
	#urls = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
	#result = urls.match(string)
	start_time = time.time()
	
	print "... START TIME at:" + str(start_time)

	tok = Tokenizer(preserve_case=False)
	
	query = "INSERT INTO questions (id, acceptedanswer, creationdate, closeddate, isclosed, score, viewcount, body, owneruserid, ownerdisplayname, title, lasteditoruserid, lasteditordisplayname, lasteditdate, lastactivitydate, tags, numbertags, answercount, commentcount, favoritecount, bodylen, bhemo, bsemo, bupper, bpunctu,bcode, titlelen, themo, tsemo, tupper, tpunctu, tagsintitle, tagsinbody,tcode) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

	for event, elem in context:
		
		if(elem.get("PostTypeId") != "1"):
			elem.clear()
			continue

		counter+=1

		stripped = ""
		tags= ""
		tagnumber = 0
		
		bodylen= 0
		bhemo= 0
		bsemo= 0 
		bupper= 0 
		bpunctu = 0
		bcode = False
		
		titlelen= 0
		themo= 0
		tsemo= 0
		tupper= 0
		tpunctu = 0
		tcode = False
		 
		tagsintitle = 0
		tagsinbody = 0

		try:
			if(elem.get("Body")):
				stripped = elem.get("Body").encode('utf-8','ignore')
				bodylen = len(stripped)					
				tokenized = tok.tokenize(stripped)
        		bhemo = sum(p in happyemoticons for p in tokenized)
        		bsemo =  sum(p in sademoticons for p in tokenized)
        		bupper =  float(sum(x.isupper() for x in stripped)) / float(len(stripped)) * 100
        		bpunctu =  sum(o in string.punctuation for o in stripped)
        		bcode = True if sum(o in codetext for o in tokenized) > 0 else False
        		del tokenized
				# res = emoticonFinder(stripped)
				# if res != "NA":
			if(elem.get("Title")):
				tstripped = elem.get("Title").encode('utf-8','ignore')
				titlelen = len(elem.get("Title").encode('utf-8','ignore'))
				tokenized = tok.tokenize(tstripped)
        		themo = sum(p in happyemoticons for p in tokenized)
        		tsemo =  sum(p in sademoticons for p in tokenized)
        		tupper =  float(sum(x.isupper() for x in tstripped)) / float(len(tstripped)) * 100
        		tpunctu =  sum(o in string.punctuation for o in tstripped)
        		tcode = True if sum(o in codetext for o in tokenized) > 0 else False
        		del tokenized

		except UnicodeDecodeError, e:
			print 'Error %s' % e    
		

		try:
			if(elem.get("Tags")):
				#print elem.get("Tags").encode('utf-8','ignore')
				tags = re.sub('[<]', '', elem.get("Tags").encode('utf-8','ignore'))
				tags = re.sub('[>]', '|', tags)[:-1]
				tagnumber = len(tags.split("|"))
				#tags = strip_tags(elem.get("Tags").encode('utf-8','ignore'))
				for t in tags.split("|"):					
					if t in elem.get("Title"):
						tagsintitle += elem.get("Title").count(t);
					if t in elem.get("Body"):
						tagsinbody += elem.get("Body").count(t)

			#print tagnumber,tagsInTitle, tagsInBody

		except UnicodeDecodeError, e:
			print 'Error %s' % e    

		question = ( elem.get("Id"), #"Id": 
					elem.get("AcceptedAnswerId") if elem.get("AcceptedAnswerId") else None, #"acceptedanswer": 
					elem.get("CreationDate"), #"CreationDate": 
					elem.get("ClosedDate") if elem.get("ClosedDate") else None, #"ClosedDate": 					
					True if elem.get("ClosedDate") else False, #"isclosed": 					
					elem.get("Score") if elem.get("Score") else None, #"score": 
					elem.get("ViewCount") if elem.get("ViewCount") else None, #"ViewCount": 					
					stripped,
					elem.get("OwnerUserId") if elem.get("OwnerUserId") else None, #"OwnerUserId":
					elem.get("OwnerDisplayName") if elem.get("OwnerDisplayName") else None, #"OwnerDisplayName"
					elem.get("Title"), #"Location": 
					elem.get("LastEditorUserId"), #"LastEditorUserId": 
					elem.get("LastEditorDisplayName") if elem.get("LastEditorDisplayName") else None, #"LastEditorDisplayName"					
					elem.get("LastEditDate"), #"LastEditDate": 
					elem.get("LastActivityDate"), #"LastActivityDate": 
       				tags,
       				tagnumber,
					elem.get("AnswerCount") if elem.get("AnswerCount") else None, #"AnswerCount": 
					elem.get("CommentCount") if elem.get("CommentCount") else None, #"CommentCount": 
					elem.get("FavoriteCount") if elem.get("FavoriteCount") else None, #"AnswerCount": 					
					bodylen, 
					bhemo, 
					bsemo, 
					bupper, 
					bpunctu, 
					bcode,
					titlelen, 
					themo, 
					tsemo, 
					tupper, 
					tpunctu, 
					tagsintitle, 
					tagsinbody,
					tcode
		)

		#print question
		
		cur.execute(query,question)

		del question

		if(counter == MAXINQUERY or elem.getnext() == False):
			numbatch+=1
			counter = 0
			#print counter
			print "... commiting batch number " + str(numbatch) + ". Elapsed time: " + str(time.time() - start_time)
			con.commit()
			#return
			
		
		elem.clear()
		while elem.getprevious() is not None:
			del elem.getparent()[0]
		#print user["Id"]

def fast_iterAnswers(context, cur,con):
	numbatch = 0
	counter = 0
	MAXINQUERY = 10000
	#urls = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
	#result = urls.match(string)
	start_time = time.time()
	
	print "... START TIME at:" + str(start_time)

	tok = Tokenizer(preserve_case=False)
	
	query = "INSERT INTO answers (id, parentId, creationdate, closeddate, isclosed, score, viewcount, body, owneruserid, ownerdisplayname, title, lasteditoruserid, lasteditordisplayname, lasteditdate, lastactivitydate, tags, numbertags, answercount, commentcount, favoritecount, bodylen, bhemo, bsemo, bupper, bpunctu,bcode, titlelen, themo, tsemo, tupper, tpunctu, tagsintitle, tagsinbody,tcode) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"

	for event, elem in context:
		
		if(elem.get("PostTypeId") != "2"):
			elem.clear()
			continue

		counter+=1

		stripped = ""
		tags= ""
		tagnumber = 0
		
		bodylen= 0
		bhemo= 0
		bsemo= 0 
		bupper= 0 
		bpunctu = 0
		bcode = False
		
		titlelen= 0
		themo= 0
		tsemo= 0
		tupper= 0
		tpunctu = 0
		tcode = False
		 
		tagsintitle = 0
		tagsinbody = 0

		try:
			if(elem.get("Body")):
				stripped = elem.get("Body").encode('utf-8','ignore')
				bodylen = len(stripped)					
				tokenized = tok.tokenize(stripped)
        		bhemo = sum(p in happyemoticons for p in tokenized)
        		bsemo =  sum(p in sademoticons for p in tokenized)
        		bupper =  float(sum(x.isupper() for x in stripped)) / float(len(stripped)) * 100
        		bpunctu =  sum(o in string.punctuation for o in stripped)
        		bcode = True if sum(o in codetext for o in tokenized) > 0 else False
        		del tokenized
				# res = emoticonFinder(stripped)
				# if res != "NA":
			if(elem.get("Title")):
				tstripped = elem.get("Title").encode('utf-8','ignore')
				titlelen = len(elem.get("Title").encode('utf-8','ignore'))
				tokenized = tok.tokenize(tstripped)
                                themo = sum(p in happyemoticons for p in tokenized)
                                tsemo =  sum(p in sademoticons for p in tokenized)
                                tupper =  float(sum(x.isupper() for x in tstripped)) / float(len(tstripped)) * 100
                                tpunctu =  sum(o in string.punctuation for o in tstripped)
                                tcode = True if sum(o in codetext for o in tokenized) > 0 else False
                                del tokenized

		except UnicodeDecodeError, e:
			print 'Error %s' % e    
		

		try:
			if(elem.get("Tags")):
				#print elem.get("Tags").encode('utf-8','ignore')
				tags = re.sub('[<]', '', elem.get("Tags").encode('utf-8','ignore'))
				tags = re.sub('[>]', '|', tags)[:-1]
				tagnumber = len(tags.split("|"))
				#tags = strip_tags(elem.get("Tags").encode('utf-8','ignore'))
				for t in tags.split("|"):
                                        if(elem.get("Title") != None):
                                            if t in elem.get("Title"):
                                                    tagsintitle += elem.get("Title").count(t);
					if t in elem.get("Body"):
						tagsinbody += elem.get("Body").count(t)

			#print tagnumber,tagsInTitle, tagsInBody

		except UnicodeDecodeError, e:
			print 'Error %s' % e    

		answer = ( elem.get("Id"), #"Id": 
					elem.get("ParentId") if elem.get("ParentId") else None, #"acceptedanswer": 
					elem.get("CreationDate"), #"CreationDate": 
					elem.get("ClosedDate") if elem.get("ClosedDate") else None, #"ClosedDate": 					
					True if elem.get("ClosedDate") else False, #"isclosed": 					
					elem.get("Score") if elem.get("Score") else None, #"score": 
					elem.get("ViewCount") if elem.get("ViewCount") else None, #"ViewCount": 					
					stripped,
					elem.get("OwnerUserId") if elem.get("OwnerUserId") else None, #"OwnerUserId":
					elem.get("OwnerDisplayName") if elem.get("OwnerDisplayName") else None, #"OwnerDisplayName"
					elem.get("Title"), #"Location": 
					elem.get("LastEditorUserId"), #"LastEditorUserId": 
					elem.get("LastEditorDisplayName") if elem.get("LastEditorDisplayName") else None, #"LastEditorDisplayName"					
					elem.get("LastEditDate"), #"LastEditDate": 
					elem.get("LastActivityDate"), #"LastActivityDate": 
       				tags,
       				tagnumber,
					elem.get("AnswerCount") if elem.get("AnswerCount") else None, #"AnswerCount": 
					elem.get("CommentCount") if elem.get("CommentCount") else None, #"CommentCount": 
					elem.get("FavoriteCount") if elem.get("FavoriteCount") else None, #"AnswerCount": 					
					bodylen, 
					bhemo, 
					bsemo, 
					bupper, 
					bpunctu, 
					bcode,
					titlelen, 
					themo, 
					tsemo, 
					tupper, 
					tpunctu, 
					tagsintitle, 
					tagsinbody,
					tcode
		)

		#print question
		
		cur.execute(query,answer)

		del answer

		if(counter == MAXINQUERY or elem.getnext() == False):
			numbatch+=1
			counter = 0
			#print counter
			print "... commiting batch number " + str(numbatch) + ". Elapsed time: " + str(time.time() - start_time)
			con.commit()
			#return
			
		
		elem.clear()
		while elem.getprevious() is not None:
			del elem.getparent()[0]
		#print user["Id"]


def fast_iterClosedQuestionsHistory(context, cur,con):
	numbatch = 0
	counter = 0
	MAXINQUERY = 10000

	start_time = time.time()
	
	print "... START TIME at:" + str(start_time)

	query = "INSERT INTO closedquestionhistory (id, postid, posthistorytypeid, revisionguid, creationdate, userid, userdisplayname, comment, originalquestion, voters, votersnumber) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
		
	for event, elem in context:
		
		if(elem.get("PostHistoryTypeId") != "10"):  #only posts voted to be closed
			elem.clear()
			continue

		counter+=1
		stripped = ""
		voters= ""
		votersnumber = 0
		originalquestion = None
		
		try:
			if(elem.get("Text")):
				histText = json.loads(elem.get("Text"))
				if("OriginalQuestionIds" in histText):
					if(len(histText["OriginalQuestionIds"]) > 0):
						originalquestion = histText["OriginalQuestionIds"][0]

				votersnumber = len(histText["Voters"])
				
				for x in histText["Voters"]:
					if("Id" in x):
						voters += str(x["Id"]) + "|"
					else:
						voters += str(x["DisplayName"]) + "|"

				voters = voters[:-1]
				del histText
				

		except UnicodeDecodeError, e:
			print 'Error %s' % e   

		historyrecord = ( elem.get("Id"), #"Id": 
					elem.get("PostId") if elem.get("PostId") else None, #"PostId": 
					elem.get("PostHistoryTypeId"), #"PostHistoryTypeId": 
					elem.get("RevisionGUID") if elem.get("RevisionGUID") else None, #"RevisionGUID": 					
					elem.get("CreationDate") if elem.get("CreationDate") else None, #"CreationDate": 										
					elem.get("UserId") if elem.get("UserId") else None, #"UserId":
					elem.get("UserDisplayName") if elem.get("UserDisplayName") else None, #"UserDisplayName"					
					elem.get("Comment") if elem.get("Comment") else None, #"Comment"					
					originalquestion,					
       				voters,
       				votersnumber       				
		)
				
		cur.execute(query,historyrecord)

		del historyrecord

		if(counter == MAXINQUERY or elem.getnext() == False):
			numbatch+=1
			counter = 0
			#print counter
			print "... commiting batch number " + str(numbatch) + ". Elapsed time: " + str(time.time() - start_time)
			con.commit()	

			
		
		elem.clear()
		while elem.getprevious() is not None:
			del elem.getparent()[0]
		#print user["Id"]



def fast_iterClosedQuestionsHistory2(context, cur,con): 
	numbatch = 0
	counter = 0
	MAXINQUERY = 10000

	start_time = time.time()
	
	print "... START TIME at:" + str(start_time)

	query = "INSERT INTO closedquestionhistory_edit (id, postid, posthistorytypeid, revisionguid, creationdate, userid, userdisplayname, comment, text) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
		
	for event, elem in context:
		
		if not(elem.get("PostHistoryTypeId") == "4" or elem.get("PostHistoryTypeId") == "5" or elem.get("PostHistoryTypeId") == "6"):  #only posts editted
			elem.clear()
			continue

		counter+=1
		stripped = ""
		
		historyrecord = ( elem.get("Id"), #"Id": 
					elem.get("PostId") if elem.get("PostId") else None, #"PostId": 
					elem.get("PostHistoryTypeId"), #"PostHistoryTypeId": 
					elem.get("RevisionGUID") if elem.get("RevisionGUID") else None, #"RevisionGUID": 					
					elem.get("CreationDate") if elem.get("CreationDate") else None, #"CreationDate": 										
					elem.get("UserId") if elem.get("UserId") else None, #"UserId":
					elem.get("UserDisplayName") if elem.get("UserDisplayName") else None, #"UserDisplayName"					
					elem.get("Comment") if elem.get("Comment") else None, #"Comment"					
                                        elem.get("Text") if elem.get("Text") else None, #Text
		)
				
		cur.execute(query,historyrecord)

		del historyrecord

		if(counter == MAXINQUERY or elem.getnext() == False):
			numbatch+=1
			counter = 0
			#print counter
			print "... commiting batch number " + str(numbatch) + ". Elapsed time: " + str(time.time() - start_time)
			con.commit()	

			
		
		elem.clear()
		while elem.getprevious() is not None:
			del elem.getparent()[0]
		#print user["Id"]

def fast_iterClosedQuestionsHistory3(context, cur,con):
	numbatch = 0
	counter = 0
	MAXINQUERY = 10000
    
	start_time = time.time()
	
	print "... START TIME at:" + str(start_time)
    
	query = "INSERT INTO closedquestionhistory_orign (id, postid, posthistorytypeid, revisionguid, creationdate, userid, userdisplayname, comment, text) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
    
	for event, elem in context:
		
		if not(elem.get("PostHistoryTypeId") == "1" or elem.get("PostHistoryTypeId") == "2" or elem.get("PostHistoryTypeId") == "3"):  #only original posts
			elem.clear()
			continue
        
		counter+=1
		stripped = ""
		
		historyrecord = ( elem.get("Id"), #"Id":
                         elem.get("PostId") if elem.get("PostId") else None, #"PostId":
                         elem.get("PostHistoryTypeId"), #"PostHistoryTypeId":
                         elem.get("RevisionGUID") if elem.get("RevisionGUID") else None, #"RevisionGUID":
                         elem.get("CreationDate") if elem.get("CreationDate") else None, #"CreationDate":
                         elem.get("UserId") if elem.get("UserId") else None, #"UserId":
                         elem.get("UserDisplayName") if elem.get("UserDisplayName") else None, #"UserDisplayName"
                         elem.get("Comment") if elem.get("Comment") else None, #"Comment"
                         elem.get("Text") if elem.get("Text") else None, #Text
                         )
                         
                cur.execute(query,historyrecord)

                del historyrecord

                if(counter == MAXINQUERY or elem.getnext() == False):
                 numbatch+=1
                 counter = 0
                 #print counter
                 print "... commiting batch number " + str(numbatch) + ". Elapsed time: " + str(time.time() - start_time)
                 con.commit()	



                elem.clear()
                while elem.getprevious() is not None:
                 del elem.getparent()[0]
                #print user["Id"]


			
def fast_iterVotes(context, cur,con):
	numbatch = 0
	counter = 0
	MAXINQUERY = 10000

	start_time = time.time()
	
	print "... START TIME at:" + str(start_time)

	query = "INSERT INTO votes (id, postid, votetypeid, userid, creationdate, bountyamount) VALUES (%s, %s, %s, %s, %s, %s)"
		
	for event, elem in context:
		
		if(elem.get("VoteTypeId") != "1" and elem.get("VoteTypeId") != "2" and elem.get("VoteTypeId") != "3"):  #only posts voted to be closed
                        #print elem.get("VoteTypeId")
			elem.clear()
			continue
		    
                counter+=1

		vote = ( elem.get("Id"), #"Id": 
					elem.get("PostId") if elem.get("PostId") else None, #"PostId": 
					elem.get("VoteTypeId"), #"VoteTypeId":
                                        elem.get("UserId") if elem.get("UserId") else None, #"UserId":
    					elem.get("CreationDate") if elem.get("CreationDate") else None, #"CreationDate": 										
					elem.get("BountyAmount") if elem.get("BountyAmount") else None, #"Comment"					      				
		)
				
		cur.execute(query,vote)

		del vote

		if(counter == MAXINQUERY or elem.getnext() == False):
			numbatch+=1
			counter = 0
			#print counter
			print "... commiting batch number " + str(numbatch) + ". Elapsed time: " + str(time.time() - start_time)
			con.commit()	

			
		
		elem.clear()
		while elem.getprevious() is not None:
			del elem.getparent()[0]
		#print user["Id"]

def fast_iterComments(context, cur,con):
	numbatch = 0
	counter = 0
	MAXINQUERY = 10000

	start_time = time.time()
	
	print "... START TIME at:" + str(start_time)

	query = "INSERT INTO comments (id, postid, score, text, creationdate, userdisplayname, userid) VALUES (%s, %s, %s, %s, %s, %s, %s)"
		
	for event, elem in context:
		    
                counter+=1

		comment = ( elem.get("Id"), #"Id": 
					elem.get("PostId") if elem.get("PostId") else None, 
					elem.get("Score") if elem.get("Score") else None, 
                                        elem.get("Text") if elem.get("Text") else None, 
    					elem.get("CreationDate") if elem.get("CreationDate") else None,  										
					elem.get("UserDisplayName") if elem.get("UserDisplayName") else None,
                                        elem.get("UserId") if elem.get("UserId") else None
		)
		
		cur.execute(query,comment)

		del comment

		if(counter == MAXINQUERY or elem.getnext() == False):
			numbatch+=1
			counter = 0
			#print counter
			print "... commiting batch number " + str(numbatch) + ". Elapsed time: " + str(time.time() - start_time)
			con.commit()	

			
		
		elem.clear()
		while elem.getprevious() is not None:
			del elem.getparent()[0]
			
def fast_iterBadges(context, cur,con):
	numbatch = 0
	counter = 0
	MAXINQUERY = 10000

	start_time = time.time()
	
	print "... START TIME at:" + str(start_time)

	query = "INSERT INTO badges (id, userid, name, date) VALUES (%s, %s, %s, %s)"
		
	for event, elem in context:
		    
                counter+=1

		badge = ( elem.get("Id"), #"Id": 
					elem.get("UserId") if elem.get("UserId") else None, 
					elem.get("Name") if elem.get("Name") else None, 
                    elem.get("Date") #if elem.get("CreationDate") else None
		)
		
		cur.execute(query,badge)

		del badge

		if(counter == MAXINQUERY or elem.getnext() == False):
			numbatch+=1
			counter = 0
			#print counter
			print "... commiting batch number " + str(numbatch) + ". Elapsed time: " + str(time.time() - start_time)
			con.commit()	

			
		
		elem.clear()
		while elem.getprevious() is not None:
			del elem.getparent()[0]

    
def main(type, op):
	
    con = None

    if(type == "users"):
        infile = 'raw_data/stackoverflow.com-Users'
    if(type == "questions"):
        infile = 'raw_data/stackoverflow.com-Posts'
    if(type == "answers"):
        infile = 'raw_data/stackoverflow.com-Posts'
    if(type == "closedquestionhistory" or type == "closedquestionhistory2" or type == "closedquestionhistory3"): # for closing question and edit questions respectively
        infile = 'raw_data/stackoverflow.com-PostHistory'
    if(type == "votes"):
        infile = 'raw_data/stackoverflow.com-Votes'
    if(type == "comments"):
        infile = 'raw_data/stackoverflow.com-Comments'
    if(type == "badges"):
        infile = 'raw_data/stackoverflow.com-Badges'

    try:
        #CAMBIA CAMBIA CAMBIA CAMBIA
        con = psycopg2.connect(database='stackquestionstest', user='postgres', password='021709Yj') 
        cur = con.cursor()
        cur.execute('SELECT version()')          
        ver = cur.fetchone()
        print "Connecting to DB: " + str(ver)

        if(type == "users" and op=="0"):
            cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('users',))
            if(cur.fetchone()[0]):
                print "... table user existing"
                cur.execute("DROP TABLE users")
            
            cur.execute("CREATE TABLE users(id INT PRIMARY KEY, reputation INT, creationdate timestamp, displayname VARCHAR(50), lastaccessdate timestamp, location TEXT, aboutme TEXT, views INT, upvotes INT, downvotes INT, emailhash TEXT, lentext INT, hemo INT, semo INT, upperc FLOAT, punctu INT, code TEXT, hasurl TEXT, uurls TEXT, numurls INT, numgooglecode INT, numgithub INT, numtwitter INT, numlinkedin INT, numgoogleplus INT, numfacebook INT)")

        if(type == "questions" and op=="0"):
            cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('questions',))
            if(cur.fetchone()[0]):
                print "... table questions existing"
                cur.execute("DROP TABLE questions")				
            
            cur.execute("CREATE TABLE questions(id INT PRIMARY KEY, acceptedanswer INT, creationdate timestamp,	closeddate timestamp, isclosed TEXT, score INT, viewcount INT, body TEXT, owneruserid INT, ownerdisplayname TEXT, title TEXT, lasteditoruserid INT, lasteditordisplayname TEXT, lasteditdate timestamp, lastactivitydate timestamp, tags TEXT, numbertags INT, answercount INT, commentcount INT, favoritecount INT, bodylen INT, bhemo INT, bsemo INT, bupper FLOAT, bpunctu INT, bcode TEXT, titlelen INT, themo INT, tsemo INT, tupper FLOAT, tpunctu INT, tagsintitle INT, tagsinbody INT, tcode TEXT)")
            
        if(type == "answers" and op=="0"):
            cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('answers',))
            if(cur.fetchone()[0]):
                print "... table answers existing"
                cur.execute("DROP TABLE answers")				
            
            cur.execute("CREATE TABLE answers(id INT PRIMARY KEY, parentId INT, creationdate timestamp,	closeddate timestamp, isclosed TEXT, score INT, viewcount INT, body TEXT, owneruserid INT, ownerdisplayname TEXT, title TEXT, lasteditoruserid INT, lasteditordisplayname TEXT, lasteditdate timestamp, lastactivitydate timestamp, tags TEXT, numbertags INT, answercount INT, commentcount INT, favoritecount INT, bodylen INT, bhemo INT, bsemo INT, bupper FLOAT, bpunctu INT, bcode TEXT, titlelen INT, themo INT, tsemo INT, tupper FLOAT, tpunctu INT, tagsintitle INT, tagsinbody INT, tcode TEXT)")

        if(type == "closedquestionhistory" and op=="0"):
            cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('closedquestionhistory',))
            if(cur.fetchone()[0]):
                print "... table closedquestionhistory existing"
                cur.execute("DROP TABLE closedquestionhistory")				
            
            cur.execute("CREATE TABLE closedquestionhistory(id INT PRIMARY KEY, postid INT, posthistorytypeid INT, revisionguid TEXT, creationdate timestamp, userid INT, userdisplayname TEXT, comment INT, originalquestion INT, voters TEXT, votersnumber INT)")

        if(type == "closedquestionhistory2" and op=="0"):
            cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('closedquestionhistory_edit',))
            if(cur.fetchone()[0]):
                print "... table closedquestionhistory_edit existing"
                cur.execute("DROP TABLE closedquestionhistory_edit")
            
            cur.execute("CREATE TABLE closedquestionhistory_edit(id INT PRIMARY KEY, postid INT, posthistorytypeid INT, revisionguid TEXT, creationdate timestamp, userid INT, userdisplayname TEXT, comment TEXT, text TEXT)")

        if(type == "closedquestionhistory3" and op=="0"):
            cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('closedquestionhistory_orign',))
            if(cur.fetchone()[0]):
                print "... table closedquestionhistory_orign existing"
                cur.execute("DROP TABLE closedquestionhistory_orign")
            
            cur.execute("CREATE TABLE closedquestionhistory_orign(id INT PRIMARY KEY, postid INT, posthistorytypeid INT, revisionguid TEXT, creationdate timestamp, userid INT, userdisplayname TEXT, comment TEXT, text TEXT)")
                
        if(type == "votes" and op=="0"):
            cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('votes',))
            if(cur.fetchone()[0]):
                print "... table votes existing"
                cur.execute("DROP TABLE votes")				
            
            cur.execute("CREATE TABLE votes(id INT PRIMARY KEY, postid INT, votetypeid INT, userid INT, creationdate timestamp, bountyamount INT)")
            
        if(type == "comments" and op=="0"):
            cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('comments',))
            if(cur.fetchone()[0]):
                print "... table comments existing"
                cur.execute("DROP TABLE comments")				
            
            cur.execute("CREATE TABLE comments(id INT PRIMARY KEY, postid INT, score INT, Text TEXT, creationdate timestamp, userdisplayname TEXT, userid INT)")

        if(type == "badges" and op=="0"):
            cur.execute("select exists(select * from information_schema.tables where table_name=%s)", ('badges',))
            if(cur.fetchone()[0]):
                print "... table badges existing"
                cur.execute("DROP TABLE badges")				
            
            cur.execute("CREATE TABLE badges(id INT PRIMARY KEY, userid INT, name TEXT, date timestamp)")


        con.commit()
		
        print "... table " + str(type) + " created"

        context = etree.iterparse(infile, events=('end',), tag='row')

        if(type == "users"):		
                questions = fast_iterUser(context, cur,con)
                print "... finished committing user data."	
                #print "... creating index"
                #cur.execute("CREATE INDEX users_id_index ON users (id);")
                con.commit()

        if(type == "questions"):		
                questions = fast_iterQuestions(context, cur,con)
                print "... finished committing question data. Part: "+op
                #print "... creating index"
                #cur.execute("CREATE INDEX id_index ON users (id);")
                #cur.execute("CREATE INDEX questions_id_index ON questions (id);")
                con.commit()

        if(type == "answers"):		
                questions = fast_iterAnswers(context, cur,con)
                print "... finished committing answers data. Part: "+op
                #print "... creating index"
                #cur.execute("CREATE INDEX id_index ON users (id);")
                #cur.execute("CREATE INDEX answers_id_index ON answers (id);")
                con.commit()

        if(type == "closedquestionhistory"):		
                questions = fast_iterClosedQuestionsHistory(context, cur,con)
                print "... finished committing closedquestionhistory data."	
                #print "... creating index"
                #cur.execute("CREATE INDEX id_index ON users (id);")
                #cur.execute("CREATE INDEX closedquestionhistory_id_index ON closedquestionhistory (id);")
                con.commit()

        if(type == "closedquestionhistory2"):
            questions = fast_iterClosedQuestionsHistory2(context, cur,con)
            print "... finished committing closedquestionhistory_edit data."
            #print "... creating index"
            #cur.execute("CREATE INDEX id_index ON users (id);")
            #cur.execute("CREATE INDEX closedquestionhistory_edit_id_index ON closedquestionhistory_edit (id);")
            con.commit()

        if(type == "closedquestionhistory3"):
            questions = fast_iterClosedQuestionsHistory3(context, cur,con)
            print "... finished committing closedquestionhistory_orign data."
            #print "... creating index"
            #cur.execute("CREATE INDEX id_index ON users (id);")
            #cur.execute("CREATE INDEX closedquestionhistory_orign_id_index ON closedquestionhistory_orign (id);")
            #cur.execute("CREATE INDEX closedquestionhistory_orign_postid_index ON closedquestionhistory_orign (postid);")
            con.commit()

        if(type == "votes"):		
                questions = fast_iterVotes(context, cur,con)
                print "... finished committing vote data."
                #print "... creating index"
                #cur.execute("CREATE INDEX id_index ON users (id);")	
                #cur.execute("CREATE INDEX votes_id_index ON votes (id);")
                con.commit()
                
        if(type == "comments"):		
                questions = fast_iterComments(context, cur,con)
                print "... finished committing comment data."	
                #print "... creating index"
                #cur.execute("CREATE INDEX id_index ON users (id);")	
                #cur.execute("CREATE INDEX comments_id_index ON comments (id);")
                con.commit()
                
        if(type == "badges"):		
                questions = fast_iterBadges(context, cur,con)
                print "... finished committing badge data."	
                #print "... creating index"
                #cur.execute("CREATE INDEX id_index ON users (id);")	
                #cur.execute("CREATE INDEX badges_id_index ON badges (id);")
                con.commit()			
        
        print "... done with indexing"

    except psycopg2.DatabaseError, e:
            print 'Error %s' % e    
            sys.exit(1)
    
    finally:
            if con:
                    con.close()
	


if __name__ == '__main__':
	#types: user, questions, answers, votes, comments, closedquestionhistory, closedquestionhistory2 (edit), closedquestionhistory3 (orign)
    type = sys.argv[1]
    main(type, sys.argv[2])



       
