# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 01:10:03 2019

@author: Samir
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk
import glob
import re as regex
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from scipy.sparse import hstack

path = r'C:\\Users\\Samir\\nile uni big data\\Courses\\DM\\proj\\tw data'
all_files = glob.glob(path + "\\*.txt")

file = []

for filename in all_files:
    df = pd.read_csv(filename, error_bad_lines=False,sep='\t',index_col=None, header=None)
    file.append(df)
    
Data = pd.concat(file, axis=0, ignore_index=True)
Data.columns=['id','label','tweet']

Test = pd.read_csv('C:\\Users\\Samir\\nile uni big data\\Courses\\DM\\proj\\test.csv')

qm=[]
em=[]
sm=[]
up=[]
hsh=[]
mms=[]

for line in Data['tweet']:
    qm.append(line.count('?'))
    em.append(line.count('!'))
    sm.append(line.count(':)'))
    up.append(len(regex.findall(r'[A-Z]',line)))
    cnt=0
    ats=0
    for word in line.split():
        if (word[0:1] == '#' and len(word) > 1):
            cnt+=1
        elif (word[0:1] == '@' and len(word) > 1):
            ats+=1
        else:
            continue
    hsh.append(cnt)
    mms.append(ats)

qmt=[]
emt=[]
smt=[]
upt=[]
hsht=[]
mmst=[]
for line in Test['tweet']:
    qmt.append(line.count('?'))
    emt.append(line.count('!'))
    smt.append(line.count(':)'))
    upt.append(len(regex.findall(r'[A-Z]',line)))
    cnt=0
    ats=0
    for word in line.split():
        if (word[0:1] == '#' and len(word) > 1):
            cnt+=1
        elif (word[0:1] == '@' and len(word) > 1):
            ats+=1
        else:
            continue
    hsht.append(cnt)
    mmst.append(ats)
    
em=pd.DataFrame(em)
qm=pd.DataFrame(qm)
sm=pd.DataFrame(sm)
up=pd.DataFrame(up)
hsh=pd.DataFrame(hsh)
mms=pd.DataFrame(mms)

emt=pd.DataFrame(emt)
qmt=pd.DataFrame(qmt)
smt=pd.DataFrame(smt)
upt=pd.DataFrame(upt)
hsht=pd.DataFrame(hsht)
mmst=pd.DataFrame(mmst)
#data_train,data_test,train_labels,test_labels = train_test_split(Data["tweet"],                   
#                                                 Data['label'], test_size=0.3,
#                                                 random_state=0)

######################################################################################################
#Cleaning the data fron username,tags,specialChars,urls
def remove_by_regex(tweets, regexp):
        tweets.loc[:, "tweet"].replace(regexp, "", inplace=True)
        return tweets
def remove_urls(tweets):
    return remove_by_regex(tweets, regex.compile(r"http.?://[^\s]+[\s]?"))

def remove_na(tweets):
    return tweets[tweets["tweet"] != "Not Available"]

#def remove_special_chars(tweets):  # it unrolls the hashtags to normal words
#    for remove in map(lambda r: regex.compile(regex.escape(r)), [",", ":", "\"", "=", "&", ";", "%", "$",
#                                                                 "@", "%", "^", "*", "(", ")", "{", "}",
#                                                                 "[", "]", "|", "/", "\\", ">", "<", "-",
#                                                                 "!", "?", ".", "'",
#                                                                 "--", "---", "#"]):
#                                                                 tweets.loc[:, "tweet"].replace(remove, "", inplace=True)
#    return tweets

def remove_usernames(tweets):
    return remove_by_regex(tweets, regex.compile(r"@[^\s]+[\s]?"))

def remove_numbers(tweets):
    return remove_by_regex(tweets, regex.compile(r"\s?[0-9]+\.?[0-9]*"))

cleanedData=remove_urls(Data)
cleanedData=remove_usernames(cleanedData)
cleanedData=remove_na(cleanedData)
#cleanedData=remove_special_chars(cleanedData)
cleanedData=remove_numbers(cleanedData)

cleanedTData=remove_urls(Test)
cleanedTData=remove_usernames(cleanedTData)
cleanedTData=remove_na(cleanedTData)
#cleanedData=remove_special_chars(cleanedData)
cleanedTData=remove_numbers(cleanedTData)


#######################################################################################################
#preprocessing
def preprocessCorpus(corpus):
    
    stops = set(stopwords.words("english"))
    filtered_docs =[]
    for doc in corpus:
        curr = ""
        item=doc.casefold()#casefolding
        for word in  regex.split("\W+",item):#removing stopwords + stemming
            if word not in stops: 
                curr = curr + PorterStemmer().stem(word) +" "
        curr = curr.strip()
        filtered_docs.append(curr)
    return filtered_docs

train_pro=preprocessCorpus(cleanedData['tweet'])
test_pro=preprocessCorpus(cleanedTData['tweet'])
train_labels=Data['label']
##########################################################################################################

#Vectorization & Classification
kf = StratifiedKFold(n_splits=10, shuffle=True)  
vectorizer = CountVectorizer(min_df=1,max_df=0.5,ngram_range=(1,2)).fit(train_pro)
data_train_vectorized = vectorizer.transform(train_pro)
data_train_vectorized=hstack([data_train_vectorized,em,qm,mms,hsh,up,sm])

data_test_vectorized = vectorizer.transform(test_pro)
data_test_vectorized=hstack([data_test_vectorized,emt,qmt,mmst,hsht,upt,smt])
print (len(vectorizer.get_feature_names()))
#clfr = naive_bayes.MultinomialNB()
#clfr = svm.SVC(kernel='linear',C=1)
#clfr = svm.SVC()\
clfr = LogisticRegression(C=1)
clfr.fit(data_train_vectorized,train_labels)
#model = BaggingClassifier(base_estimator=clfr, n_estimators=50, random_state=666)
#model = AdaBoostClassifier(n_estimators=50, random_state=666)
#clfr = RandomForestRegressor()
#clfr = tree.DecisionTreeClassifier()  
scores = cross_val_score(clfr, data_train_vectorized, train_labels, cv=kf, scoring='accuracy' )
fscores = cross_val_score(clfr, data_train_vectorized, train_labels, cv=kf, scoring='f1_macro'  )
print (scores)
print ("Avg Accu: %0.3f (+/-%0.2f)" %(scores.mean(), scores.std() *2))
print (fscores)
print ("Avg F1: %0.3f (+/-%0.2f)" %(fscores.mean(), fscores.std() *2))

predicted = clfr.predict(data_test_vectorized)

#################################################################################
pred=list(predicted)
pred2=pd.DataFrame(pred)
pred2.columns=['label']
pred_data = pd.DataFrame({'id': Test['id'],'label': pred2['label']})

pred_data.to_csv(r'C:\Users\Samir\nile uni big data\Courses\DM\proj\pred.csv')

pred_data.to_csv('result.csv',index=False)