# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:42:49 2019

@author: avi
"""

import pandas as pd 
import numpy as np 
import sklearn 
import matplotlib.pyplot as plt #visualize data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression, LinearRegression
import re
from textblob import Word
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from gensim.models.word2vec import Word2Vec
from xgboost import XGBClassifier
import xgboost
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import operator
nltk.download('lexicon')
nltk.downloader.download('vader_lexicon')



##############################Data load
train_datas=pd.read_csv('D:\\Python project\Accenture\\Accenture ML Question\\dataset\\train.csv')
test_datas=pd.read_csv('D:\\Python project\\Accenture\\Accenture ML Question\\dataset\\test.csv')

train_datas.isnull().sum().sum()

test_datas.isnull().sum()


train_datas.columns

cmvc=train_datas['comment'].value_counts()

scorevc=train_datas['score'].value_counts()

sns.countplot(x = 'score', data=train_datas[0:20])

sns.set(style="whitegrid")

sns.boxplot(x=train_datas['score'])

plt.scatter(train_datas.index, train_datas['score'], alpha=0.5)
plt.title('Scatter plot')
plt.xlabel('Index')
plt.ylabel('Score')
plt.show()

train_datas.max()

train_datas['score'].min()

stop = stopwords.words('english')

train_datas['allcomment']=train_datas['comment']+" "+train_datas['parent_comment']

#train_datas=train_datas.drop("allcommenyt",axis=1)

train_datas['parent_comment'] = train_datas['parent_comment'].str.lower()

train_datas['parent_comment'] = train_datas['parent_comment'].str.replace(r"[^a-zA-Z]+", " ")

train_datas['parent_comment'] = train_datas['parent_comment'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

train_datas['parent_comment'] = train_datas['parent_comment'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

#test
test_datas['allcomment']=test_datas['comment']+" "+test_datas['parent_comment']

test_datas['parent_comment'] = test_datas['parent_comment'].str.lower()

test_datas['parent_comment'] = test_datas['parent_comment'].fillna("")

test_datas['parent_comment'] = test_datas['parent_comment'].str.replace(r"[^a-zA-Z]+", " ")

test_datas['parent_comment'] = test_datas['parent_comment'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

test_datas['parent_comment'] = test_datas['parent_comment'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


sid = SentimentIntensityAnalyzer()

listy = []

for index, row in train_datas.iterrows():
  ss = sid.polarity_scores(row["parent_comment"])
  listy.append(ss)
se = pd.Series(listy)
train_datas['polarity1'] = pd.DataFrame(se.values)

pol_val=train_datas['polarity1'].apply(pd.Series)

df_final = pd.concat([train_datas, pol_val], axis = 1)

testpol=[]

for index, row in test_datas.iterrows():
  sss = sid.polarity_scores(row["parent_comment"])
  testpol.append(sss)
se2 = pd.Series(testpol)
test_datas['polarity1'] = pd.DataFrame(se2.values)

pol_val_test=test_datas['polarity1'].apply(pd.Series)

df_final_test= pd.concat([test_datas, pol_val_test], axis = 1)

df_final.columns
df_final_test.columns

def getpola(text):
    return TextBlob(text).sentiment.polarity
    
df_final['polarity']=df_final['parent_comment'].apply(getpola)

def getsubv(text):
    return TextBlob(text).sentiment.subjectivity

df_final['subjectivity']=df_final['parent_comment'].apply(getsubv)

    
df_final_test['polarity']=df_final_test['parent_comment'].apply(getpola)

df_final_test['subjectivity']=df_final_test['parent_comment'].apply(getsubv)


df_final['sentiment'] = df_final['parent_comment'].apply(lambda x: TextBlob(x).sentiment[0] )
df_final[['UID','sentiment']].head()

df_final.columns

######
df_final['neg_neu']=df_final['neg']+df_final['neu']

df_final['comp_pos_ps']=df_final['compound']+df_final['pos']+df_final['polarity']+df_final['subjectivity']

df_final_test['neg_neu']=df_final_test['neg']+df_final_test['neu']

df_final_test['comp_pos_ps']=df_final_test['compound']+df_final_test['pos']+df_final_test['polarity']+df_final_test['subjectivity']

colx=['pos']


#############
Lg_model = LogisticRegression()
Lg_model.fit(df_final[colx],df_final['score'])
pred_mnew = Lg_model.predict(df_final_test[colx])

clf=Pipeline([('vect',TfidfVectorizer(norm='l2',ngram_range=(1,2),use_idf=True,smooth_idf=True, sublinear_tf=False)),('clf',LogisticRegression())])

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
tfidf_vect_ngram.fit(train_datas['parent_comment'])
xtrain_tfidf =  tfidf_vect_ngram.transform(train_datas['parent_comment'])
xtest_tfidf =  tfidf_vect_ngram.transform(test_datas['parent_comment'])

clf=XGBClassifier(n_estimators=50,max_depth=5,min_child_weight=1,learning_rate=0.1,silent=True,objective='binary:logistic',gamma=0,max_delta_step=0,subsample=1,colsample_bytree=1,colsample_bylevel=1,
                           reg_alpha=0,
                           reg_lambda=0,
                           scale_pos_weight=1,
                           seed=1,
                           missing=None)

X_train, X_test, y_train, y_test = train_test_split(
    xtrain_tfidf,train_datas['score'], test_size=0.2, random_state=42)
model2=Lg_model.fit(X_train,y_train)
pred_mnew = model2.predict(X_test)

model2=Lg_model.fit(xtrain_tfidf,train_datas['score'])
pred_mnew=model2.predict(xtest_tfidf)

############
model = xgboost.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=100,
                 min_child_weight=1.5,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42) 

######split reg
X_train, X_test, y_train, y_test = train_test_split(
    df_final[colx],df_final['score'], test_size=0.2, random_state=42)
model.fit(X_train,y_train)
pred_mnew = model.predict(X_test)

###########
model.fit(df_final[colx],df_final['score'])
pred_mnew = model.predict(df_final_test[colx])


#regress
from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(y_test, pred_mnew))

#####classificati acc
from sklearn.metrics import confusion_matrix
#confusion_matrix(X_test, y_test)
accuracy_score(y_test, pred_mnew) 


model=clf.fit(train_datas['allcomment'],train_datas['score'])

from sklearn.naive_bayes import MultinomialNB
clf1 = MultinomialNB()


'''
def getvalues(prdy):
    dsd=[]
    for i in prdy:
         dsd.append(i[1])                  
    return dsd  

pid=getvalues(prdy)'''



test_datas["score"]=pred_mnew

test_datas["score"].value_counts()

out_file=test_datas[["UID","score"]]


out_file.to_csv("D:\\Python project\Accenture\\Accenture ML Question\\dataset\\output.csv", index=False, header=True)










