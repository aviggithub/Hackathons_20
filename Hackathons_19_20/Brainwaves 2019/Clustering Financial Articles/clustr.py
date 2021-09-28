# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 12:26:53 2019

@author: avi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
file_n_news2="D:\\Python project\\brainwaves 2019\\Clustering Financial Articles\\dataset\\news.csv"
train_data_news=pd.read_csv(file_n_news2)

train_data_news.tail(10)

train_data_news.headline= train_data_news.headline.apply(lambda x: x.lower())
train_data_news.text= train_data_news.text.apply(lambda x: x.lower())
#preprocess
#remove punctation
#train_data_news["headline"] = train_data_news["headline"].str.replace('[^\w\s]','')
train_data_news["headline"] = train_data_news["headline"].str.replace('[01234567890]','')
train_data_news["text"] = train_data_news["text"].str.replace('[01234567890]','')

h_new=[]
#remove html tags


for j in train_data_news.headline:
    p1 = BeautifulSoup(j, "html.parser").text
    s1 = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,^%*&\r\n]", " ",p1 )
    sentence = " ".join(re.split("\s+", s1, flags=re.UNICODE))
    h_new.append(sentence)
    
#train_data_news=train_data_news.drop(['text_2','headline_2'],axis=1,inplace=True)

train_data_news["headline_2"]=h_new

t_new=[]
for j in train_data_news.text:
    p2 = BeautifulSoup(j, "html.parser").text
    s2 = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,^%*&'\r\n]", " ",p2 )
    sentence = " ".join(re.split("\s+", s2, flags=re.UNICODE))
    t_new.append(sentence)
    
train_data_news['text_2']=t_new

#remove stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
train_data_news["headline_2"] = train_data_news["headline_2"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train_data_news["text_2"] = train_data_news["text_2"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

train_data_news.head()

#train_data_news["headline_2"]= train_data_news["headline_2"].apply(lambda x: " ".join(x.lower() for x in x.split()))

#lemmatization 
from textblob import Word
train_data_news["headline_2"] = train_data_news["headline_2"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train_data_news["text_2"] = train_data_news["text_2"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

train_data_news.head()

#stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
train_data_news["headline_2"]=train_data_news["headline_2"].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
train_data_news["text_2"]=train_data_news["text_2"].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

#sspell correct
from textblob import TextBlob
train_data_news["headline_2"]=train_data_news["headline_2"].apply(lambda x: str(TextBlob(x).correct()))
train_data_news["text_2"]=train_data_news["text_2"].apply(lambda x: str(TextBlob(x).correct()))

df=train_data_news[["headline_2","text_2"]]

df.rename(columns={'headline_2' :'headline','text_2':'text'}, inplace=True)

df['textn']=df['headline'] + df['text']
df.head()
#vectorizer = TfidfVectorizer(max_df=0.85,stop_words='english',max_features=2)

def get_stop_words(stop_file_path):
    """load stop words """
    
    with open(stop_file_path, 'r', encoding="utf-8") as f:
        stopwords = f.readlines()
        stop_set = set(m.strip() for m in stopwords)
        return frozenset(stop_set)
stopwords=get_stop_words("D:\\Python project\\stopwords_en.txt")
vectorizer=CountVectorizer(max_df=0.20,stop_words=stopwords,max_features=2)
XX = vectorizer.fit_transform(df.textn)
XXy=XX.toarray()
np.savetxt('D:\\Python project\\brainwaves 2019\\Clustering Financial Articles\\dataset\\samplearray_idf2.txt', XXy, delimiter=' ')

true_k=5

model = KMeans(n_clusters=5).fit(XX)
#kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=500, n_init=1)
#p_data=model.fit(XX)

#Y = vectorizer.transform(df.textn)
prediction = model.predict(XX)
#print(prediction)

f_tblel=train_data_news.copy()

f_tblel['cluster']=prediction

f_pred=f_tblel[["id","cluster"]]
f_pred.tail()
f_pred.groupby('cluster').size()

f_pred.to_csv("D:\\Python project\\brainwaves 2019\\Clustering Financial Articles\\dataset\\sampleCsvt2.csv", sep=',', encoding='utf-8',index=False)




#new_clean_data=train_data_news[["id","headline_2","text_2"]]

#new_clean_data.to_csv("D:\\Python project\\brainwaves 2019\\Clustering Financial Articles\\dataset\\clustering_op_2.csv", sep=',', encoding='utf-8',index=False)


