# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 21:55:20 2018

@author: avi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.spatial import distance_matrix
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer
file_n_news="D:\\Python project\\brainwaves 2019\\Clustering Financial Articles\\dataset\\news.csv"
train_data_news=pd.read_csv(file_n_news)

train_data_news.head(2)
    
#preprocess
#remove punctation
train_data_news["headline"] = train_data_news["headline"].str.replace('[^\w\s]','')
train_data_news["headline"].head()

#remove stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
train_data_news["headline"] = train_data_news["headline"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train_data_news["headline"].head()

train_data_news["headline"]= train_data_news["headline"].apply(lambda x: " ".join(x.lower() for x in x.split()))

#lemmatization 
from textblob import Word
train_data_news["headline"] = train_data_news["headline"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
train_data_news["headline"].head()

#stemming
from nltk.stem import PorterStemmer
st = PorterStemmer()
train_data_news["headline"]=train_data_news["headline"].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

#sspell correct
from textblob import TextBlob
train_data_news["headline"]=train_data_news["headline"][:5].apply(lambda x: str(TextBlob(x).correct()))

mdl = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
head_l = mdl.fit_transform(train_data_news["headline"])
print(head_l)


head_line_data=train_data_news["headline"]
ary_hed=head_line_data

ary_hed.shape

vectorizer = TfidfVectorizer(stop_words='english',max_features=2, lowercase=True)
X = vectorizer.fit_transform(train_data_news.headline.values.astype('U'))
Xary=X.toarray() 
dis_X=distance_matrix(Xary,Xary)

type(dis_X)

coo_matrix([[0]], dtype=np.float16).todense()
todense(dis_X)

Xary1=pd.DataFrame(Xary)
len(Xary)

xarray=[]
for i in dis_X:
    xarray.append(i)

f_xarray=np.asarray(xarray)
    
len(f_xarray)
type(X)

true_k=5
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
p_data=model.fit(X)
p_data.precompute_distances

print("Top terms per cluster :")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print
print("\n")

Y = vectorizer.transform(ary_hed)
prediction = model.predict(Y)
print(prediction)


op_d=pd.DataFrame(prediction)
op_d.rename(columns={0 :'cluster'}, inplace=True)

op_d.columns

op_d.dtypes

op_d.shape

op_d.groupby('cluster').size()

cluster_count=op_d.groupby('cluster').count()
plt.bar(op_d.index.values, op_d['cluster'])
plt.xlabel('Y')
plt.ylabel('X')
plt.show()


#op_d['id']=train_data_news["id"]

f_tble=train_data_news.copy()

f_tble['cluster']=prediction


f_pred=f_tble[["id","cluster"]]

f_pred.groupby('cluster').size()

file = open("D:\\Python project\\brainwaves 2019\\Clustering Financial Articles\\dataset\\clustering.txt",”w”) 
 
file.write(“Hello World”) 

f_pred.to_csv("D:\\Python project\\brainwaves 2019\\Clustering Financial Articles\\dataset\\clustering_op_2.csv", sep=',', encoding='utf-8',index=False)

new_file=open("D:\\Python project\\brainwaves 2019\\Clustering Financial Articles\\dataset\\disf.txt",mode="w",encoding="utf-8")
#A = np.squeeze(np.asarray(xarray))
for i in dis_X:
    new_file.write(str(i))
new_file.close()

A.shape
A = np.asarray(Y)

print(A)

from scipy import sparse
b=sparse.csr_matrix(X)
print(b)
#op_d.columns
#op_d.rename(columns={0 :'cluster'}, inplace=True)

#op_d=op_d.rename(columns={0: 'cluster'}, inplace=True)



from sklearn.feature_extraction.text import CountVectorizer
bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
train_bow = bow.fit_transform(train_data_news['headline'].values.astype('U'))

train_bow.shape
train_bow.dtype

print(train_bow.as.array)

#train_data_news['headline'].dtype
train_data_news['headline'].apply(lambda x: TextBlob(x).sentiment)

































#88888888888888888888888888
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score




documents = ["This little kitty came to play when I was eating at a restaurant.",
             "Merley has the best squooshy kitten belly.",
             "Google Translate app is incredible.",
             "If you open 100 tab in google you get a smiley face.",
             "Best cat photo I've ever taken.",
             "Climbing ninja cat.",
             "Impressed with google map feedback.",
             "Key promoter extension for Google Chrome."]

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

true_k = 2
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
model.fit(X)

print("Top terms per cluster:")
order_centroids = model.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()
for i in range(true_k):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind]),
    print

print("\n")
print("Prediction")

Y = vectorizer.transform(["chrome browser to open."])
prediction = model.predict(Y)
print(prediction)

Y = vectorizer.transform(["My cow is very hungry."])
prediction = model.predict(Y)
print(prediction)


#hdbsacn
data=Y.copy()
import hdbscan
from sklearn.datasets import make_blobs
import time
import sklearn.cluster as cluster
%matplotlib inline
sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}
data, _ = make_blobs(1000)

clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
cluster_labels = clusterer.fit_predict(data)

cluster_labels

def plot_clusters(data, algorithm, args, kwds):
    start_time = time.time()
    labels = algorithm(*args, **kwds).fit_predict(data)
    end_time = time.time()
    palette = sns.color_palette('deep', np.unique(labels).max() + 1)
    colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]
    plt.scatter(data.T[0], data.T[1], c=colors, **plot_kwds)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    plt.title('Clusters found by {}'.format(str(algorithm.__name__)), fontsize=24)
    plt.text(-0.5, 0.7, 'Clustering took {:.2f} s'.format(end_time - start_time), fontsize=14)

#plot scatter 
plt.scatter(data.T[0], data.T[1], c='b', **plot_kwds)
frame = plt.gca()
#frame.axes.get_xaxis().set_visible(False)
#frame.axes.get_yaxis().set_visible(False)

#kmeans
plot_clusters(X, cluster.KMeans, (), {'n_clusters':5})

#dbscan plot
plot_clusters(data, cluster.DBSCAN, (), {'eps':0.025})

#AgglomerativeClustering
plot_clusters(data, cluster.AgglomerativeClustering, (), {'n_clusters':6, 'linkage':'ward'})

#SpectralClustering
plot_clusters(data, cluster.SpectralClustering, (), {'n_clusters':6})

#MeanShift
plot_clusters(data, cluster.MeanShift, (0.175,), {'cluster_all':False})

#AffinityPropagation
plot_clusters(data, cluster.AffinityPropagation, (), {'preference':-5.0, 'damping':0.95})


#plot_clusters(Y, hdbscan.HDBSCAN, (), {'min_cluster_size':10})

