# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 16:08:46 2019

@author: avi
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from textblob import TextBlob
from textblob import Word
import nltk
from nltk.corpus import stopwords
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
import nltk

Uans=pd.read_csv("D:\\Python project\\DataSets\\dddata.csv")


train_data=pd.read_csv("D:\\Python project\\Incedo\\incedo_participant\\train_dataset.csv")
test_data=pd.read_csv("D:\\Python project\\Incedo\\incedo_participant\\test_dataset.csv")
sub_data=pd.read_csv("D:\\Python project\\Incedo\\incedo_participant\\sample_submission.csv")


train_data.head()
train_data.isnull().sum()
train_data.Essayset.value_counts()

train_data.clarity.value_counts()

train_data.coherent.value_counts()

train_data.Essayset=train_data.Essayset.fillna(train_data.Essayset.mean())
train_data.score_3=train_data.score_3.fillna(train_data.score_3.mean())
train_data.score_4=train_data.score_4.fillna(train_data.score_4.mean())
train_data.score_5=train_data.score_5.fillna(train_data.score_5.mean())

train_data.clarity=train_data.clarity.fillna(train_data.clarity.mode())
train_data.coherent=train_data.coherent.fillna(train_data.coherent.mode())

#find mode of score1 to score 5
train_data["final_score"]=train_data.iloc[:,4:9].mode(axis=1)

train_data.dtypes

test_data.isnull().sum()

stop = stopwords.words('english')

train_data['comb_all'] = train_data['EssayText'] + " " +train_data['clarity'].fillna("") + " "+train_data['coherent'].fillna("")+ " "+train_data['Essayset'].astype(str)


train_data.max_score.value_counts()

train_data['EssayText'] = train_data['EssayText'].str.lower()

train_data['EssayText'] = train_data['EssayText'].str.replace(r"[^a-zA-Z]+", " ")

train_data['comb_all'] = train_data['comb_all'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

train_data['comb_all'] = train_data['comb_all'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


test_data['EssayText'] = test_data['EssayText'].str.lower()

test_data['EssayText'] = test_data['EssayText'].str.replace(r"[^a-zA-Z]+", " ")

test_data['comb_all'] = test_data['EssayText'] + " " +test_data['clarity'].fillna("") + " "+test_data['coherent'].fillna("")+ " "+test_data['Essayset'].astype(str)



test_data['comb_all'] = test_data['comb_all'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

test_data['comb_all'] = test_data['comb_all'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

train_data.final_score=train_data.final_score.astype("int64")

#test_data['comb_all2']= test_data['comb_all'] + " "+ test_data['min_score'].astype(str) 
test_data['comb_all3']=test_data['comb_all']+ " "+test_data['max_score'].astype(str)

#train_data['comb_all2']= train_data['comb_all'] + " "+ train_data['min_score'].astype(str) 
train_data['comb_all3']=train_data['comb_all']+ " "+train_data['max_score'].astype(str)

#from lightgbm import LGBMClassifier

clf=Pipeline([('vect',TfidfVectorizer(norm='l2',ngram_range=(1,2),use_idf=True,smooth_idf=True, sublinear_tf=False)),('clf',LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1500,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0))])

train_data.final_score=train_data.final_score.astype("category").cat.codes

train_data.final_score.value_counts()

train_data.dtypes

X_train, X_test, y_train, y_test = train_test_split(
    train_data['EssayText'],train_data['score_1'], test_size=0.2, random_state=42)

model=clf.fit(X_train,y_train)
pred=model.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
accuracy_score(y_test,pred) 
confusion_matrix(y_test,pred) 

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, pred))
print(rms)

train_data.score_3=train_data.score_3.astype("int64")
train_data.score_4=train_data.score_4.astype("int64")
train_data.score_5=train_data.score_5.astype("int64")


model2=clf.fit(train_data['comb_all3'],train_data['score_5'])
pred_mnew=model2.predict(test_data['comb_all3'])

test_data["score_1"]=pred_mnew
test_data["score_2"]=pred_mnew
test_data["score_3"]=pred_mnew
test_data["score_4"]=pred_mnew
test_data["score_5"]=pred_mnew

test_data.score_5.value_counts()

test_data["final_score"]=test_data.iloc[:,7:12].median(axis=1)

#test_data["essay_score"]=test_data.iloc[:,7:12].mean(axis=1)

test_data["final_score"].value_counts()

test_data["essay_score"]=test_data["essay_score"].astype("int64")

#dd={0:0.0,1:0.5,2:1.0,3:1.5,4:2.0,5:2.5,6:3.0}

test_data.dtypes
test_data.essay_score.value_counts()

fin_op=test_data[["ID","Essayset"]]

fin_op["id"]=fin_op.ID
fin_op["essay_set"]=fin_op.Essayset
fin_op["essay_score"]=test_data["final_score"]

#fin_op["essay_score"] = fin_op["essay_score"].map(dd)

fin_op["essay_score"]=fin_op["essay_score"].round()

fin_op["essay_score"].value_counts()

op_file=fin_op[["id","essay_set","essay_score"]]

op_file.to_csv("D:\\Python project\\Incedo\\incedo_participant\\outputnlp.csv",index=False,header=True)


