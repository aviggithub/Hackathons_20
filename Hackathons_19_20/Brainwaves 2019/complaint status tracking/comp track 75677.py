# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 13:16:36 2019

@author: avi
"""
import pandas as pd #data manipulation and data anlysis (read files)
import numpy as np #transform data into format that model can understand
import sklearn #helps to create machine learning model
import matplotlib.pyplot as plt #visualize data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import re
from textblob import Word
from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC
stop = stopwords.words('english')
stop_f=stopwords.words('spanish')
stop_s=stopwords.words('french')

from nltk.stem.snowball import SnowballStemmer
sbFr = SnowballStemmer('french')
sbEsp = SnowballStemmer('spanish')
sbEng = SnowballStemmer('english')
######################################Pre Process procedure
com_int_cat ={0:'Closed with explanation', 
1:'Closed with non-monetary relief',
2:'Closed',
3:'Closed with monetary relief',
4:'Untimely response'}

input_int_cat ={'Closed with explanation':0, 
'Closed with non-monetary relief':1,
'Closed':2,
'Closed with monetary relief':3,
'Untimely response':4}


##############################Data load
file_read_train='D:\\Python project\\brainwaves 2019\\complaint status tracking\\train.csv'
file_read_test='D:\\Python project\\brainwaves 2019\\complaint status tracking\\test.csv'

df_train = pd.read_csv(file_read_train)
df_test = pd.read_csv(file_read_test)

#####bkp copy

df_train_cp=df_train
df_test_cp=df_test
#### Combining text
df_train["processed_summary"]=df_train["Consumer-complaint-summary"].fillna('') +" "+ df_train['Transaction-Type'].fillna('No')+" "+ df_train['Consumer-disputes'].fillna('') + " " +df_train['Company-response'].fillna('')+" "+df_train['Complaint-reason'].fillna('') 
df_test["processed_summary"]=df_test["Consumer-complaint-summary"].fillna('') +" "+ df_test['Transaction-Type'].fillna('No')+" "+ df_test['Consumer-disputes'].fillna('') + " " +df_test['Company-response'].fillna('')+" "+df_test['Complaint-reason'].fillna('') 
##### Cleaning data

### A: Train
df_train['Complaint_Status'] = df_train['Complaint-Status'].map(input_int_cat)
df_train['processed_summary']= df_train['processed_summary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df_train['processed_summary']= df_train['processed_summary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_f))
df_train['processed_summary']= df_train['processed_summary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_s))
#df_train['processed_summary']= df_train['processed_summary'].apply(lambda x: " ".join([x for x in x.split() if len(x)>2]))

#### B: Test
#df_test['processed_summary']= df_test['processed_summary'].str.lower()
df_test['processed_summary']= df_test['processed_summary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df_test['processed_summary']= df_test['processed_summary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_f))
df_test['processed_summary']= df_test['processed_summary'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_s))
#df_test['processed_summary']= df_test['processed_summary'].apply(lambda x: " ".join([x for x in x.split() if len(x)>2]))

#stemming
# A - Train  Stemming
df_train['processed_summary'] = df_train['processed_summary'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df_train['processed_summary'] = df_train['processed_summary'].apply(lambda x: " ".join([sbFr.stem(item) for item in x.split()]))
df_train['processed_summary'] = df_train['processed_summary'].apply(lambda x: " ".join([sbEsp.stem(item) for item in x.split()]))

df_train['processed_summary'] = df_train['processed_summary'].str.replace(r"[^a-zA-Z]+", " ")
df_train['processed_summary']=df_train['processed_summary'].str.replace("XXXX"," ")
df_train['processed_summary']=df_train['processed_summary'].str.replace("XX"," ")
df_train['processed_summary']=df_train['processed_summary'].str.replace(",","")
df_train['processed_summary']=df_train['processed_summary'].str.replace(".","")
df_train['processed_summary']=df_train['processed_summary'].str.replace("    "," ")
df_train['processed_summary']=df_train['processed_summary'].str.replace("  "," ")
df_train['processed_summary']=df_train['processed_summary'].str.replace("   "," ")
df_train['processed_summary']=df_train['processed_summary'].str.replace("     "," ")

# B - Test Stemming
df_test['processed_summary'] = df_test['processed_summary'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df_test['processed_summary'] = df_test['processed_summary'].apply(lambda x: " ".join([sbFr.stem(item) for item in x.split()]))
df_test['processed_summary'] = df_test['processed_summary'].apply(lambda x: " ".join([sbEsp.stem(item) for item in x.split()]))

df_test['processed_summary'] = df_test['processed_summary'].str.replace(r"[^a-zA-Z]+", " ")
df_test['processed_summary']=df_test['processed_summary'].str.replace("XXXX"," ")
df_test['processed_summary']=df_test['processed_summary'].str.replace("XX"," ")

df_test['processed_summary']=df_test['processed_summary'].str.replace(",","")
df_test['processed_summary']=df_test['processed_summary'].str.replace(".","")
df_test['processed_summary']=df_test['processed_summary'].str.replace("    "," ")
df_test['processed_summary']=df_test['processed_summary'].str.replace("  "," ")
df_test['processed_summary']=df_test['processed_summary'].str.replace("   "," ")
df_test['processed_summary']=df_test['processed_summary'].str.replace("     "," ")

##############################################################
### Split data  -train into  : test-train
"""
df_train_d, df_test_d = train_test_split(df_train,test_size=0.1,random_state=0)
df_test_d['Complaint_Status_acc']=df_test_d['Complaint-Status']
"""
######################## test / train assigment
df_train_d, df_test_d = train_test_split(df_train,test_size=0.0,random_state=0)
df_test_d=df_test
##################################################Model
### Model execution
fr_text_clf=Pipeline([('vect',TfidfVectorizer(norm='l2',ngram_range=(1,5),use_idf=True,smooth_idf=True, sublinear_tf=False)),('clf',LinearSVC(C=1.0,tol=0.1))])
#svc=LinearSVC(C=2.3,tol=0.1)
model = fr_text_clf.fit(df_train_d['processed_summary'],df_train_d['Complaint_Status'])
df_test_d['new_complain_status']=model.predict(df_test_d["processed_summary"])
df_test_d['Complaint-Status'] = df_test_d['new_complain_status'].map(com_int_cat)

df_test_d['Complaint-Status'].value_counts()
#######################################Accuracy Check
"""
df_test_d['Complaint-Status'].value_counts()

from sklearn.metrics import confusion_matrix
confusion_matrix(df_test_d["Complaint_Status_acc"], df_test_d['Complaint-Status'])
accuracy_score(df_test_d["Complaint_Status_acc"], df_test_d["Complaint-Status"]) 

"""
##################################################
### ##Output File creation
df_test_output= df_test_d[['Complaint-ID','Complaint-Status']]

df_test_output.to_csv("D:\\Python project\\brainwaves 2019\\complaint status tracking\\output_new270119_ra2cbcbxcb.csv", index=False, header=True)
##################################################
