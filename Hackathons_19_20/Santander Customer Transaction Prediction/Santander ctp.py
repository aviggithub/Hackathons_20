# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 01:38:56 2019

@author: avi
"""
import pandas as pd 
import numpy as np 
import sklearn 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression,LinearRegression
import seaborn as sns
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
##############################Data load
train_data=pd.read_csv('D:\\Python project\kaggle\\Santander Customer Transaction Prediction\\train.csv')
test_data=pd.read_csv('D:\\Python project\\kaggle\\Santander Customer Transaction Prediction\\test.csv')
sampl=pd.read_csv('D:\\Python project\\kaggle\\Santander Customer Transaction Prediction\\sample_submission.csv')

sampl.target.value_counts()

cor_df=train_data[['max', 'min', 'mean', 'sum', 'median','target']].corr()

train_data.columns[0:5]

max_var=cor_df.loc[cor_df["target"]>0.5]

chdata=cor_df[0:1]

train_data['nw_var']=train_data.iloc[:,2:].mean(axis=1)

train_data=train_data.drop("nw_var",axis=1)

target_vc=train_data['target'].value_counts()

train_data.isnull().sum().sum()

test_data.isnull().sum().sum()

sns.countplot(x = 'target', data=train_data)

'''
X_train, X_test, y_train, y_test = train_test_split(
    xtrain, ytrain, test_size=0.2, random_state=42)'''


model2=XGBClassifier(n_estimators=50,max_depth=10,min_child_weight=1,learning_rate=0.1,silent=True,objective='binary:logistic',gamma=0,max_delta_step=0,subsample=1,colsample_bytree=1,colsample_bylevel=1,
                           reg_alpha=0,
                           reg_lambda=0,
                           scale_pos_weight=1,
                           seed=1,
                           missing=None)

model2=LinearRegression()
model2=LogisticRegression()

from lightgbm import LGBMRegressor, LGBMClassifier

from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=2)

from sklearn.neural_network import MLPClassifier
model2 = MLPClassifier(alpha=2)

model2=LGBMClassifier()
model2=LGBMRegressor()

train_data['max']=train_data.iloc[:,2:203].max(axis=1)
train_data['min']=train_data.iloc[:,2:203].min(axis=1)
train_data['mean']=train_data.iloc[:,2:203].mean(axis=1)
train_data['sum']=train_data.iloc[:,2:203].sum(axis=1)
train_data['median']=train_data.iloc[:,2:203].median(axis=1)

test_data['max']=test_data.iloc[:,1:].max(axis=1)
test_data['min']=test_data.iloc[:,1:].min(axis=1)
test_data['mean']=test_data.iloc[:,1:].mean(axis=1)
test_data['sum']=test_data.iloc[:,1:].sum(axis=1)
test_data['median']=test_data.iloc[:,1:].median(axis=1)

train_data['sum'].value_counts()

train_data[['max', 'sum', 'median']]
train_data.iloc[1:2,202:]

train_data.columns[0:4]

X = StandardScaler().fit_transform(train_data[train_data.columns[2:202]])
nwX=train_data[mncol]
nwX.drop(["ID_code","target"], axis=1, inplace=True)
X_train, X_test, y_train, y_test = train_test_split(
    train_data[['max', 'sum', 'median']],train_data["target"], test_size=0.2, random_state=42)
model=model2.fit(X_train,y_train)
pred_mnew = model.predict(X_test)

from sklearn.metrics import confusion_matrix
accuracy_score(y_test,pred_mnew) 
confusion_matrix(y_test,pred_mnew) 

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, pred_mnew))
print(rms)



clf_model=model2.fit(train_data[train_data.columns[-5:]],train_data["target"])

XX=StandardScaler().fit_transform(test_data[test_data.columns[1:201]])

predc=clf_model.predict(test_data[test_data.columns[1:]])

'''
accuracy_score(y_test, predc) 
fmodel2 = model2.fit(xtrain,ytrain)'''

test_data["target"]=predc

test_data["target"].value_counts()

out_file=test_data[["ID_code","target"]]

out_file.to_csv("D:\\Python project\\kaggle\\Santander Customer Transaction Prediction\\output2.csv", index=False, header=True)



########feature selection
from sklearn.ensemble import RandomForestRegressor
df=train_data[train_data.columns[2:202]]
model = RandomForestRegressor(random_state=1, max_depth=10)
df=pd.get_dummies(df)
model.fit(df,train_data["target"])
features = df.columns
importances = model.feature_importances_
indices = np.argsort(importances)[-100:]  # top 10 features
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

mncol=[]
for i in indices:
    mncol.append(features[i])
    

from sklearn.feature_selection import SelectFromModel
feature = SelectFromModel(model)
Fit = feature.fit_transform(df, train_data["target"])

############
from sklearn.feature_selection import SelectPercentile, chi2
X_new = SelectPercentile(percentile=80).fit_transform(train_data[train_data.columns[2:202]],train_data["target"])

X_new.shape

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
bestfeatures = SelectKBest(k=50)
fit = bestfeatures.fit(train_data[train_data.columns[2:202]],train_data["target"])
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(train_data.columns)
#concat two dataframes for better visualization 
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
featureScores.Specs
topf=featureScores.nlargest(10,'Score')  #print 10 best features

coln=topf.Specs