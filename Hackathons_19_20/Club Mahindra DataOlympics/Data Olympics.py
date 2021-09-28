# -*- coding: utf-8 -*-
"""
Created on Fri May  3 09:10:46 2019

@author: avi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMClassifier,LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

train_data=pd.read_csv("D:\\Python project\\Club Mahindra DataOlympics\\train.csv")
test_data=pd.read_csv("D:\\Python project\\Club Mahindra DataOlympics\\test.csv")

train_data_c=pd.read_csv("D:\\Python project\\Club Mahindra DataOlympics\\train.csv")
test_data_c=pd.read_csv("D:\\Python project\\Club Mahindra DataOlympics\\test.csv")
#print top rows
train_data.head(5)

#find correlation
train_data.corr()
#summary of data
train_data.describe()
#count null values per variable
train_data.isnull().sum()
#total null values
train_data.isnull().sum().sum()

#fill the null data function
def fillnull_data(df):
    return df.fillna(df.mean())
#fill null value using mean
train_data.state_code_residence=fillnull_data(train_data.state_code_residence)
train_data.season_holidayed_code=fillnull_data(train_data.season_holidayed_code)
#find value counts of feature
train_data.channel_code.value_counts()
#plot
sns.countplot(train_data.channel_code)
#boxplot
sns.barplot(train_data.state_code_residence)

#find data type of all features
train_data.dtypes

#find diff data type column names and group them
str_col_datatype=train_data.columns.astype("object")

#replace / in booking date
train_data.booking_date=train_data.booking_date.str.replace("/","")

train_data.booking_date.head()

#function for replace / in dateformate
def replace_bacs(df):
    return df.str.replace("/","")

train_data.checkin_date=replace_bacs(train_data.checkin_date)
train_data.checkout_date=replace_bacs(train_data.checkout_date)

train_data.head(5)

#from string get first 4 digit  
def get_day(txt):
    #txt="1234566"
    return txt[:2]

def get_month(txt):
    #txt="1234566"
    return txt[2:4]

def get_year(txt):
    #txt="1234566"
    return txt[4:]

#use apply function for process each element want
#train_data.booking_date=train_data_c.booking_date.apply(get_daymonth)
#train_data.checkin_date=train_data_c.checkin_date.apply(get_daymonth)
#train_data.checkout_date=train_data_c.checkout_date.apply(get_daymonth)
    
train_data["checkin_date_day"]=train_data.checkin_date.apply(get_day)
train_data["checkin_date_month"]=train_data.checkin_date.apply(get_month)
train_data["checkin_date_year"]=train_data.checkin_date.apply(get_year)

train_data["checkin_date_day"]=train_data["checkin_date_day"].astype("int64")
train_data["checkin_date_month"]=train_data["checkin_date_month"].astype("int64")
train_data["checkin_date_year"]=train_data["checkin_date_year"].astype("int64")

train_data["checkout_date_day"]=train_data.checkout_date.apply(get_day)
train_data["checkout_date_month"]=train_data.checkout_date.apply(get_month)
train_data["checkout_date_year"]=train_data.checkout_date.apply(get_year)

train_data["checkout_date_day"]=train_data["checkout_date_day"].astype("int64")
train_data["checkout_date_month"]=train_data["checkout_date_month"].astype("int64")
train_data["checkout_date_year"]=train_data["checkout_date_year"].astype("int64")

train_data["booking_date_day"]=train_data.booking_date.apply(get_day)
train_data["booking_date_month"]=train_data.booking_date.apply(get_month)
train_data["booking_date_year"]=train_data.booking_date.apply(get_year)

train_data["booking_date_day"]=train_data["booking_date_day"].astype("int64")
train_data["booking_date_month"]=train_data["booking_date_month"].astype("int64")
train_data["booking_date_year"]=train_data["booking_date_year"].astype("int64")

train_data["checkout_date_day"]=train_data.checkout_date.apply(get_day)
train_data["checkout_date_month"]=train_data.checkout_date.apply(get_month)
train_data["checkout_date_year"]=train_data.checkout_date.apply(get_year)

train_data["checkout_date_day"]=train_data["checkin_date_day"].astype("int64")
train_data["checkout_date_month"]=train_data["checkin_date_month"].astype("int64")
train_data["checkout_date_year"]=train_data["checkin_date_year"].astype("int64")

train_data["chkin_chkout_day"]=train_data["checkout_date_day"]-train_data["checkin_date_day"]

#object to int
train_data["booking_date_int"]=train_data["booking_date"].astype("int64")
train_data["checkin_date_int"]=train_data["checkin_date"].astype("int64")
train_data["checkout_date_int"]=train_data["checkout_date"].astype("int64")

#object category to code
def cat_to_codes(df):
    df=df.astype("category").cat.codes
    return df.astype("int64")

train_data.resort_id.value_counts()

train_data["memberid_code"]=train_data.memberid.astype("category").cat.codes
train_data["member_age_buckets_code"]=cat_to_codes(train_data["member_age_buckets"])
train_data["cluster_code _code"]=cat_to_codes(train_data.cluster_code )
train_data["reservationstatusid_code_code"]=cat_to_codes(train_data.reservationstatusid_code)
train_data["resort_id _code"]=cat_to_codes(train_data.resort_id )
#train_data["booking_date"]=cat_to_codes(train_data.booking_date)
#train_data["checkout_date"]=cat_to_codes(train_data.checkout_date)
train_data["booking_date_code"]=train_data_c.booking_date
train_data["checkin_date_code"]=train_data_c.checkin_date
train_data["checkout_date_code"]=train_data_c.checkout_date

from datetime import datetime
from dateutil.relativedelta import relativedelta
from datetime import date
#convert object to Date time format

train_data["booking_date_code"]=pd.to_datetime(train_data["booking_date_code"], format='%d/%m/%y') 
train_data["checkin_date_code"]=pd.to_datetime(train_data["checkin_date_code"], format='%d/%m/%y') 
train_data["checkout_date_code"]=pd.to_datetime(train_data["checkout_date_code"],format='%d/%m/%y')

train_data["checkout_date_code"].head()

#find the days stay in there(diff betn checkin and checkout) 
train_data['diff_days'] = train_data['checkout_date_code'] - train_data['checkin_date_code']
train_data['diff_days']=train_data['diff_days']/np.timedelta64(1,'D')

train_data['diff_book_check'] = train_data['checkin_date_code'] - train_data['booking_date_code']
train_data['diff_book_check']=train_data['diff_book_check']/np.timedelta64(1,'D')

train_data['diff_book_chkout'] = train_data['checkout_date_code'] - train_data['booking_date_code']
train_data['diff_book_chkout']=train_data['diff_book_chkout']/np.timedelta64(1,'D')

train_data['diff_btn_nights_day'] = train_data['diff_days'] - train_data['roomnights']

train_data['roomnights'].max()
train_data.columns
##############################test data
test_data.isnull().sum()
test_data.state_code_residence=fillnull_data(test_data.state_code_residence)
test_data.season_holidayed_code=fillnull_data(test_data.season_holidayed_code)

test_data.booking_date=replace_bacs(test_data.booking_date)
test_data.checkin_date=replace_bacs(test_data.checkin_date)
test_data.checkout_date=replace_bacs(test_data.checkout_date)

test_data["checkin_date_day"]=test_data.checkin_date.apply(get_day)
test_data["checkin_date_month"]=test_data.checkin_date.apply(get_month)
test_data["checkin_date_year"]=test_data.checkin_date.apply(get_year)

test_data["checkin_date_day"]=test_data["checkin_date_day"].astype("int64")
test_data["checkin_date_month"]=test_data["checkin_date_month"].astype("int64")
test_data["checkin_date_year"]=test_data["checkin_date_year"].astype("int64")

test_data["checkout_date_day"]=test_data.checkout_date.apply(get_day)
train_data["checkout_date_month"]=train_data.checkout_date.apply(get_month)
train_data["checkout_date_year"]=train_data.checkout_date.apply(get_year)

test_data["checkout_date_day"]=test_data["checkout_date_day"].astype("int64")
train_data["checkout_date_month"]=train_data["checkout_date_month"].astype("int64")
train_data["checkout_date_year"]=train_data["checkout_date_year"].astype("int64")

test_data["booking_date_day"]=test_data.booking_date.apply(get_day)
test_data["booking_date_month"]=test_data.booking_date.apply(get_month)
test_data["booking_date_year"]=test_data.booking_date.apply(get_year)

test_data["booking_date_day"]=test_data["booking_date_day"].astype("int64")
test_data["booking_date_month"]=test_data["booking_date_month"].astype("int64")
test_data["booking_date_year"]=test_data["booking_date_year"].astype("int64")

test_data["chkin_chkout_day"]=test_data["checkout_date_day"]-test_data["checkin_date_day"]


test_data["memberid_code"]=test_data.memberid.astype("category").cat.codes
test_data["member_age_buckets_code"]=cat_to_codes(test_data["member_age_buckets"])
test_data["cluster_code _code"]=cat_to_codes(test_data.cluster_code )
test_data["reservationstatusid_code_code"]=cat_to_codes(test_data.reservationstatusid_code)
test_data["resort_id _code"]=cat_to_codes(test_data.resort_id )

test_data["booking_date_code"]=pd.to_datetime(test_data_c["booking_date"], format='%d/%m/%y') 
test_data["checkin_date_code"]=pd.to_datetime(test_data_c["checkin_date"], format='%d/%m/%y') 
test_data["checkout_date_code"]=pd.to_datetime(test_data_c["checkout_date"],format='%d/%m/%y')

test_data['diff_days'] =(test_data['checkout_date_code']) - (test_data['checkin_date_code'])
test_data['diff_days']=test_data['diff_days']/np.timedelta64(1,'D')

test_data['diff_book_check'] = test_data['checkin_date_code'] - test_data['booking_date_code']
test_data['diff_book_check']=test_data['diff_book_check']/np.timedelta64(1,'D')

test_data['diff_book_chkout'] = test_data['checkout_date_code'] - test_data['booking_date_code']
test_data['diff_book_chkout']= test_data['diff_book_chkout']/np.timedelta64(1,'D')


test_data['diff_btn_nights_day'] = test_data['diff_days'] - test_data['roomnights']

#totatl persons
train_data["total_persons"]=train_data.numberofadults + train_data.numberofchildren
test_data["total_persons"]=test_data.numberofadults + test_data.numberofchildren

test_data.dtypes

train_data.dtypes

all_inputs=["diff_days","checkin_date_day","checkin_date_month","checkin_date_year","resort_id _code","reservationstatusid_code_code","cluster_code _code","member_age_buckets_code","memberid_code","booking_type_code","total_pax","state_code_resort","state_code_residence","season_holidayed_code","roomnights","room_type_booked_code","resort_type_code","resort_region_code","persontravellingid","numberofchildren","numberofadults","main_product_code","channel_code"]
#all_inputs=["diff_days","checkin_date_day","checkin_date_month","checkin_date_year","resort_id _code","reservationstatusid_code_code","cluster_code _code","member_age_buckets_code","memberid_code","booking_type_code","total_pax","state_code_resort","state_code_residence","season_holidayed_code","roomnights","room_type_booked_code","resort_type_code","resort_region_code","persontravellingid","numberofchildren","numberofadults","main_product_code","channel_code"]

op_var=["amount_spent_per_room_night_scaled"]

new_col=["diff_days","diff_book_check","diff_book_chkout","total_persons","numberofadults","numberofchildren","roomnights","booking_type_code","total_pax","state_code_residence","resort_type_code","resort_id _code","reservationstatusid_code_code"]

X_train, X_test, y_train, y_test = train_test_split(
    train_data[all_inputs],train_data["amount_spent_per_room_night_scaled"], test_size=0.2, random_state=42)

#clf=LinearRegression()
#97.99
clf=LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
       importance_type='split', learning_rate=0.111, max_depth=-1,
       min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
       n_estimators=225, n_jobs=-1, num_leaves=31, objective=None,
       random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
       subsample=1.0, subsample_for_bin=200000, subsample_freq=0)
#clf=LGBMClassifier()
#clf=LinearSVC()

model=clf.fit(X_train,y_train)
pred=model.predict(X_test)

train_data_c.dtypes

from sklearn.metrics import mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, pred))
print(rms)


model2=clf.fit(train_data[all_inputs],train_data[op_var])
pred_v=model2.predict(test_data[all_inputs])

test_data["amount_spent_per_room_night_scaled"]=pred_v

op_file=test_data[["reservation_id","amount_spent_per_room_night_scaled"]]

#op_file.head(2)

op_file.to_csv("D:\\Python project\\Club Mahindra DataOlympics\\output.csv",index=False,header=True)



