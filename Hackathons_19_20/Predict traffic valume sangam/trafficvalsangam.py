# -*- coding: utf-8 -*-
"""
Created on Sun Aug  4 23:04:56 2019

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

train_data= pd.read_csv("Train.csv")
test_data_= pd.read_csv("Test.csv")
samp_data= pd.read_csv("sample_submission.csv")

train_data.isnull().sum()
test_data_.isnull().sum()
train_data.dtypes


train_data["is_holiday"].value_counts()

train_data["is_holiday_num"]=train_data.is_holiday.astype("category").cat.codes
train_data["weather_type_num"]=train_data.weather_type.astype("category").cat.codes
train_data["weather_description_num"]=train_data.weather_description.astype("category").cat.codes

test_data_["is_holiday_num"]=test_data_.is_holiday.astype("category").cat.codes
test_data_["weather_type_num"]=test_data_.weather_type.astype("category").cat.codes
test_data_["weather_description_num"]=test_data_.weather_description.astype("category").cat.codes


train_data["is_holiday_num"].value_counts()

X_train=train_data[["air_pollution_index","humidity","wind_direction","clouds_all","rain_p_h","snow_p_h","is_holiday_num","weather_type_num","weather_description_num"]]
y_train = train_data["traffic_volume"]

X_test=test_data_[["air_pollution_index","humidity","wind_direction","clouds_all","rain_p_h","snow_p_h","is_holiday_num","weather_type_num","weather_description_num"]]

X_train11=train_data[["air_pollution_index","humidity","wind_direction","clouds_all","rain_p_h","snow_p_h","is_holiday_num","weather_type_num","weather_description_num","traffic_volume"]]

X_train11.corr()

clf=LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
       importance_type='split', learning_rate=0.111, max_depth=-1,
       min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
       n_estimators=225, n_jobs=-1, num_leaves=31, objective=None,
       random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
       subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

model=clf.fit(X_train,y_train)
pred=model.predict(X_test)


test_data_["traffic_volume"]=pred

test_data_["traffic_volume"].value_counts()

op_file=test_data_[["date_time","traffic_volume"]]

op_file.to_csv("output.csv",index=False,header=True)
