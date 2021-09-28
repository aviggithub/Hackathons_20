# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 21:55:20 2018

@author: avi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
file_n_trn="D:\\Python project\\brainwaves 2019\\complaint status tracking\\train.csv"
file_n_tst="D:\\Python project\\brainwaves 2019\\complaint status tracking\\test.csv"
train_data=pd.read_csv(file_n_trn)
test_data=pd.read_csv(file_n_tst)

train_data.head(2)

train_data.info()

train_data.columns

train_data.isnull().sum()

tran_typ=train_data.groupby(['Transaction-Type']).size()

tran_typ

tran_typ.shape

tran_typ.hist()

comp_sta_group=train_data.groupby(['Complaint-Status']).size()

con_dispute=train_data.groupby(['Consumer-disputes']).size()

train_data.groupby([['Complaint-Status','Transaction-Type']]).size()







