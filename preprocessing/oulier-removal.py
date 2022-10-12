# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 15:32:28 2021
@author: Akib Zaman

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

########## BASIC UNCLEAN DATA LOAD ################

# data = pd.read_csv("alldata/train_data_happy.csv")
# data = pd.read_csv("alldata/train_data_sad.csv")
data = pd.read_csv("alldata/train_data_normal.csv")

numeric_col = ['Delta','Theta', 'Alphalow', 'AlphaHigh', 
             'Betalow', 'Betahigh', 'GamaLow', 'GamaMid']

data.boxplot(numeric_col)


for x in ['Delta']:
    q75,q25 = np.percentile(data.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    data.loc[data[x] < min,x] = np.nan
    data.loc[data[x] > max,x] = np.nan
    
for x in ['Theta']:
    q75,q25 = np.percentile(data.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    data.loc[data[x] < min,x] = np.nan
    data.loc[data[x] > max,x] = np.nan

for x in ['Alphalow']:
    q75,q25 = np.percentile(data.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    data.loc[data[x] < min,x] = np.nan
    data.loc[data[x] > max,x] = np.nan

for x in ['AlphaHigh']:
    q75,q25 = np.percentile(data.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    data.loc[data[x] < min,x] = np.nan
    data.loc[data[x] > max,x] = np.nan


for x in ['Betalow']:
    q75,q25 = np.percentile(data.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    data.loc[data[x] < min,x] = np.nan
    data.loc[data[x] > max,x] = np.nan

for x in ['Betahigh']:
    q75,q25 = np.percentile(data.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    data.loc[data[x] < min,x] = np.nan
    data.loc[data[x] > max,x] = np.nan
    
for x in ['GamaLow']:
    q75,q25 = np.percentile(data.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    data.loc[data[x] < min,x] = np.nan
    data.loc[data[x] > max,x] = np.nan

for x in ['GamaMid']:
    q75,q25 = np.percentile(data.loc[:,x],[75,25])
    intr_qr = q75-q25
 
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    data.loc[data[x] < min,x] = np.nan
    data.loc[data[x] > max,x] = np.nan


data.isnull().sum()
clean_data=data.dropna(axis = 0)
clean_data=clean_data.reset_index()
clean_data=clean_data.drop(['index'], axis=1)
# clean_data.to_csv("alldata/clean-original/clean_happy_data.csv", index=None)
# clean_data.to_csv("alldata/clean-original/clean_sad_data.csv", index=None)
clean_data.to_csv("alldata/clean-original/clean_normal_data.csv", index=None)


