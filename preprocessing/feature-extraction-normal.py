
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 23:04:07 2021

@author: xaman@19
"""

#import numpy as np
import pandas as pd
#from datetime import datetime, timedelta

col_names = ['Time','poorSignal', 'eegRawValue', 'eegRawValueVolts' ,'Attention', 'Mediation',
             'Blinkstrength', 'Delta','Theta', 'Alphalow', 'AlphaHigh', 
             'Betalow', 'Betahigh', 'GamaLow', 'GamaMid','tag','Location', 'Status']

refined_col=['eegRawValue', 'eegRawValueVolts' ,'Attention', 'Mediation',
             'Blinkstrength', 'Delta','Theta', 'Alphalow', 'AlphaHigh', 
             'Betalow', 'Betahigh', 'GamaLow', 'GamaMid', 'Status']


#### Dataset Load : Normal----->ormal due to the \n problem
UserFiles= ['new_data\ormal\person1-n.csv'
,'new_data\ormal\person2(N).csv'
,'new_data\ormal\person3(N).csv'
,'new_data\ormal\person4(N).csv'
,'new_data\ormal\person5(N).csv'
,'new_data\ormal\person6(N).csv'
,'new_data\ormal\person7(N).csv'
# ,'new_data\ormal\person8(N).csv'
# ,'new_data\ormal\person9(N).csv'
# ,'new_data\ormal\person10(N).csv'
,'new_data\ormal\person11(N).csv'
,'new_data\ormal\person12(N).csv'
,'new_data\ormal\person13(N).csv'
,'new_data\ormal\person14(N).csv'
,'new_data\ormal\person15(N).csv'
,'new_data\ormal\person16(N).csv'
,'new_data\ormal\person17(N).csv'
,'new_data\ormal\person18(N).csv'
]

###Data File Assigning
uservarnames = []
for i, _ in enumerate(UserFiles):
    uservarnames.append("User_"+str(i+1)+"N")
userfilesnamedict = {}
for name, file in zip(uservarnames, UserFiles):
    userfilesnamedict[name] = file

###Data_Cleaning
usernames = []
for i, _ in enumerate(range(17)):
    usernames.append("user"+str(i+1))
datadict = {}
for name, userfilename in zip(usernames, userfilesnamedict):
    datadict[name] = pd.read_csv(userfilesnamedict[userfilename], header=None, names=col_names)
    datadict[name]['Status'] = 'N'
    #p1=p1.drop(['Time','eegRawValue', 'eegRawValueVolts','tag','Location' ], axis=1)
    # datadict[name]=datadict[name].drop(['Time','Attention', 'Mediation','Blinkstrength', 'Delta','Theta', 'Alphalow', 'AlphaHigh', 
    #          'Betalow', 'Betahigh', 'GamaLow', 'GamaMid','tag','Location' ], axis=1)
    datadict[name] = datadict[name][datadict[name]['poorSignal'] == '0']
# for item in datadict:
#     datadict(item, "=", datadict[item])
#     datadict(item)


###Asigning the Variables
list=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
for i,item in zip(range(17),datadict):
  vars()["p"+list[i]]=datadict[item]



###Initialize a DataFrame
#df = pd.DataFrame(columns= ['timestampMs','date','time','date_time'])
## df = pd.DataFrame(columns= ['timestampMs','date','time','second'])
#
##Copy timestamp from EEG data to DataFrame
#df['timestampMs'] = pd.to_datetime(p1['Time'], unit='ms')
##print (df)
#
##Convert timestamp to UTC date and time
#df['timestampMs'] = pd.to_datetime(df.timestampMs)
### Adding Hours
#hours_to_add = 6 #Defining the time zone UTC+06
#df['timestampMs'] = df['timestampMs'] + timedelta(hours = hours_to_add) #Converting to local time zone
#df['date'] = df['timestampMs'].dt.strftime('%d/%m/%Y')
#df['time'] = df['timestampMs'].dt.strftime('%H:%M:%S')
## print ( df['time'].dtypes)
#df['date_time'] = df['time'] + " " + df['date']
#p1['Time']=df['date_time']



normal_frames = [p1,p2, p3, p4 ,p5 ,p6 ,p7]# ,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18]
train_data_normal = pd.concat(normal_frames)
print(train_data_normal.shape)
train_data_normal=train_data_normal.drop_duplicates()
train_data_normal=train_data_normal.drop(['Time','poorSignal','tag','Location'], axis=1)
print(train_data_normal.shape)
train_data_normal = train_data_normal[train_data_normal['Attention'] > '10']
train_data_normal = train_data_normal[train_data_normal['Mediation'] > '10']
train_data_normal = train_data_normal[train_data_normal['Blinkstrength'] > '10']

train_data_normal.to_csv("alldata/train_data_normal.csv", index=None, columns= refined_col)

