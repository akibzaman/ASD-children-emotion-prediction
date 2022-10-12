
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


# load dataset
UserFiles= ['new_data\sad\person1-sad.csv'
,'new_data\sad\person2-sad.csv'
,'new_data\sad\person3-sad.csv'
,'new_data\sad\person4-sad.csv'
,'new_data\sad\person5-sad.csv'
,'new_data\sad\person6-sad.csv'
,'new_data\sad\person7-sad.csv'
,'new_data\sad\person8-sad.csv'
,'new_data\sad\person9-sad.csv'
,'new_data\sad\person10-sad.csv'
,'new_data\sad\person11-sad.csv'
,'new_data\sad\person12-sad.csv'
,'new_data\sad\person13-sad.csv'
,'new_data\sad\person14-sad.csv'
,'new_data\sad\person15-sad.csv'
,'new_data\sad\person16-sad.csv'
,'new_data\sad\person17-sad.csv'
,'new_data\sad\person18-sad.csv'
,'new_data\sad\person19-sad.csv'
,'new_data\sad\person20-sad.csv'
]

###Data File Assigning
uservarnames = []
for i, _ in enumerate(UserFiles):
    uservarnames.append("User_"+str(i+1)+"S")
userfilesnamedict = {}
for name, file in zip(uservarnames, UserFiles):
    userfilesnamedict[name] = file

###Data_Cleaning
usernames = []
for i, _ in enumerate(range(20)):
    usernames.append("user"+str(i+1))
datadict = {}
for name, userfilename in zip(usernames, userfilesnamedict):
    datadict[name] = pd.read_csv(userfilesnamedict[userfilename], header=None, names=col_names)
    datadict[name]['Status'] = 'S'
    # #p1=p1.drop(['Time','eegRawValue', 'eegRawValueVolts','tag','Location' ], axis=1)
    # datadict[name]=datadict[name].drop(['Time','Attention', 'Mediation','Blinkstrength', 'Delta','Theta', 'Alphalow', 'AlphaHigh', 
    #          'Betalow', 'Betahigh', 'GamaLow', 'GamaMid','tag','Location' ], axis=1)
    datadict[name] = datadict[name][datadict[name]['poorSignal'] == '0']
# for item in datadict:
#     datadict(item, "=", datadict[item])
#     datadict(item)


###Asigning the Variables
list=['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
for i,item in zip(range(20),datadict):
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



sad_frames = [p1,p2, p3, p4 ,p5 ,p6 ,p7, p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18, p19, p20]
train_data_sad = pd.concat(sad_frames)
print(train_data_sad.shape)
train_data_sad=train_data_sad.drop_duplicates()
train_data_sad=train_data_sad.drop(['Time','poorSignal','tag','Location'], axis=1)
print(train_data_sad.shape)
train_data_sad = train_data_sad[train_data_sad['Attention'] > '10']
train_data_sad = train_data_sad[train_data_sad['Mediation'] > '10']
train_data_sad = train_data_sad[train_data_sad['Blinkstrength'] > '10']
train_data_sad.to_csv("alldata/train_data_sad.csv", index=None, columns= refined_col)


