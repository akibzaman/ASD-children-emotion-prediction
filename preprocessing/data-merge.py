# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 04:51:16 2021

@author: Hp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier

from yellowbrick.target import FeatureCorrelation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.multiclass import OneVsRestClassifier

import xgboost as xgb
from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import seaborn as sns
import pickle
from boruta import BorutaPy

from collections import Counter
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE
 

feature_names=['eegRawValue', 'eegRawValueVolts' ,'Attention', 'Mediation',
             'Blinkstrength', 'Delta','Theta', 'Alphalow', 'AlphaHigh', 
             'Betalow', 'Betahigh', 'GamaLow', 'GamaMid', 'Status']


train_feature=['eegRawValue', 'eegRawValueVolts' ,'Attention', 'Mediation',
             'Blinkstrength', 'Delta','Theta', 'Alphalow', 'AlphaHigh', 
             'Betalow', 'Betahigh', 'GamaLow', 'GamaMid']




happy ='alldata/clean-original/clean_happy_data.csv'
sad ='alldata/clean-original/clean_sad_data.csv'
normal='alldata/clean-original/clean_normal_data.csv'
# happy ='alldata/train_data_happy.csv'
# sad ='alldata/train_data_sad.csv'
# normal='alldata/train_data_normal.csv'

df_happy = pd.read_csv(happy, header=None, names=feature_names)
df_sad = pd.read_csv(sad, header=None, names=feature_names)
df_normal = pd.read_csv(normal, header=None, names=feature_names)


data_frame=[df_happy, df_sad, df_normal]
#data_frame=[df_happy, df_sad]
dataset=pd.concat(data_frame)
dataset=dataset.drop(0)
dataset=dataset.reset_index()
dataset=dataset.drop(['index'], axis=1)


dataset=dataset.sample(frac=1) ##shuffling the dataset
dataset=dataset.reset_index()
dataset=dataset.drop(['index'], axis=1)

X = dataset[train_feature] # Features
for x in range(len(train_feature)):
    X[train_feature[x]] = pd.to_numeric(X[train_feature[x]],
           errors='coerce')

# #########Normalization:
# scaler=MinMaxScaler()
# scaler.fit(X)
# X=scaler.transform(X)

labelencoder = LabelEncoder()

y = dataset['Status']# Target variable
y= labelencoder.fit_transform(y)

counter = Counter(y)
for k,v in counter.items():
	per = v / len(y) * 100
	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# plot the distribution
pyplot.bar(counter.keys(), counter.values())
pyplot.show()


# oversample = SMOTE()
# X, y = oversample.fit_resample(X, y)
# # summarize distribution
# counter = Counter(y)
# for k,v in counter.items():
#  	per = v / len(y) * 100
#  	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()

y=pd.DataFrame(labelencoder.inverse_transform(y), columns=['Status'])
data_frame=[X, y]
dataset=pd.concat(data_frame, axis=1)
dataset.shape

# dataset.to_csv("alldata/dataset.csv", index=None, columns= feature_names)
dataset.to_csv("alldata/clean-original/clean-dataset.csv", index=None, columns= feature_names)

##################################FEATURE RANKING########################

# ######################  ANOVA f-test Feature Selection  ##############
# fs = SelectKBest(score_func=f_classif, k='all')
# fs.fit(X, y)
# # what are scores for the features
# for i in range(len(fs.scores_)):
#  	print('Feature %d: %f' % (i, fs.scores_[i]))
    
# # what are scores for the features
# selected_feature = []
# for i in range(len(fs.pvalues_)):
#     if(fs.pvalues_[i]<0.05):
#         selected_feature.append(fs.feature_names_in_[i])
#         print('Feature %d : %f' % (i,fs.pvalues_[i]),fs.feature_names_in_[i])
        
# # data_selected = data [selected_feature]
# # data = pd.read_csv("text-data/text-100-raw.csv")
# # y = data['category']
# # data_frame=[data_selected, y]
# # data_selected=pd.concat(data_frame, axis=1)
# # data_selected.to_csv("text-data/100_pval_selected.csv")
# # # plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()


# ######################  Mutual Information  #####################
# fs = SelectKBest(score_func=mutual_info_classif, k='all')
# fs.fit(X, y)
# # what are scores for the features
# for i in range(len(fs.scores_)):
# 	print('Feature %d: %f' % (i, fs.scores_[i]))
    
# # what are scores for the features
# selected_feature = []
# for i in range(len(fs.scores_)):
#     if(fs.scores_[i]>0):
#         selected_feature.append(fs.feature_names_in_[i])
#         print('Feature %d : %f' % (i,fs.scores_[i]),fs.feature_names_in_[i])
        
# # data_selected = data [selected_feature]
# # data = pd.read_csv("image-data/image-selected.csv")
# # y = data['category']
# # data_frame=[data_selected, y]
# # data_selected=pd.concat(data_frame, axis=1)
# # data_selected.to_csv("image-data/mi_selected_features.csv")
# # plot the scores
# pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
# pyplot.show()




# # # ##Mutual-Information Classification

# visualizer = FeatureCorrelation(
#     method='mutual_info-classification', feature_names=None, sort=True)

# visualizer.fit(X, y)     # Fit the data to the visualizer
# visualizer.show()   


# # from yellowbrick.features import JointPlotVisualizer
# # visualizer = JointPlotVisualizer(columns="cement")

# # visualizer.fit_transform(X, y)        # Fit and transform the data
# # visualizer.show()


# # from yellowbrick.features import Rank1D
# # visualizer = Rank1D(algorithm='shapiro')

# # visualizer.fit(X, y)           # Fit the data to the visualizer
# # a=visualizer.transform(X)        # Transform the data
# # visualizer.show()  