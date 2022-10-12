# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 04:51:16 2021

@author: Hp
"""

# import numpy as np
import pandas as pd
import scikitplot as skplt
# import matplotlib.pyplot as plt
# import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

# from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline

# from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier

# from sklearn import svm
# from sklearn.svm import LinearSVC
# from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import IsolationForest
# from sklearn.ensemble import StackingClassifier
# from sklearn.ensemble import BaggingClassifier

# from sklearn.multiclass import OneVsRestClassifier

import xgboost as xgb
import lightgbm as lgb
# from sklearn.neural_network import MLPClassifier

# from sklearn.ensemble import VotingClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
# import pickle
# from boruta import BorutaPy
from yellowbrick.classifier import ROCAUC

# from collections import Counter
# from matplotlib import pyplot
# from imblearn.over_sampling import SMOTE

# import seaborn as sns
 

feature_names=['eegRawValue', 'eegRawValueVolts' ,'Attention', 'Mediation',
             'Blinkstrength', 'Delta','Theta', 'Alphalow', 'AlphaHigh', 
             'Betalow', 'Betahigh', 'GamaLow', 'GamaMid', 'Status']


train_feature=[#'eegRawValue', 'eegRawValueVolts' ,
               'Attention', 'Mediation',
             'Blinkstrength', 'Delta','Theta', 'Alphalow', 
             'AlphaHigh', 
             'Betalow', 'Betahigh', 'GamaLow', 'GamaMid']

# # dataset = pd.read_csv("alldata/dataset.csv")
# # dataset = pd.read_csv("alldata/selected-dataset.csv")
# dataset = pd.read_csv("alldata/clean-mean/selected-clean-dataset.csv")
# # dataset = pd.read_csv("alldata/clean-original/selected-clean-dataset.csv")

# #df['Time'] = df['Time'].astype(float)
# for x in range(len(train_feature)):
#     #df['Time'] = pd.to_numeric(df['Time'],errors='coerce')
#     #df['Heartrate'] = pd.to_numeric(df['Heartrate'],errors='coerce')
#     dataset[train_feature[x]] = pd.to_numeric(dataset[train_feature[x]],
#            errors='coerce')

# #dataset=dataset.dropna()
    
# #print (df.dtypes)
# #print(df)




# labelencoder = LabelEncoder()
# dataset['Status']= labelencoder.fit_transform(dataset['Status'])
# #dataset['Status']=  pd.to_numeric(dataset['Status'],errors='coerce')
# #print(df['Status'])



# X = dataset[train_feature] # Features
# y = dataset['Status']# Target variable

# # #########Normalization:
# #scaler = StandardScaler()
# scaler=MinMaxScaler()
# scaler.fit(X)
# X=scaler.transform(X)
# #X=X.dropna()
# #y=y.dropna()
# #print(X)
# #print(y)
# # np.any(np.isnan(X))
# # np.all(np.isfinite(X))

# # y = dataset['Status']# Target variable
# # y= labelencoder.fit_transform(y)

# counter = Counter(y)
# for k,v in counter.items():
#  	per = v / len(y) * 100
#  	print('Class=%d, n=%d (%.3f%%)' % (k, v, per))
# # plot the distribution
# pyplot.bar(counter.keys(), counter.values())
# pyplot.show()


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

# # Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1
#                                                     #,stratify=y, 
#                                                     ,random_state=32)


# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2
#                                                     #,stratify=y, 
#                                                     ,random_state=32)


train_data = pd.read_csv("alldata/clean-mean/selected-clean-train.csv")
valid_data = pd.read_csv("alldata/clean-mean/selected-clean-valid.csv")
test_data = pd.read_csv("alldata/clean-mean/selected-clean-test.csv")

labelencoder = LabelEncoder()
train_data['Status']= labelencoder.fit_transform(train_data['Status'])
valid_data['Status']= labelencoder.fit_transform(valid_data['Status'])
test_data['Status']= labelencoder.fit_transform(test_data['Status'])

X_train = train_data[train_feature] # Features
y_train = train_data['Status']# Target variable

X_valid = valid_data[train_feature] # Features
y_valid = valid_data['Status']# Target variable

X_test = test_data[train_feature] # Features
y_test = test_data['Status']# Target variable


print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)
print(X_test.shape, y_test.shape)

# ################## DATA GENARTION (TRAIN TEST VALIDATION)  ###################
# column_name=['Attention', 'Mediation',
#               'Blinkstrength', 'Delta','Theta', 'Alphalow', 'AlphaHigh', 
#               'Betalow', 'Betahigh', 'GamaLow', 'GamaMid']
# X_train=pd.DataFrame(X_train, columns = column_name)
# X_valid=pd.DataFrame(X_valid, columns = column_name)
# X_test=pd.DataFrame(X_test, columns = column_name)

# X_train=X_train.reset_index()
# X_train=X_train.drop(['index'], axis=1)

# X_valid=X_valid.reset_index()
# X_valid=X_valid.drop(['index'], axis=1)

# X_test=X_test.reset_index()
# X_test=X_test.drop(['index'], axis=1)

# y_train=y_train.reset_index()
# y_train=y_train.drop(['index'], axis=1)

# y_valid=y_valid.reset_index()
# y_valid=y_valid.drop(['index'], axis=1)

# y_test=y_test.reset_index()
# y_test=y_test.drop(['index'], axis=1)

# y_train_output=pd.DataFrame(labelencoder.inverse_transform(y_train), columns =['Status'])
# y_valid_output=pd.DataFrame(labelencoder.inverse_transform(y_valid), columns =['Status'])
# y_test_output=pd.DataFrame(labelencoder.inverse_transform(y_test), columns =['Status'])

# data_frame=[X_train, y_train_output]
# train_dataset=pd.concat(data_frame, axis=1)
# train_dataset.shape
# train_dataset.to_csv("alldata/clean-mean/selected-clean-train.csv")

# data_frame=[X_test, y_test_output]
# test_dataset=pd.concat(data_frame, axis=1)
# test_dataset.shape
# test_dataset.to_csv("alldata/clean-mean/selected-clean-test.csv")

# data_frame=[X_valid, y_valid_output]
# valid_dataset=pd.concat(data_frame, axis=1)
# valid_dataset.shape
# valid_dataset.to_csv("alldata/clean-mean/selected-clean-valid.csv")




###############     KLRE and ALRE     ########################

######VALIDATION
######KNN
# print("KNN------------------------------->")
# model_KNN = KNeighborsClassifier(n_neighbors=3,
#             algorithm='ball_tree',
#             leaf_size=5,
#             metric='minkowski',
#             p=2,
#             metric_params=None,
#             n_jobs=1)
# model_KNN.fit(X_train, y_train)
# C1_knn_pred = model_KNN.predict(X_valid)
# C1_knn_pred = pd.DataFrame(C1_knn_pred, columns=['C1KNN'])
# C1_knn_pred = C1_knn_pred.reset_index()
# C1_knn_pred = C1_knn_pred.drop(['index'], axis=1)

# ########Adaboost Model
print("ADB------------------------------->")
model_KNN = AdaBoostClassifier(n_estimators=100,learning_rate=0.5, random_state=0)
model_KNN.fit(X_train, y_train)
C1_knn_pred = model_KNN.predict(X_valid)
C1_knn_pred = pd.DataFrame(C1_knn_pred, columns=['C1KNN'])
C1_knn_pred = C1_knn_pred.reset_index()
C1_knn_pred = C1_knn_pred.drop(['index'], axis=1)


###LGBM
print("LGBM------------------------------->")
lgb_model=lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, 
                             min_data_in_leaf=2,
                              learning_rate=0.8, n_estimators=100,  
                              objective='multiclass', class_weight='balanced', 
                              min_split_gain=0.0, 
                              min_child_weight=0.03, min_child_samples=20, 
                              subsample=1.0, 
                              subsample_freq=500, colsample_bytree=1.0,
                              reg_lambda=0.65, random_state=0,num_classes=3)
lgb_model.fit(X_train,y_train)
C1_lgbm_pred = lgb_model.predict(X_valid)
C1_lgbm_pred = pd.DataFrame(C1_lgbm_pred, columns=['C2LGBM'])
C1_lgbm_pred = C1_lgbm_pred.reset_index()
C1_lgbm_pred = C1_lgbm_pred.drop(['index'], axis=1)

#RandonForest
print("--------------- RF ------------------------------->")
model_RF = RandomForestClassifier( n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=9,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,)
model_RF.fit(X_train,y_train)
C1_rf_pred = model_RF.predict(X_valid)
C1_rf_pred = pd.DataFrame(C1_rf_pred, columns=['C3RF'])
C1_rf_pred = C1_rf_pred.reset_index()
C1_rf_pred = C1_rf_pred.drop(['index'], axis=1)

actual = pd.DataFrame(y_valid, columns=['Status'])
actual = actual.reset_index()
actual = actual.drop(['index'], axis=1)

C_Final = pd.concat([C1_knn_pred, C1_lgbm_pred,C1_rf_pred, actual], axis=1)

# C_Final=[[0,0,1,1],
#          [0,1,1,1],
#          [0,0,0,0],
#          [0,1,0,0],
#          [0,1,0,1]]
# C_Final = pd.DataFrame(C_Final)

VALUE1=0.33
VALUE2=0.67
def weight_calc(df):
    weight=[1.0]*3
    for i in range (len(df)):
        instance = df.iloc[i]
        correct = 0
        print(instance[3])
        flag=[0] *3
        if(instance[0]==instance[3]):
            correct +=1
            flag[0]=1
        if(instance[1]==instance[3]):
            correct +=1
            flag[1]=1
        if(instance[2]==instance[3]):
            correct +=1
            flag[2]=1
        print(correct)
        print(flag)
        if(correct>0 and correct<3):
            print("inside")
            if(correct==2):
                for i in range (len(flag)):
                    if(flag[i]==1):
                        weight[i]+=VALUE1 
            if(correct==1):
                for i in range (len(flag)):
                    if(flag[i]==1):
                        weight[i]+=VALUE2  
    #print(weight)
    weight[0]/=(weight[1]+weight[0]+weight[2])
    weight[1]/=(weight[1]+weight[0]+weight[2])
    weight[2]/=(weight[1]+weight[0]+weight[2])
    return weight

weight = weight_calc(C_Final)
print(weight)

# print(weight[0]/(weight[1]+weight[0]+weight[2]))
# print(weight[1]/(weight[1]+weight[0]+weight[2]))
# print(weight[2]/(weight[1]+weight[0]+weight[2]))

# X_w = C_Final[["C1KNN", "C2LGBM","C3RF"]]
# model_RF_w = RandomForestClassifier(n_estimators=100,max_depth=2, random_state=0)
# model_RF_w.fit(X_w,y_valid)

######TEST
######KNN
C1_knn_test_pred = model_KNN.predict(X_test)
C1_knn_test_pred = pd.DataFrame(C1_knn_test_pred, columns=['C1KNN'])
C1_knn_test_pred = C1_knn_test_pred.reset_index()
C1_knn_test_pred = C1_knn_test_pred.drop(['index'], axis=1)

###LGBM
C1_lgbm_test_pred = lgb_model.predict(X_test)
C1_lgbm_test_pred = pd.DataFrame(C1_lgbm_test_pred, columns=['C2LGBM'])
C1_lgbm_test_pred = C1_lgbm_test_pred.reset_index()
C1_lgbm_test_pred = C1_lgbm_test_pred.drop(['index'], axis=1)

#RandonForest
C1_rf_test_pred = model_RF.predict(X_test)
C1_rf_test_pred = pd.DataFrame(C1_rf_test_pred, columns=['C3RF'])
C1_rf_test_pred = C1_rf_test_pred.reset_index()
C1_rf_test_pred = C1_rf_test_pred.drop(['index'], axis=1)


C_test_Final = pd.concat([C1_knn_test_pred, C1_lgbm_test_pred, C1_rf_test_pred], axis=1)


def AWMVE(df,weight):
    prdicted_classes=[0]*len(df)
    for i in range (len(df)):
        matrix= [[0.0,0.0,0.0]] * 3
        instance = df.iloc[i]
        for j in range(len(instance)):
            if(instance[j]==0):
                matrix[j]= [weight[0],0.0,0.0]
            elif(instance[j]==1):
                matrix[j]= [0.0,weight[1],0.0]
            else:
                matrix[j]= [0.0,0.0,weight[2]]
        sum = [0.0] * 3
        for k in range(len (matrix)):
            sum[k] = matrix[0][k] + matrix [1][k] +matrix[2][k]
        max_value = max(sum)
        index = sum.index(max_value)
        prdicted_classes[i]=index
    return prdicted_classes

# ensemble_test_pred = model_RF_w.predict(C_test_Final)
ensemble_test_pred = AWMVE(C_test_Final,weight)
print(metrics.classification_report(y_test, ensemble_test_pred))
# print(ensemble_test_pred)
# cm = confusion_matrix(y_test, ensemble_test_pred)
# print(cm)
# sns.heatmap(cm, annot=True, fmt="d", linewidth=.5)

skplt.metrics.plot_confusion_matrix(
   y_test, ensemble_test_pred,
    figsize=(10,10),
    title_fontsize='18',
    text_fontsize='28',
    title =' ',
    cmap='BuGn'
    )






# # ##MULTICLASS


########GNB
print("GNB------------------------------->")
model_GNB = GaussianNB().fit(X_train,y_train)
y_pred_train=model_GNB.predict(X_train)
y_pred_test = model_GNB.predict(X_test)
print(metrics.classification_report(y_train, y_pred_train))
print(metrics.classification_report(y_test, y_pred_test))


######KNN
print("KNN------------------------------->")
model_KNN = KNeighborsClassifier(n_neighbors=3,
            algorithm='ball_tree',
            leaf_size=5,
            metric='minkowski',
            p=2,
            metric_params=None,
            n_jobs=1)
model_KNN.fit(X_train, y_train)
y_pred_train=model_KNN.predict(X_train)
y_pred_test = model_KNN.predict(X_test)
print(metrics.classification_report(y_train, y_pred_train))
print(metrics.classification_report(y_test, y_pred_test))

visualizer = ROCAUC(model_KNN)
visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
visualizer.score(X_test, y_test)        # Evaluate the model on the test data
visualizer.show() 


########ADB
print("ADB------------------------------->")
model_adb = AdaBoostClassifier(n_estimators=100,learning_rate=0.5,
                                random_state=0)
model_adb = model_adb.fit(X_train,y_train)
y_pred_train=model_adb.predict(X_train)
y_pred_test = model_adb.predict(X_test)
print(metrics.classification_report(y_train, y_pred_train))
print(metrics.classification_report(y_test, y_pred_test))
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred_4))
#print("f1-score:",metrics.f1_score(y_test, y_pred_4))
#pickle.dump(model_4, open('trained_model\model_adb.pkl','wb'))

###LGBM
print("LGB------------------------------->")
lgb_model=lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, 
                              min_data_in_leaf=2,
                              learning_rate=0.8, n_estimators=100,  
                              objective='multiclass', class_weight='balanced', 
                              min_split_gain=0.0, 
                              min_child_weight=0.03, min_child_samples=20, 
                              subsample=1.0, 
                              subsample_freq=500, colsample_bytree=1.0,
                              reg_lambda=0.65, random_state=0,num_classes=3)
lgb_model= lgb_model.fit(X_train,y_train)
y_pred_train=lgb_model.predict(X_train)
y_pred_test = lgb_model.predict(X_test)
print(metrics.classification_report(y_train, y_pred_train))
print(metrics.classification_report(y_test, y_pred_test))
#pickle.dump(xgb_model, open('trained_model\model_lgb.pkl','wb'))


# print("XGB------------------------------>")
# ##XGBoost Classifier(54.05)
# xgb_model = xgb.XGBClassifier(objective= 'multi:softmax',num_classes=3,
#                               learning_rate=0.8)
# xgb_model= xgb_model.fit(X_train, y_train)
# y_pred_train=xgb_model.predict(X_train)
# y_pred_test = xgb_model.predict(X_test)
# print(metrics.classification_report(y_train, y_pred_train))
# print(metrics.classification_report(y_test, y_pred_test))
# #pickle.dump(xgb_model, open('trained_model\model_xgb.pkl','wb'))


#RandonForest
print("RF ------------------------------->")
model_RF = RandomForestClassifier( n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=9,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="auto",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,)
model_RF.fit(X_train,y_train)
y_pred_train=model_RF.predict(X_train)
y_pred_test = model_RF.predict(X_test)
print(metrics.classification_report(y_train, y_pred_train))
print(metrics.classification_report(y_test, y_pred_test))
# pickle.dump(model_RF, open('trained_model\model_RF.pkl','wb'))
