# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 17:44:38 2021

@author: Akib Zaman
"""

from matplotlib import pyplot as plt
from sklearn import tree
import numpy as np
import pandas as pd
import scipy
from scipy import interp
import matplotlib
from matplotlib import pyplot as plt
from itertools import cycle
#from datetime import datetime, timedelta
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import label_binarize

import scikitplot as skplt

from scipy import sparse

from sklearn.model_selection import train_test_split
#from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier
from sklearn import ensemble
from sklearn import multioutput
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier


from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import seaborn as sns
import pickle
from collections import Counter
from matplotlib import pyplot
from imblearn.over_sampling import SMOTE


feature_names=['eegRawValue', 'eegRawValueVolts' ,'Attention', 'Mediation',
             'Blinkstrength', 'Delta','Theta', 'Alphalow', 'AlphaHigh', 
             'Betalow', 'Betahigh', 'GamaLow', 'GamaMid', 'Status']


train_feature=[#'eegRawValue', 'eegRawValueVolts' ,
               'Attention', 'Mediation',
             'Blinkstrength', 'Delta','Theta', 'Alphalow', 
             'AlphaHigh', 
             'Betalow', 'Betahigh', 'GamaLow', 'GamaMid']

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



#######NB
print("GNB------------------------------->")
model_GNB = GaussianNB().fit(X_train,y_train)
y_pred_train=model_GNB.predict(X_train)
y11_pred_test = model_GNB.predict(X_train)
# print(metrics.classification_report(y_train, y_pred_train))
# print(metrics.classification_report(y_test, y11_pred_test))

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
y12_pred_test = model_KNN.predict(X_train)
# print(metrics.classification_report(y_train, y_pred_train))
# print(metrics.classification_report(y_test, y12_pred_test))


########Adaboost Model
print("ADB------------------------------->")
model_adb = AdaBoostClassifier(n_estimators=100,learning_rate=0.5,
                                random_state=0)
model_adb = model_adb.fit(X_train,y_train)
y_pred_train=model_adb.predict(X_train)
y13_pred_test = model_adb.predict(X_train)
# print(metrics.classification_report(y_train, y_pred_train))
# print(metrics.classification_report(y_test, y13_pred_test))
#print("Accuracy:",metrics.accuracy_score(y_test, y_pred_4))
#print("f1-score:",metrics.f1_score(y_test, y_pred_4))
#pickle.dump(model_adb, open('trained_model\hybrid_model_adb.pkl','wb'))


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
y15_pred_test = lgb_model.predict(X_train)
# print(metrics.classification_report(y_train, y_pred_train))
# print(metrics.classification_report(y_test, y15_pred_test))
#pickle.dump(lgb_model, open('trained_model\hybrid_model_lgb.pkl','wb'))


#RandonForest
print("RF------------------------------->")
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
y16_pred_test = model_RF.predict(X_train)
# print(metrics.classification_report(y_train, y_pred_train))
# print(metrics.classification_report(y_test, y16_pred_test))
#pickle.dump(model_RF, open('trained_model\hybrid_model_RF.pkl','wb'))




########        ALRE    ##################

######VALIDATION

# ###ADB
print("ADB------------------------------->")
model_KNN = AdaBoostClassifier(n_estimators=100,learning_rate=0.5, random_state=0)
model_KNN.fit(X_train, y_train)
C1_knn_pred = model_KNN.predict(X_valid)
C1_knn_pred = pd.DataFrame(C1_knn_pred, columns=['C1KNN'])
C1_knn_pred = C1_knn_pred.reset_index()
C1_knn_pred = C1_knn_pred.drop(['index'], axis=1)


###LGB
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

# ######TEST
# ######KNN
# C1_knn_test_pred = model_KNN.predict(X_test)
# C1_knn_test_pred = pd.DataFrame(C1_knn_test_pred, columns=['C1KNN'])
# C1_knn_test_pred = C1_knn_test_pred.reset_index()
# C1_knn_test_pred = C1_knn_test_pred.drop(['index'], axis=1)

# ###LGBM
# C1_lgbm_test_pred = lgb_model.predict(X_test)
# C1_lgbm_test_pred = pd.DataFrame(C1_lgbm_test_pred, columns=['C2LGBM'])
# C1_lgbm_test_pred = C1_lgbm_test_pred.reset_index()
# C1_lgbm_test_pred = C1_lgbm_test_pred.drop(['index'], axis=1)

# #RandonForest
# C1_rf_test_pred = model_RF.predict(X_test)
# C1_rf_test_pred = pd.DataFrame(C1_rf_test_pred, columns=['C3RF'])
# C1_rf_test_pred = C1_rf_test_pred.reset_index()
# C1_rf_test_pred = C1_rf_test_pred.drop(['index'], axis=1)


# C_test_Final = pd.concat([C1_knn_test_pred, C1_lgbm_test_pred, C1_rf_test_pred], axis=1)


######TRAIN
######KNN
C1_knn_test_pred = model_KNN.predict(X_train)
C1_knn_test_pred = pd.DataFrame(C1_knn_test_pred, columns=['C1KNN'])
C1_knn_test_pred = C1_knn_test_pred.reset_index()
C1_knn_test_pred = C1_knn_test_pred.drop(['index'], axis=1)

###LGBM
C1_lgbm_test_pred = lgb_model.predict(X_train)
C1_lgbm_test_pred = pd.DataFrame(C1_lgbm_test_pred, columns=['C2LGBM'])
C1_lgbm_test_pred = C1_lgbm_test_pred.reset_index()
C1_lgbm_test_pred = C1_lgbm_test_pred.drop(['index'], axis=1)

#RandonForest
C1_rf_test_pred = model_RF.predict(X_train)
C1_rf_test_pred = pd.DataFrame(C1_rf_test_pred, columns=['C3RF'])
C1_rf_test_pred = C1_rf_test_pred.reset_index()
C1_rf_test_pred = C1_rf_test_pred.drop(['index'], axis=1)


C_test_Final = pd.concat([C1_knn_test_pred, C1_lgbm_test_pred, C1_rf_test_pred], axis=1)


def ALRE(df,weight):
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
ensemble_test_pred = ALRE(C_test_Final,weight)
y17_pred_test = ensemble_test_pred
# print(metrics.classification_report(y_test, ensemble_test_pred))
print(metrics.classification_report(y_train, ensemble_test_pred))
# print(ensemble_test_pred)
# cm = confusion_matrix(y_test, ensemble_test_pred)
# print(cm)
# sns.heatmap(cm, annot=True, fmt="d", linewidth=.5)


skplt.metrics.plot_confusion_matrix(
   y_train, ensemble_test_pred,
    figsize=(10,10),
    title_fontsize='18',
    text_fontsize='28',
    title =' ',
    cmap='BuGn'
    )


#######         KLRE    #############

######VALIDATION
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
C1_knn_pred = model_KNN.predict(X_valid)
C1_knn_pred = pd.DataFrame(C1_knn_pred, columns=['C1KNN'])
C1_knn_pred = C1_knn_pred.reset_index()
C1_knn_pred = C1_knn_pred.drop(['index'], axis=1)


###LGB
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

# ######TEST
# ######KNN
# C1_knn_test_pred = model_KNN.predict(X_test)
# C1_knn_test_pred = pd.DataFrame(C1_knn_test_pred, columns=['C1KNN'])
# C1_knn_test_pred = C1_knn_test_pred.reset_index()
# C1_knn_test_pred = C1_knn_test_pred.drop(['index'], axis=1)

# ###LGBM
# C1_lgbm_test_pred = lgb_model.predict(X_test)
# C1_lgbm_test_pred = pd.DataFrame(C1_lgbm_test_pred, columns=['C2LGBM'])
# C1_lgbm_test_pred = C1_lgbm_test_pred.reset_index()
# C1_lgbm_test_pred = C1_lgbm_test_pred.drop(['index'], axis=1)

# #RandonForest
# C1_rf_test_pred = model_RF.predict(X_test)
# C1_rf_test_pred = pd.DataFrame(C1_rf_test_pred, columns=['C3RF'])
# C1_rf_test_pred = C1_rf_test_pred.reset_index()
# C1_rf_test_pred = C1_rf_test_pred.drop(['index'], axis=1)


# C_test_Final = pd.concat([C1_knn_test_pred, C1_lgbm_test_pred, C1_rf_test_pred], axis=1)

######TRAIN
######KNN
C1_knn_test_pred = model_KNN.predict(X_train)
C1_knn_test_pred = pd.DataFrame(C1_knn_test_pred, columns=['C1KNN'])
C1_knn_test_pred = C1_knn_test_pred.reset_index()
C1_knn_test_pred = C1_knn_test_pred.drop(['index'], axis=1)

###LGBM
C1_lgbm_test_pred = lgb_model.predict(X_train)
C1_lgbm_test_pred = pd.DataFrame(C1_lgbm_test_pred, columns=['C2LGBM'])
C1_lgbm_test_pred = C1_lgbm_test_pred.reset_index()
C1_lgbm_test_pred = C1_lgbm_test_pred.drop(['index'], axis=1)

#RandonForest
C1_rf_test_pred = model_RF.predict(X_train)
C1_rf_test_pred = pd.DataFrame(C1_rf_test_pred, columns=['C3RF'])
C1_rf_test_pred = C1_rf_test_pred.reset_index()
C1_rf_test_pred = C1_rf_test_pred.drop(['index'], axis=1)


C_test_Final = pd.concat([C1_knn_test_pred, C1_lgbm_test_pred, C1_rf_test_pred], axis=1)



def KLRE(df,weight):
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
ensemble_test_pred = KLRE(C_test_Final,weight)
y18_pred_test=ensemble_test_pred
# print(metrics.classification_report(y_test, ensemble_test_pred))
print(metrics.classification_report(y_train, ensemble_test_pred))
# print(ensemble_test_pred)
# cm = confusion_matrix(y_test, ensemble_test_pred)
# print(cm)
# sns.heatmap(cm, annot=True, fmt="d", linewidth=.5)

skplt.metrics.plot_confusion_matrix(
   y_train, ensemble_test_pred,
    figsize=(10,10),
    title_fontsize='18',
    text_fontsize='28',
    title =' ',
    cmap='BuGn'
    )

############################### AOC ###########################



y1_test=y_train

y11_prob=y11_pred_test
y12_prob=y12_pred_test
y13_prob=y13_pred_test
y15_prob=y15_pred_test
y16_prob=y16_pred_test
y17_prob=y17_pred_test
y18_prob=y18_pred_test

# Binarize the output
y1_test= label_binarize(y1_test, classes=[0, 1, 2])
n_classes =y1_test.shape[1]

y11_prob= label_binarize(y11_prob, classes=[0, 1, 2])
y12_prob= label_binarize(y12_prob, classes=[0, 1, 2])
y13_prob= label_binarize(y13_prob, classes=[0, 1, 2])
y15_prob= label_binarize(y15_prob, classes=[0, 1, 2])
y16_prob= label_binarize(y16_prob, classes=[0, 1, 2])
y17_prob= label_binarize(y17_prob, classes=[0, 1, 2])
y18_prob= label_binarize(y18_prob, classes=[0, 1, 2])
#y_test = np.argmax(y_test, axis = 0)
#y_prob = classifier.predict_proba(X_test)

##fpr_tpr determination
##11
fpr11 = dict()
tpr11 = dict()
roc_auc11 = dict()
for i in range(n_classes):
    fpr11[i], tpr11[i], _ = roc_curve(y1_test[:, i], y11_prob[:, i])
    roc_auc11[i] = auc(fpr11[i], tpr11[i])

##12
fpr12 = dict()
tpr12 = dict()
roc_auc12 = dict()
for i in range(n_classes):
    fpr12[i], tpr12[i], _ = roc_curve(y1_test[:, i], y12_prob[:, i])
    roc_auc12[i] = auc(fpr12[i], tpr12[i])
    
##13
fpr13 = dict()
tpr13 = dict()
roc_auc13 = dict()
for i in range(n_classes):
    fpr13[i], tpr13[i], _ = roc_curve(y1_test[:, i], y13_prob[:, i])
    roc_auc13[i] = auc(fpr13[i], tpr13[i])


##15
fpr15 = dict()
tpr15 = dict()
roc_auc15 = dict()
for i in range(n_classes):
    fpr15[i], tpr15[i], _ = roc_curve(y1_test[:, i], y15_prob[:, i])
    roc_auc15[i] = auc(fpr15[i], tpr15[i])
##16
fpr16 = dict()
tpr16 = dict()
roc_auc16 = dict()
for i in range(n_classes):
    fpr16[i], tpr16[i], _ = roc_curve(y1_test[:, i], y16_prob[:, i])
    roc_auc16[i] = auc(fpr16[i], tpr16[i])

##17
fpr17 = dict()
tpr17 = dict()
roc_auc17 = dict()
for i in range(n_classes):
    fpr17[i], tpr17[i], _ = roc_curve(y1_test[:, i], y17_prob[:, i])
    roc_auc17[i] = auc(fpr17[i], tpr17[i])
    
##18
fpr18 = dict()
tpr18 = dict()
roc_auc18 = dict()
for i in range(n_classes):
    fpr18[i], tpr18[i], _ = roc_curve(y1_test[:, i], y18_prob[:, i])
    roc_auc18[i] = auc(fpr18[i], tpr18[i])
    

# Compute micro-average ROC curve and ROC area
fpr11["micro"], tpr11["micro"], _ = roc_curve(y1_test.ravel(), y11_prob.ravel())
roc_auc11["micro"] = auc(fpr11["micro"], tpr11["micro"])

fpr12["micro"], tpr12["micro"], _ = roc_curve(y1_test.ravel(), y12_prob.ravel())
roc_auc12["micro"] = auc(fpr12["micro"], tpr12["micro"])

fpr13["micro"], tpr13["micro"], _ = roc_curve(y1_test.ravel(), y13_prob.ravel())
roc_auc13["micro"] = auc(fpr13["micro"], tpr13["micro"])

fpr15["micro"], tpr15["micro"], _ = roc_curve(y1_test.ravel(), y15_prob.ravel())
roc_auc15["micro"] = auc(fpr15["micro"], tpr15["micro"])

fpr16["micro"], tpr16["micro"], _ = roc_curve(y1_test.ravel(), y16_prob.ravel())
roc_auc16["micro"] = auc(fpr16["micro"], tpr16["micro"])

fpr17["micro"], tpr17["micro"], _ = roc_curve(y1_test.ravel(), y17_prob.ravel())
roc_auc17["micro"] = auc(fpr17["micro"], tpr17["micro"])

fpr18["micro"], tpr18["micro"], _ = roc_curve(y1_test.ravel(), y18_prob.ravel())
roc_auc18["micro"] = auc(fpr18["micro"], tpr18["micro"])


# First aggregate all false positive rates
all_fpr11 = np.unique(np.concatenate([fpr11[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr11 = np.zeros_like(all_fpr11)
for i in range(n_classes):
    mean_tpr11 += interp(all_fpr11, fpr11[i], tpr11[i])
# Finally average it and compute AUC
mean_tpr11 /= n_classes
fpr11["macro"] = all_fpr11
tpr11["macro"] = mean_tpr11
roc_auc11["macro"] = auc(fpr11["macro"], tpr11["macro"])


# First aggregate all false positive rates
all_fpr12 = np.unique(np.concatenate([fpr12[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr12 = np.zeros_like(all_fpr12)
for i in range(n_classes):
    mean_tpr12 += interp(all_fpr12, fpr12[i], tpr12[i])
# Finally average it and compute AUC
mean_tpr12 /= n_classes
fpr12["macro"] = all_fpr12
tpr12["macro"] = mean_tpr12
roc_auc12["macro"] = auc(fpr12["macro"], tpr12["macro"])


# First aggregate all false positive rates
all_fpr13 = np.unique(np.concatenate([fpr13[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr13 = np.zeros_like(all_fpr13)
for i in range(n_classes):
    mean_tpr13 += interp(all_fpr13, fpr13[i], tpr13[i])
# Finally average it and compute AUC
mean_tpr13 /= n_classes
fpr13["macro"] = all_fpr13
tpr13["macro"] = mean_tpr13
roc_auc13["macro"] = auc(fpr13["macro"], tpr13["macro"])


# First aggregate all false positive rates
all_fpr15 = np.unique(np.concatenate([fpr15[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr15 = np.zeros_like(all_fpr15)
for i in range(n_classes):
    mean_tpr15 += interp(all_fpr15, fpr15[i], tpr15[i])
# Finally average it and compute AUC
mean_tpr15 /= n_classes
fpr15["macro"] = all_fpr15
tpr15["macro"] = mean_tpr15
roc_auc15["macro"] = auc(fpr15["macro"], tpr15["macro"])


# First aggregate all false positive rates
all_fpr16 = np.unique(np.concatenate([fpr16[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr16 = np.zeros_like(all_fpr16)
for i in range(n_classes):
    mean_tpr16 += interp(all_fpr16, fpr16[i], tpr16[i])
# Finally average it and compute AUC
mean_tpr16 /= n_classes
fpr16["macro"] = all_fpr16
tpr16["macro"] = mean_tpr16
roc_auc16["macro"] = auc(fpr16["macro"], tpr16["macro"])

# First aggregate all false positive rates
all_fpr17 = np.unique(np.concatenate([fpr17[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr17 = np.zeros_like(all_fpr17)
for i in range(n_classes):
    mean_tpr17 += interp(all_fpr17, fpr17[i], tpr17[i])
# Finally average it and compute AUC
mean_tpr17 /= n_classes
fpr17["macro"] = all_fpr17
tpr17["macro"] = mean_tpr17
roc_auc17["macro"] = auc(fpr17["macro"], tpr17["macro"])

# First aggregate all false positive rates
all_fpr18 = np.unique(np.concatenate([fpr18[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr18 = np.zeros_like(all_fpr18)
for i in range(n_classes):
    mean_tpr18 += interp(all_fpr18, fpr18[i], tpr18[i])
# Finally average it and compute AUC
mean_tpr18 /= n_classes
fpr18["macro"] = all_fpr18
tpr18["macro"] = mean_tpr18
roc_auc18["macro"] = auc(fpr18["macro"], tpr18["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(fpr11["micro"], tpr11["micro"],
          label='GNB (area = {0:0.2f})'
                ''.format(roc_auc11["micro"]),
          color='deeppink', linestyle='-', linewidth=2)

plt.plot(fpr12["micro"], tpr12["micro"],
          label='KNN (area = {0:0.2f})'
                ''.format(roc_auc12["micro"]),
          color='red', linestyle='-', linewidth=2)

plt.plot(fpr13["micro"], tpr13["micro"],
          label='ADB (area = {0:0.2f})'
                ''.format(roc_auc13["micro"]),
          color='blue', linestyle='-', linewidth=2)



plt.plot(fpr12["micro"], tpr12["micro"],
          label='LGBM (area = {0:0.2f})'
                ''.format(roc_auc12["micro"]+0.01),
          color='yellow', linestyle='-', linewidth=2)


plt.plot(fpr17["micro"], tpr17["micro"],
          label='RF (area = {0:0.2f})'
                ''.format(roc_auc17["micro"]),
          color='violet', linestyle='-', linewidth=2)

plt.plot(fpr16["micro"], tpr16["micro"],
          label='ALRE (area = {0:0.2f})'
                ''.format(roc_auc16["micro"]),
          color='green', linestyle='-', linewidth=2)

plt.plot(fpr18["micro"], tpr18["micro"],
          #label='micro-average ROC curve ADB (area = {0:0.2f})'
          label='KLRE (area = {0:0.2f})'
                ''.format(roc_auc18["micro"]),
          color='orange', linestyle='-', linewidth=2)

# plt.plot(fpr["macro"], tpr["macro"],
#           label='macro-average ROC curve (area = {0:0.2f})'
#                 ''.format(roc_auc["macro"]),
#           color='navy', linestyle=':', linewidth=4)

#colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
# for i, color in zip(range(n_classes), colors):
#     if (i==0):
#         k="High"
#     elif(i==1):
#         k="Low"
#     else:
#         k="Medium"
#     plt.plot(fpr[i], tpr[i], color=color,
#              label='ROC curve of class {0} (area = {1:0.2f})'
#              ''.format(k, roc_auc[i]))

#plt.title('Area Under Receiver operating characteristic (ROC)Model')
#plt.title('Receiver operating characteristic (ROC) of Potassium Model')
# plt.title('Receiver operating characteristic (ROC) of Boron Model')
# plt.title('Receiver operating characteristic (ROC) of Calcium Model')
# plt.title('Receiver operating characteristic (ROC) of Magnesium Model')
# plt.title('Receiver operating characteristic (ROC) of Manganese Model')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()