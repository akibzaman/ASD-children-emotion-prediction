# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 12:01:02 2022

@author: Akib Zaman
"""

import pandas as pd
from yellowbrick.target import FeatureCorrelation
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from matplotlib import pyplot
##Text
# ##Text-100-raw
#data = pd.read_csv("alldata/dataset.csv")
data = pd.read_csv("alldata/clean-original/clean-dataset.csv")
 

print(data.dtypes)


# # ##Text-100-moderate
# data = pd.read_csv("text-data/text-100-moderate.csv")
# train_feature=[ 'home_liwc','i_liwc','deception_emp','furniture_emp',
#                'time_liwc','childish_emp','wedding_emp','number_liwc','article_liwc',
#                'pride_emp','WC_liwc','zest_emp','wealthy_emp','irritability_emp',
#                'clothing_emp','auxverb_liwc','horror_emp','future_liwc',
#                'real_estate_emp','economics_emp','exasperation_emp','fabric_emp',
#                'shopping_emp','car_emp','anticipation_emp','insight_liwc']

# ##Image
# data = pd.read_csv("image-data/image-all.csv")
# train_feature=['avg_red','sd_red','avg_green','sd_green','avg_blue','sd_blue','avg_rgb',
#                 'sd_rgb','avg_gray','sd_gray','avg_hue','sd_hue','avg_sat','sd_sat','avg_brght',
#                 'sd_brght','No. Faces','age','gender','angry','disgust','fear','happy','sad',
#                 'surprise','neutral','asian','indian','black','white','middle eastern',
#                 'latino hispanic']

labelencoder = LabelEncoder()
data['Status']=labelencoder.fit_transform(data['Status'])
X, y = data.loc[:,"eegRawValue":"GamaMid"], data['Status']
nunique = data.nunique()
cols_to_drop = nunique[nunique == 1].index
data.drop(cols_to_drop, axis=1)
feature_names = X.columns.values.tolist()
#########Normalization:
# scaler=MinMaxScaler()
# #scaler=StandardScaler()
# scaler.fit(X)
# X=scaler.transform(X)



#######################BORUTAPY-FEATURE-RANKING##########################

# #model=xgb.XGBClassifier()
# model = RandomForestClassifier(n_estimators = 100, random_state=30)

# # define Boruta feature selection method
# feat_selector = BorutaPy(model, n_estimators='auto', verbose=2, random_state=1)

# # find all relevant features
# feat_selector.fit(X, y)

# # check selected features
# print(feat_selector.support_)

# # check ranking of features
# print(feat_selector.ranking_)

# # zip feature names, ranks, and decisions 
# feature_ranks = list(zip(feature_names, 
#                          feat_selector.ranking_, 
#                          feat_selector.support_))
# ranked_data = pd.DataFrame (feature_ranks, columns = ['feature_name','rank','support'])
# # print the results
# for feat in feature_ranks:
#     print('Feature: {:<30} Rank: {},  Keep: {}'.format(feat[0], feat[1], feat[2]))


# # Create a list of the feature names
# features = np.array(data['avg_red'])

##### Instantiate the visualizer
# visualizer = FeatureCorrelation(labels=None)

# visualizer.fit(X, y)        # Fit the data to the visualizer
# visualizer.show()           # Finalize and render the figure



##################################FEATURE RANKING########################

######################  ANOVA f-test Feature Selection  ##############
fs = SelectKBest(score_func=f_classif, k='all')
fs.fit(X, y)
# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))
    
# what are scores for the features
selected_feature = []
for i in range(len(fs.pvalues_)):
    if(fs.pvalues_[i]<0.05):
        selected_feature.append(fs.feature_names_in_[i])
        print('Feature %d : %f' % (i,fs.pvalues_[i]),fs.feature_names_in_[i])
#print(selected_feature)
data_selected = data [selected_feature]
# data = pd.read_csv("alldata/dataset.csv")
data = pd.read_csv("alldata/clean-original/clean-dataset.csv")
y = data['Status']
data_frame=[data_selected, y]
data_selected=pd.concat(data_frame, axis=1)
data_selected.to_csv("alldata/clean-mean/selected-clean-dataset.csv")
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()






# # ##Mutual-Information Classification

# visualizer = FeatureCorrelation(
#     method='mutual_info-classification', feature_names=None, sort=True)

# visualizer.fit(X, y)     # Fit the data to the visualizer
# visualizer.show()   


# from yellowbrick.features import JointPlotVisualizer
# visualizer = JointPlotVisualizer(columns="cement")

# visualizer.fit_transform(X, y)        # Fit and transform the data
# visualizer.show()


# from yellowbrick.features import Rank1D
# visualizer = Rank1D(algorithm='shapiro')

# visualizer.fit(X, y)           # Fit the data to the visualizer
# a=visualizer.transform(X)        # Transform the data
# visualizer.show()  
















# #############https://www.scikit-yb.org/en/latest/api/target/feature_correlation.html