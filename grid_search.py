import pandas as pd
import numpy as np
import pprint

from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier


model_type  = "adaboost" # adaboost / svm / ...

features = [
"verified",
"followers_count",
"friends_count",
"listed_count",
"statuses_count",
"contributors_enabled",
"created_at",
"geo_enabled",
"hashtag_count",
"hashtag_avg",
"symbols_count",
"symbols_avg",
"urls_count",
"urls_avg",
"user_mentions_count",
"user_mentions_avg",
"favorite_count",
"favorite_avg",
"retweet_count",
"retweet_avg",
]

sorted_features  = {}

class_array = np.genfromtxt('data.csv', delimiter=',', usecols=([0]))

for column in range( 1, 20 ) :
  print features[column]
  # get only one column of the training data
  feature_array  = np.genfromtxt('data.csv', delimiter=',', usecols=([0,column]))
  feature_array = np.delete(feature_array, 0, 1)
  
  pprint.pprint( feature_array )
  
  if model_type == "svm" :
    clf = svm.SVC( kernel="poly", C=10, degree=3, verbose=True ).fit(feature_array, class_array)
  if model_type == "adaboost" :
    clf = AdaBoostClassifier(n_estimators=10)

  accuracy =  np.mean(cross_validation.cross_val_score(clf, feature_array, class_array, cv=10)) 
  sorted_features[features[column]] = accuracy
  pprint.pprint( accuracy )

for key in sorted_features.keys():
  print str(key) + " " + str(sorted_features[key])
  