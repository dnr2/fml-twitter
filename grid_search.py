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
"symbols_count",
"urls_count",
"user_mentions_count",
"favorite_count",
"retweet_count",
"hashtag_avg",
"symbols_avg",
"urls_avg",
"user_mentions_avg",
"favorite_avg",
"retweet_avg",
]



for column in range( 1, 19 ) :
  data = np.genfromtxt('train.csv', delimiter=',')
  feature_array = np.delete(data, 0, 1)
  pprint.pprint(feature_array)
  feature_array = np.delete(feature_array, 0, column)
  pprint.pprint(feature_array)
  class_array = np.genfromtxt('train.csv', delimiter=',', usecols=([0]))

  if model_type == "svm" :
    clf = svm.SVC( kernel="poly", C=10, degree=3, verbose=True ).fit(feature_array, class_array)
  if model_type == "adaboost" :
    clf = AdaBoostClassifier(n_estimators=200)

  print features[column]
  pprint.pprint( np.mean(cross_validation.cross_val_score(clf, feature_array, class_array, cv=10)) )
