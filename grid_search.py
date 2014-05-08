import pandas as pd
import numpy as np
import pprint

from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier

'''

RESULTS:

feature #10 symbols_count         0.578902903781
feature #5  contributors_enabled  0.57923847425
feature #7  geo_enabled           0.57923847425
feature #11 symbols_avg           0.57923847425
feature #13 urls_avg              0.57923847425
feature #15 user_mentions_avg     0.57923847425
feature #2  friends_count         0.585778163749
feature #14 user_mentions_count   0.610936674424
feature #4  statuses_count        0.672314031005
feature #6  created_at            0.678515339561
feature #8  hashtag_count         0.703330410441
feature #9  hashtag_avg           0.722449214754
feature #12 urls_count            0.740568895934
feature #18 retweet_count         0.747113925331
feature #19 retweet_avg           0.750131249087
feature #1  followers_count       0.765052893101
feature #16 favorite_count        0.768734893708
feature #17 favorite_avg          0.78651254033
feature #3  listed_count          0.869025215563
'''


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

sorted_features  = []

training_data = np.genfromtxt('data.csv', delimiter=',')
np.random.shuffle(training_data)

for column in range( 1, 20 ) :
  print features[column]
  # get only one column of the training data
  feature_array  = training_data[:,[column]]
  class_array = training_data[:,0]

  if model_type == "svm" :
    clf = svm.SVC( kernel="poly", C=10, degree=3, verbose=True ).fit(feature_array, class_array)
  if model_type == "adaboost" :
    clf = AdaBoostClassifier(n_estimators=200)

  accuracy =  np.mean(cross_validation.cross_val_score(clf, feature_array, class_array, cv=10))
  sorted_features.append( ( accuracy, features[column] , column) );
  pprint.pprint( accuracy )


sorted_features.sort(key=lambda tup: tup[0])

for tup in sorted_features:
  print "feature #" + str(tup[2]) + " " + str(tup[1]) + "\t" + str(tup[0])
