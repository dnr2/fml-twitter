import pandas as pd
import numpy as np
import pprint

from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier


model_type  = "adaboost" # adaboost / svm / ...

data = np.genfromtxt('data.csv', delimiter=',')
feature_array = np.delete(data, 0, 1)
class_array = np.genfromtxt('data.csv', delimiter=',', usecols=([0]))

if model_type == "svm" :
  clf = svm.SVC( kernel="poly", C=10, degree=3, verbose=True ).fit(feature_array, class_array)
if model_type == "adaboost" :
  clf = AdaBoostClassifier(n_estimators=200)

pprint.pprint( np.mean(cross_validation.cross_val_score(clf, feature_array, class_array, cv=10)) )


