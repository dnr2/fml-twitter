import pandas as pd
import numpy as np
import pprint

from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier


model_type  = "adaboost" #the ML algorithm model: adaboost / svm / ...
cv_folds = 10 #number of cross validation folds
use_CV = False #use cross validation

training_data = np.genfromtxt('traindata.csv', delimiter=',')
testing_data = np.genfromtxt('testdata.csv', delimiter=',')

training_X = training_data[:,1:19]
training_Y = training_data[:,0]

testing_X = testing_data[:,1:19]
testing_Y = testing_data[:,0]

if model_type == "svm" :
  clf = svm.SVC( kernel="poly", C=10, degree=3, verbose=True )
if model_type == "adaboost" :
  clf = AdaBoostClassifier(n_estimators=200)
  
if use_CV :
  pprint.pprint( np.mean(cross_validation.cross_val_score(clf, training_X, training_Y, cv=cv_folds)) )
else :
  pprint.pprint( clf.fit(training_X, training_Y).score(testing_X , testing_Y) )
  

