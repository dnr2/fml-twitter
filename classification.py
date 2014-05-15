import pandas as pd
import numpy as np
import pylab as pl
import pprint
import csv
import sqlite3

from sklearn import svm
from sklearn import cross_validation
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn import tree
from sklearn import svm, datasets
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

#possible names = "adaboost" , "stochastic_gradient_descent" , "nearestneighbor", "decision_tree" 
names = ["adaboost","stochastic_gradient_descent","nearestneighbor","decision_tree"] #classifiers used for cross validation
name = "adaboost" #classifier used for testing
cv_folds = 10 #number of cross validation folds
use_CV = False #use cross validation
create_PR = False #Precision Recall
save_Classifer = True 
use_nlp = False #use natural language processing analysis

training_data = np.genfromtxt('traindata.csv', delimiter=',')
np.random.shuffle(training_data)
testing_data = np.genfromtxt('testdata.csv', delimiter=',')

training_X = training_data[:,1:19]
training_Y = training_data[:,0]

testing_X = testing_data[:,1:19]
testing_Y = testing_data[:,0]


#add nlp features (under construction)
if use_nlp : 
  nlp_features_columns = [8]
  #TODO change that 
  use_CV = True
  deletethis = pd.read_table('data_nlp.csv', delimiter=',', dtype=None)
  pprint.pprint( deletethis )
  training_data = np.array( deletethis )
  pprint.pprint( training_data )
  np.random.shuffle(training_data)
  training_X = training_data[:,nlp_features_columns]
  training_Y = training_data[:,0]
  pprint.pprint( training_X )
  ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1, 2), min_df=1)
  counts = ngram_vectorizer.fit_transform(training_X[:,0] )
  pprint.pprint( ngram_vectorizer.get_feature_names() )
  training_X = counts.toarray()

if use_CV :
  print 'Cross validation table'
  a = {}
  for name in names :
    if name == "svm" :
      clf = svm.SVC( kernel="poly", C=10, degree=3, verbose=True )
    if name == "adaboost" :
      clf = AdaBoostClassifier(n_estimators=200)
    if name == "stochastic_gradient_descent" :
      clf = SGDClassifier(loss="hinge", penalty="l2")
    if name == "nearestneighbor" :
      clf = NearestCentroid()
    if name == "decision_tree" :
      clf = tree.DecisionTreeClassifier()

    a[name] = [np.mean(cross_validation.cross_val_score(clf, training_X, training_Y, cv=cv_folds))]
    print name,a[name]

else:
  if name == "svm" :
    clf = svm.SVC( kernel="poly", C=10, degree=3, verbose=True )
  if name == "adaboost" :
    clf = AdaBoostClassifier(n_estimators=200)
  if name == "stochastic_gradient_descent" :
    clf = SGDClassifier(loss="hinge", penalty="l2")
  if name == "nearestneighbor" :
    clf = NearestCentroid()
  if name == "decision_tree" :
    clf = tree.DecisionTreeClassifier()
  pprint.pprint( clf.fit(training_X, training_Y).score(testing_X , testing_Y) )
  
if save_Classifer:
  joblib.dump(clf, name+'_classifier.pkl', compress=3)

if create_PR:
  # Run classifier
  probas_ = clf.fit(training_X, training_Y).predict_proba(testing_X)

  # Compute Precision-Recall and plot curve
  precision, recall, thresholds = precision_recall_curve(testing_Y, probas_[:, 1])
  area = auc(recall, precision)
  print("Area Under Curve: %0.2f" % area)

  pl.clf()
  pl.plot(recall, precision, label='Precision-Recall curve')
  pl.xlabel('Recall')
  pl.ylabel('Precision')
  pl.ylim([0.0, 1.05])
  pl.xlim([0.0, 1.0])
  pl.title('Precision-Recall example: AUC=%0.2f' % area)
  pl.legend(loc="lower left")
  pl.show()
