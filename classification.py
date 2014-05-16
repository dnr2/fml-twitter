import pandas as pd
import numpy as np
import pylab as pl
import pprint
import csv
import sqlite3
import math

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

#possible names = "svm", "adaboost" , "stochastic_gradient_descent" , "nearestneighbor", "decision_tree" 
# names = ["adaboost","stochastic_gradient_descent","nearestneighbor","decision_tree"] #classifiers used for cross validation
names = ["adaboost"] #classifiers used for cross validation
name = "adaboost" #classifier used for testing
cv_folds = 5 #number of cross validation folds
use_CV = False #use cross validation
create_PR = False #Precision Recall
save_Classifer = False 
use_nlp = True #use natural language processing analysis
num_nlp_columns = 3

training_data = np.genfromtxt('traindata.csv', delimiter=',')
np.random.shuffle(training_data)
testing_data = np.genfromtxt('testdata.csv', delimiter=',')

training_X = training_data[:,1:]
training_Y = training_data[:,0]

testing_X = testing_data[:,1:]
testing_Y = testing_data[:,0]

#add nlp features (under construction)
if use_nlp :

  #TODO using CV while I don't split nlp data into training and testing, must change that later
  use_CV = True  
  training_data  = np.array( pd.read_csv( 'data_nlp.csv', sep=',', quotechar="\"", na_values="nan", keep_default_na=False ))

  training_X = training_data[:,(num_nlp_columns+1):].astype(float)
  training_Y = training_data[:,0].astype(float)  
  
  for col in range( 1, num_nlp_columns + 1 ) :    
    nlp_features = training_data[:,col]
    # pprint.pprint( nlp_features )
    for line in range(0,nlp_features.shape[0] ) :
      if isinstance( nlp_features[line], float) :
        pprint.pprint( line )
        pprint.pprint( nlp_features[line] )
    
    ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1,2), min_df=1)
    counts = ngram_vectorizer.fit_transform( nlp_features )
    new_features = np.array(counts.toarray()).astype(float)    
    if col == 1 :
      training_X = new_features
    else :
      training_X = np.concatenate( (training_X , new_features ), axis = 1)
    pprint.pprint( training_X.shape )  
  
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
