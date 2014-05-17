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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def classify(name, pr):
  #possible names = "svm", "adaboost" , "stochastic_gradient_descent" , "nearestneighbor", "decision_tree"
  # names = ["adaboost","stochastic_gradient_descent","nearestneighbor","decision_tree"] #classifiers used for cross validation
  names = ["adaboost"] #classifiers used for cross validation
  #name = "adaboost" #classifier used for testing
  cv_folds = 5 #number of cross validation folds
  use_CV = False #use cross validation
  save_Classifer = False

  use_nlp = True #use natural language processing analysis
  num_nlp_columns = 3
  use_tfidf = False #use natural language term-frequency inverse document-frequency feature

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
    #do not shuffle the data...
    use_CV = True
    training_data  = np.array( pd.read_csv( 'data_nlp.csv', sep=',', quotechar="\"", na_values="nan", keep_default_na=False ))

    training_X = training_data[:,(num_nlp_columns+1):].astype(float)
    training_Y = training_data[:,0].astype(float)

    for col in range( 1, num_nlp_columns + 1 ):
      nlp_features = training_data[:,col]
      ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1,3), min_df=1)
      # ngram_vectorizer = CountVectorizer(analyzer='word', min_df=1)
      counts = ngram_vectorizer.fit_transform( nlp_features )
      new_features = np.array(counts.toarray()).astype(float)
      if col == 1 :
        training_X = new_features
      else :
        training_X = np.concatenate( (training_X , new_features ), axis = 1)
      pprint.pprint( training_X.shape )

    if use_tfidf :
      transformer = TfidfTransformer()
      training_X = transformer.fit_transform(training_X).toarray()
      pprint.pprint( training_X.shape )

  #use cross validation and grid search
  if use_CV :
    print 'Cross validation table'
    a = {}
    for name in names :
      if name == "svm" :
        clf = svm.SVC( kernel="linear", C=10, degree=3, verbose=True )
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

  #use test data
  else:
    if name == "svm" :
      clf = svm.SVC( C=10, degree=3)
    if name == "adaboost" :
      clf = AdaBoostClassifier(n_estimators=200)
    if name == "stochastic_gradient_descent" :
      clf = SGDClassifier(loss="hinge", penalty="l2")
    if name == "nearestneighbor" :
      clf = NearestCentroid()
    if name == "decision_tree" :
      clf = tree.DecisionTreeClassifier()
    pprint.pprint( clf.fit(training_X, training_Y).score(testing_X , testing_Y) )
    if(pr == True):
      y_true, y_pred = testing_Y, clf.fit(training_X, training_Y).predict(testing_X)
      return y_true, y_pred

  if save_Classifer:
    joblib.dump(clf, name+'_classifier.pkl', compress=3)
