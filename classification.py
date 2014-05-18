import pandas as pd
import numpy as np
import pylab as pl
import pprint
import csv
import sqlite3
import math

from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
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
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def extract_nlp( nlp_columns, data , use_tfidf) :
  for col in nlp_columns:
    nlp_features = data[:,col]
    ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(1,3), min_df=3)
    counts = ngram_vectorizer.fit_transform( nlp_features )
    new_features = np.array(counts.toarray()).astype(float)
    if col == 1 :
      extracted = new_features
    else :
      extracted = np.concatenate( (extracted , new_features ), axis = 1)
    pprint.pprint( extracted.shape )

  if use_tfidf :
    transformer = TfidfTransformer()
    extracted = transformer.fit_transform(extracted).toarray()
    pprint.pprint( extracted.shape )
  return extracted


def classify(name, pr, use_CV):
  #possible names = "svm", "adaboost" , "stochastic_gradient_descent" , "nearestneighbor", "decision_tree"
  names = ["adaboost"] #classifiers used for cross validation
  cv_folds = 10 #number of cross validation folds
  save_Classifer = False

  use_nlp = True #use natural language processing analysis
  num_nlp_columns = 3
  use_tfidf = False #use natural language term-frequency inverse document-frequency feature

  training_data = np.array(pd.read_csv('data_train.csv', sep=',', quotechar='"', na_values="nan", keep_default_na=False))
  testing_data = np.array(pd.read_csv('data_test.csv', sep=',', quotechar='"', na_values="nan", keep_default_na=False))

  training_X = training_data[:,num_nlp_columns+1:].astype(float)
  training_Y = training_data[:,0].astype(float)

  testing_X = testing_data[:,num_nlp_columns+1:].astype(float)
  testing_Y = testing_data[:,0].astype(float)

  #add nlp features (under construction)
  if use_nlp :
    training_X = extract_nlp( range(1,num_nlp_columns+1), training_data, use_tfidf)
    testing_X = extract_nlp( range(1,num_nlp_columns+1), testing_data, use_tfidf)

  #use cross validation and grid search
  if use_CV :
    print 'Using Cross Validation'
    ##Choose classifier
    #linear SVC
    if name == "svm" :
      clf = svm.SVC(kernel='linear', degree=3, cache_size=1000)
    #Naive Bayes
    if name == "nearest_centroid":
      clf = GaussianNB()
    #Ensemble Methods
    if name == "adaboost" :
      clf = AdaBoostClassifier(n_estimators=100)
    if name == "random_forest" :
      clf = RandomForestClassifier(n_estimators=100)
    if name == "decision_tree" :
      clf = tree.DecisionTreeClassifier()

    scaler = preprocessing.StandardScaler()
    training_X = scaler.fit_transform(training_X)
    testing_X = scaler.transform(testing_X)

    pprint.pprint( clf.fit(training_X, training_Y).score(testing_X, testing_Y) )
    if not name == "svm" and not name =="nearest_neighbor" and not name == "nearest_centroid":
      print clf.feature_importances_
    if(pr == True):
      y_true, y_pred = testing_Y, clf.fit(training_X, training_Y).predict(testing_X)
      return y_true, y_pred



  #use test data
  else:
    print 'Using Test Validation'
    ##Choose classifier
    #linear SVC
    if name == "svm" :
      clf = svm.SVC(kernel='linear', degree=3, cache_size=1000)
    #KNeighbors
    if name == "nearest_centroid":
      clf = NearestCentroid()
    #Ensemble Methods
    if name == "adaboost" :
      clf = AdaBoostClassifier(n_estimators=100)
    if name == "random_forest" :
      clf = RandomForestClassifier(n_estimators=100)
    if name == "decision_tree" :
      clf = tree.DecisionTreeClassifier()

    scaler = preprocessing.StandardScaler()
    training_X = scaler.fit_transform(training_X)
    testing_X = scaler.transform(testing_X)

    np.mean(cross_validation.cross_val_score(clf, training_X, training_Y, cv=cv_folds))
    if(pr == True):
      y_true, y_pred = testing_Y, clf.fit(training_X, training_Y).predict(testing_X)
      return y_true, y_pred

  if save_Classifer:
    joblib.dump(clf, name+'_classifier.pkl', compress=3)
