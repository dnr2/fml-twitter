from classification import classify
import pylab as pl
from sklearn.metrics import classification_report
from sklearn.metrics import auc


names = ["svm", "adaboost", "random_forest", "decision_tree", "nearest_centroid"]
for algorithm in names:
  print (algorithm)
  test_true, test_predict = classify(name = algorithm, pr= True, use_CV=True, use_nlp=True, use_tfidf=False)
  print(classification_report(test_true, test_predict))
