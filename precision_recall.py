from classification import classify
import pylab as pl
from sklearn.metrics import classification_report
from sklearn.metrics import auc
import time


names = ["svm", "adaboost", "random_forest", "decision_tree", "nearest_centroid"]
for algorithm in names:
  print (algorithm)
  t = time.time()
  test_true, test_predict = classify(algorithm, True, True)
  run_time = time.time() - t

  print(classification_report(test_true, test_predict))
  print "Run time: " + str(run_time)
