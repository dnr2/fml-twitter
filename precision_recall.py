from classification import classify
import pylab as pl
from sklearn.metrics import classification_report
from sklearn.metrics import auc


names = ["svm", "adaboost", "stochastic_gradient_descent", "nearestneighbor", "decision_tree"]
for algorithm in names:
  print (algorithim)
  test_true, test_predict = classify(algorithm, True)

  print(classification_report(test_true, test_predict))
