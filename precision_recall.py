from classification import classify
import pylab as pl
from sklearn.metrics import classification_report
from sklearn.metrics import auc
import time
import pprint

names = ["svm", "adaboost", "random_forest", "decision_tree", "MultinomialNB"]
for algorithm in names:
  for use_nlp in [False,True]:
    for use_tfidf in [False,True]:
      if not use_nlp and use_tfidf :
        continue
      for n_gram in [1,2,3] :      
        if not use_nlp and n_gram > 1 :
          continue
        pprint.pprint( algorithm )
        pprint.pprint( "use_nlp = " + str( use_nlp) )
        pprint.pprint( "use_tfidf = " + str( use_tfidf) )
        pprint.pprint( "n_gram = " + str( n_gram) )
        
        t = time.time()
        test_true, test_predict = classify(name = algorithm, pr= True, use_CV=True, use_nlp=use_nlp, use_tfidf=use_tfidf, n_grams = n_gram, combine_numerical_nlp = False)  
        run_time = time.time() - t
        print(classification_report(test_true, test_predict))
        print "Run time: " + str(run_time)
