from sklearn import svm
import pandas as pd
import numpy as np
import pprint

data = np.genfromtxt('data.csv', delimiter=',')
feature_array = np.delete(data, 0, 1)
class_array = np.genfromtxt('data.csv', delimiter=',', usecols=([0]))

pprint.pprint(feature_array)
pprint.pprint(class_array)

classifier = svm.SVC(verbose=True)
classifier.fit(feature_array, class_array)
