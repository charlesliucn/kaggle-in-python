# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
train_url = base_url + "optdigits.tra"
test_url = base_url + "optdigits.tes"

digits_train = pd.read_csv(train_url, header = None)
digits_test = pd.read_csv(test_url, header = None)

x_train = digits_train[np.arange(64)]
y_train = digits_train[64]
x_test = digits_test[np.arange(64)]
y_test = digits_test[64]


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 10)
kmeans.fit(x_train)
y_pred = kmeans.predict(x_test)

from sklearn import metrics
km_score = metrics.adjusted_rand_score(y_test, y_pred)

print("km_score: %s" % km_score)
