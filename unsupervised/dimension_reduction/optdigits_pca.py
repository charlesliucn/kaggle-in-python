# -*- coding:utf-8 -*-

# 导入数据
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

# PCA选择主成分(2维)
from sklearn.decomposition import PCA
estimator = PCA(n_components = 2)
x_pca = estimator.fit_transform(x_train)

# 作图展示二维图
colors = ["black","blue","purple","yellow","darkblue",
	"red","lime","cyan","orange","gray"]
for i in range(len(colors)):
	px = x_pca[:,0][y_train.as_matrix() == i]
	py = x_pca[:,1][y_train.as_matrix() == i]
	plt.scatter(px, py, color = colors[i])
plt.legend(np.arange(10).astype(str))
plt.xlabel("First Principle Component")
plt.ylabel("Second Principle Component")
plt.show()
