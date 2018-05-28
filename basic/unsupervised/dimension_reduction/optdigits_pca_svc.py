# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd

# 导入数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
train_url = base_url + "optdigits.tra"
test_url = base_url + "optdigits.tes"
digits_train = pd.read_csv(train_url, header = None)
digits_test = pd.read_csv(test_url, header = None)

# 训练集和测试集
x_train = digits_train[np.arange(64)]
y_train = digits_train[64]
x_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 线性支持向量机分类
from sklearn.svm import LinearSVC
svc = LinearSVC()
svc.fit(x_train, y_train)
svc_y_predict = svc.predict(x_test)
svc_score = svc.score(x_test, y_test)

# PCA之后再用SVC分类
from sklearn.decomposition import PCA
pca = PCA(n_components = 30)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)
svc_pca = LinearSVC()
svc_pca.fit(x_train_pca, y_train)
svc_pca_y_predict = svc_pca.predict(x_test_pca)
svc_pca_score = svc_pca.score(x_test_pca, y_test)

# 评价分类效果
from sklearn.metrics import classification_report
res_svc = classification_report(y_test, svc_y_predict,
	target_names = np.arange(10).astype(str))
res_svc_pca = classification_report(y_test, svc_pca_y_predict,
	target_names = np.arange(10).astype(str))

print("Linear SVC with raw features:")
print("score: %s " % svc_score)
print(res_svc)

print("Linear SVC PCA:")
print("score: %s " % svc_pca_score)
print(res_svc_pca)