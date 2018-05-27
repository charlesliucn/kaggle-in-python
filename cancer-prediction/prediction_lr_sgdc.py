# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np

# input data
column_names = []
with open("column_names.txt","r") as f:
	lines = f.readlines()
	for line in lines:
		name = line.strip()
		column_names.append(name)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
data = pd.read_csv(url, names = column_names)

# data preprocessing
## 将?替换为缺失值nan
data = data.replace(to_replace = "?", value = np.nan)
## 去除所有带有缺失值的数据
data = data.dropna(how = "any")

# prepare the train and test data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
	data[column_names[1:10]],
	data[column_names[10]],
	test_size = 0.25,
	random_state = 33)
## 查看分割好的训练集和测试集
print(y_train.value_counts())
print(y_test.value_counts())

# 训练/测试模型
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
## 数据的标准化/归一化，均值为0，方差为1
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

## Logistic回归预测
lr = LogisticRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)

## 随机梯度分类预测
sgdc = SGDClassifier()
sgdc.fit(x_train, y_train)
sgdc_y_predict = sgdc.predict(x_test)


# 分析模型结果和预测效果
from sklearn.metrics import classification_report

res1 = classification_report(y_test, lr_y_predict, target_names = ["Benign", "Malignant"])
print("Logistic Regression: %s" % lr.score(x_test, y_test))
print(res1)
res2 = classification_report(y_test, sgdc_y_predict, target_names = ["Benign", "Malignant"])
print("SGD Classifier: %s" % sgdc.score(x_test, y_test))
print(res2)