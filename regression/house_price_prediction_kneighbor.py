# -*- coding:utf-8 -*-

# 导入数据
import pandas as pd
from sklearn.datasets import load_boston
boston = load_boston()
X = boston.data
y = boston.target
# print(boston.DESCR)

# 分割成训练集和测试集
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
	X, y, test_size = 0.25, random_state = 33)

# 预处理，进行标准化
from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
ss_y = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train.reshape(-1, 1))
y_test = ss_y.transform(y_test.reshape(-1, 1))

# K近邻方法回归预测
from sklearn.neighbors import KNeighborsRegressor
## 平均回归
uni_knr = KNeighborsRegressor(weights = "uniform")
uni_knr.fit(x_train, y_train)
uni_knr_y_predict = uni_knr.predict(x_test)
## 距离加权回归
dis_knr = KNeighborsRegressor(weights = "distance")
dis_knr.fit(x_train, y_train)
dis_knr_y_predict = dis_knr.predict(x_test)

# 评价K近邻方法
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
## score
uni_knr_score = uni_knr.score(x_test, y_test)
dis_knr_score = dis_knr.score(x_test, y_test)
## 平均回归
y_test = ss_y.inverse_transform(y_test)
uni_knr_y_predict = ss_y.inverse_transform(uni_knr_y_predict)
uni_knr_r2 = r2_score(y_test, uni_knr_y_predict)
uni_knr_mse = mean_squared_error(y_test, uni_knr_y_predict)
uni_knr_mae = mean_absolute_error(y_test, uni_knr_y_predict)
## 距离加权回归
dis_knr_y_predict = ss_y.inverse_transform(dis_knr_y_predict)
dis_knr_r2 = r2_score(y_test, dis_knr_y_predict)
dis_knr_mse = mean_squared_error(y_test, dis_knr_y_predict)
dis_knr_mae = mean_absolute_error(y_test, dis_knr_y_predict)

# 打印出结果
print("KNeighbors Uniform平均回归：")
print("	score: %s" % uni_knr_score)
print("	r2: %s" % uni_knr_r2)
print("	mse: %s" % uni_knr_mse)
print("	mae: %s" % uni_knr_mae)

print("KNeighbors Distance距离加权回归：")
print("	score: %s" % dis_knr_score)
print("	r2: %s" % dis_knr_r2)
print("	mse: %s" % dis_knr_mse)
print("	mae: %s" % dis_knr_mae)
