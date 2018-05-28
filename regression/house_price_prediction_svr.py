# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

# 导入数据
from sklearn.datasets import load_boston
boston = load_boston()
# print(boston.DESCR)

# 数据分割成训练集和测试集
from sklearn.cross_validation import train_test_split
import numpy as np
X = boston.data
y = boston.target
x_train, x_test, y_train, y_test = train_test_split(
	X, y, test_size = 0.25, random_state = 33)

# 数据的标准化
from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
ss_y = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)
y_train = ss_y.fit_transform(y_train.reshape(-1,1))
y_test = ss_y.transform(y_test.reshape(-1,1))

# SVR: 支持向量回归
from sklearn.svm import SVR
## 线性核
linear_svr = SVR(kernel = "linear")
linear_svr.fit(x_train, y_train)
linear_svr_y_predict = linear_svr.predict(x_test)
linear_svr_score = linear_svr.score(x_test, y_test)
## 多项式核
poly_svr = SVR(kernel = "poly")
poly_svr.fit(x_train, y_train)
poly_svr_y_predict = poly_svr.predict(x_test)
poly_svr_score = poly_svr.score(x_test, y_test)
## RBF核
rbf_svr = SVR(kernel = "rbf")
rbf_svr.fit(x_train, y_train)
rbf_svr_y_predict = rbf_svr.predict(x_test)
rbf_svr_score = rbf_svr.score(x_test, y_test)

# 评价SVR三种核函数的不同效果
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
y_test = ss_y.inverse_transform(y_test)
linear_svr_y_predict = ss_y.inverse_transform(linear_svr_y_predict)
poly_svr_y_predict = ss_y.inverse_transform(poly_svr_y_predict)
rbf_svr_y_predict = ss_y.inverse_transform(rbf_svr_y_predict)

linear_r2 = r2_score(y_test, linear_svr_y_predict)
linear_mse = mean_squared_error(y_test, linear_svr_y_predict)
linear_mae = mean_absolute_error(y_test, linear_svr_y_predict)

poly_r2 = r2_score(y_test, poly_svr_y_predict)
poly_mse = mean_squared_error(y_test, poly_svr_y_predict)
poly_mae = mean_absolute_error(y_test, poly_svr_y_predict)

rbf_r2 = r2_score(y_test, rbf_svr_y_predict)
rbf_mse = mean_squared_error(y_test, rbf_svr_y_predict)
rbf_mae = mean_absolute_error(y_test, rbf_svr_y_predict)

print("Linear SVR:")
print("R2: %s" % linear_r2)
print("MSE: %s" % linear_mse)
print("MAE: %s" % linear_mae)

print("Poly SVR:")
print("R2: %s" % poly_r2)
print("MSE: %s" % poly_mse)
print("MAE: %s" % poly_mae)

print("RBF SVR:")
print("R2: %s" % rbf_r2)
print("MSE: %s" % rbf_mse)
print("MAE: %s" % rbf_mae)