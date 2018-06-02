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

# 线性回归预测
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_y_predict = lr.predict(x_test)
lr_score = lr.score(x_test, y_test)
print("Linear Regression: %s" % lr_score)

# 随机梯度下降回归预测
from sklearn.linear_model import SGDRegressor
sgdr = SGDRegressor()
sgdr.fit(x_train, y_train)
sgdr_y_predict = sgdr.predict(x_test)
sgdr_score = sgdr.score(x_test, y_test)
print("SGD Regressor: %s" % sgdr_score)

# 比较两种回归模型性能
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
y_test_inv = ss_y.inverse_transform(y_test)
## Linear Regression
lr_r2 = r2_score(y_test, lr_y_predict)
lr_y_inv = ss_y.inverse_transform(lr_y_predict)
lr_mse = mean_squared_error(y_test_inv, lr_y_inv)
lr_mae = mean_absolute_error(y_test_inv, lr_y_inv)
## SGDRegressor
sgdr_r2 = r2_score(y_test, sgdr_y_predict)
sgdr_y_inv = ss_y.inverse_transform(sgdr_y_predict)
sgdr_mse = mean_squared_error(y_test_inv, sgdr_y_inv)
sgdr_mae = mean_absolute_error(y_test_inv, sgdr_y_inv)
## 打印模型比较结果
print("Linear Regression")
print("R2: %s" % lr_r2)
print("MSE: %s" % lr_mse)
print("MAE: %s" % lr_mae)
print("SGD Regressor")
print("R2: %s" % sgdr_r2)
print("MSE: %s" % sgdr_mse)
print("MAE: %s" % sgdr_mae)

