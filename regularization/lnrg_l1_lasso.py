# -*- coding:utf-8 -*-

# 训练数据和测试数据
import numpy as np
x_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]
## 真实的测试数据，比较在测试集上的R2分数
x_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]

from sklearn.preprocessing import PolynomialFeatures
poly4 = PolynomialFeatures(degree = 4)
x_train_poly4 = poly4.fit_transform(x_train)
x_test_poly4 = poly4.transform(x_test)

from sklearn.linear_model import Lasso
las = Lasso()
las.fit(x_train_poly4, y_train)
print("Lasso score: %f" % las.score(x_test_poly4, y_test))
print(las.coef_)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
y_predict = las.predict(x_test_poly4)
r2 = r2_score(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
mae = mean_absolute_error(y_test, y_predict)
print("r2: %f" % r2)
print("mse: %f" % mse)
print("mae: %f" % mae)
