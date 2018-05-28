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

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor

rfr = RandomForestRegressor()
rfr.fit(x_train, y_train)
rfr_y_predict = rfr.predict(x_test)
rfr_score = rfr.score(x_test, y_test)

etr = ExtraTreesRegressor()
etr.fit(x_train, y_train)
etr_y_predict = etr.predict(x_test)
etr_score = etr.score(x_test, y_test)

gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train)
gbr_y_predict = gbr.predict(x_test)
gbr_score = gbr.score(x_test, y_test)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

y_test = ss_y.inverse_transform(y_test)

rfr_y_predict = ss_y.inverse_transform(rfr_y_predict)
rfr_r2 = r2_score(y_test, rfr_y_predict)
rfr_mse = mean_squared_error(y_test, rfr_y_predict)
rfr_mae = mean_absolute_error(y_test, rfr_y_predict)

etr_y_predict = ss_y.inverse_transform(etr_y_predict)
etr_r2 = r2_score(y_test, etr_y_predict)
etr_mse = mean_squared_error(y_test, etr_y_predict)
etr_mae = mean_absolute_error(y_test, etr_y_predict)

gbr_y_predict = ss_y.inverse_transform(gbr_y_predict)
gbr_r2 = r2_score(y_test, gbr_y_predict)
gbr_mse = mean_squared_error(y_test, gbr_y_predict)
gbr_mae = mean_absolute_error(y_test, gbr_y_predict)

print("Random Forest:")
print("score: %s" % rfr_score)
print("r2: %s" % rfr_r2)
print("mse: %s" % rfr_mse)
print("mae: %s" % rfr_mae)

print("Extra Trees:")
print("score: %s" % etr_score)
print("r2: %s" % etr_r2)
print("mse: %s" % etr_mse)
print("mae: %s" % etr_mae)

print("Gradient Boosting:")
print("score: %s" % gbr_score)
print("r2: %s" % gbr_r2)
print("mse: %s" % gbr_mse)
print("mae: %s" % gbr_mae)