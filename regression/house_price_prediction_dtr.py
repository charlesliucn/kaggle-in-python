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

# 使用单一的决策树回归模型
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train, y_train)
dtr_y_predict = dtr.predict(x_test)
dtr_score = dtr.score(x_test, y_test)

# 评价回归效果
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
y_test = ss_y.inverse_transform(y_test)
dtr_y_predict = ss_y.inverse_transform(dtr_y_predict)

dtr_r2 = r2_score(y_test, dtr_y_predict)
dtr_mse = mean_squared_error(y_test, dtr_y_predict)
dts_mae = mean_absolute_error(y_test, dtr_y_predict)

print("Decision Tree: ")
print("r2: %s" % dtr_r2)
print("mse: %s" % dtr_mse)
print("mae: %s" % dts_mae)
