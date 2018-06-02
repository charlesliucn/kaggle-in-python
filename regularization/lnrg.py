# -*- coding:utf-8 -*-

# 训练数据和测试数据
import numpy as np
x_train = [[6],[8],[10],[14],[18]]
y_train = [[7],[9],[13],[17.5],[18]]
x_test = np.linspace(0,26,100)
x_test = x_test.reshape(x_test.shape[0],1)

# 线性回归(一阶)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)
y_predict = lr.predict(x_test)
lr_score = lr.score(x_train, y_train)
print("Degree 1:")
print("R2 on training data: %f" % lr_score)

# 线性回归(二阶)
from sklearn.preprocessing import PolynomialFeatures
poly2 = PolynomialFeatures(degree = 2)
x_train_poly2 = poly2.fit_transform(x_train)
x_test_poly2 = poly2.transform(x_test)
lr_poly2 = LinearRegression()
lr_poly2.fit(x_train_poly2, y_train)
y_predict_poly2 = lr_poly2.predict(x_test_poly2)
lr_score_poly2 = lr_poly2.score(x_train_poly2, y_train)
print("Degree 2:")
print("R2 on training data: %f" % lr_score_poly2)
# 线性回归(四阶)
poly4 = PolynomialFeatures(degree = 4)
x_train_poly4 = poly4.fit_transform(x_train)
x_test_poly4 = poly4.transform(x_test)
lr_poly4 = LinearRegression()
lr_poly4.fit(x_train_poly4, y_train)
y_predict_poly4 = lr_poly4.predict(x_test_poly4)
lr_score_poly4 = lr_poly4.score(x_train_poly4, y_train)
print("Degree 4:")
print("R2 on training data: %f" % lr_score_poly4)

# 比较三种线性回归模型的拟合曲线
import matplotlib.pyplot as plt
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_test, y_predict, "k-", label = "Degree = 1")
plt.plot(x_test, y_predict_poly2, "b:", label = "Degree = 2")
plt.plot(x_test, y_predict_poly4, "g-.", label = "Degree = 4")
plt.xlabel("Diameter of Pizza")
plt.ylabel("Price of Pizza")
plt.legend()
plt.axis([0,25,0,25])
plt.show()

# 真实的测试数据，比较在测试集上的R2分数
x_test = [[6],[8],[11],[16]]
y_test = [[8],[12],[15],[18]]
print("Degree 1 score: %f" % lr.score(x_test, y_test))
print("Degree 2 score: %f" % lr_poly2.score(poly2.transform(x_test),y_test))
print("Degree 4 score: %f" % lr_poly4.score(poly4.transform(x_test),y_test))

# 计算各阶模型在测试集上的分数
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
y_predict1 = lr.predict(x_test)
x_test_poly2 = poly2.transform(x_test)
y_predict2 = lr_poly2.predict(x_test_poly2)
x_test_poly4 = poly4.transform(x_test)
y_predict4 = lr_poly4.predict(x_test_poly4)

r2_score1 = r2_score(y_test, y_predict1)
r2_score2 = r2_score(y_test, y_predict2)
r2_score4 = r2_score(y_test, y_predict4)

mse1 = mean_squared_error(y_test, y_predict1)
mse2 = mean_squared_error(y_test, y_predict2)
mse4 = mean_squared_error(y_test, y_predict4)

mae1 = mean_absolute_error(y_test, y_predict1)
mae2 = mean_absolute_error(y_test, y_predict2)
mae4 = mean_absolute_error(y_test, y_predict4)

print("Degree 1:")
print([r2_score1,mse1, mae1])
print("Degree 2:")
print([r2_score2,mse2, mae2])
print("Degree 4:")
print([r2_score4,mse4, mae4])