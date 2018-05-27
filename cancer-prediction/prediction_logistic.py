# -*- coding:utf-8 -*-

# input data
import pandas as pd
df_train = pd.read_csv("breast-cancer-train.csv")
df_test  = pd.read_csv("breast-cancer-test.csv")
print(df_train.shape)
df_test_negative = df_test.loc[df_test["Type"] == 0][["Clump Thickness","Cell Size"]]
df_test_positive = df_test.loc[df_test["Type"] == 1][["Clump Thickness","Cell Size"]]

# plot the data
import matplotlib.pyplot as plt
plt.scatter(df_test_negative["Clump Thickness"], df_test_negative["Cell Size"],
	marker = "o", s = 200, c = "red")
plt.scatter(df_test_positive["Clump Thickness"], df_test_positive["Cell Size"],
	marker = "x", s = 150, c = "black")
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")
plt.show()

# # random plot
# import numpy as np
# intercept = np.random.random([1])
# coef = np.random.random([2])
# lx = np.arange(0, 12)
# ly = (-intercept - lx * coef[0]) / coef[1]
# plt.plot(lx, ly, c = "yellow")
# plt.scatter(df_test_negative["Clump Thickness"], df_test_negative["Cell Size"],
# 	marker = "o", s = 200, c = "red")
# plt.scatter(df_test_positive["Clump Thickness"], df_test_positive["Cell Size"],
# 	marker = "x", s = 150, c = "black")
# plt.xlabel("Clump Thickness")
# plt.ylabel("Cell Size")
# plt.show()

# Logistic Regression for Prediction
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(df_train[["Clump Thickness","Cell Size"]], df_train["Type"])
test_accuracy = lr.score(df_test[["Clump Thickness", "Cell Size"]], df_test["Type"])
print("Test Accuracy: %s " % test_accuracy)

intercept = lr.intercept_
coef = lr.coef_[0,:]
ly_lr = (-intercept - lx * coef[0]) / coef[1]
plt.plot(lx, ly_lr, c = "green")
plt.scatter(df_test_negative["Clump Thickness"], df_test_negative["Cell Size"],
	marker = "o", s = 200, c = "red")
plt.scatter(df_test_positive["Clump Thickness"], df_test_positive["Cell Size"],
	marker = "x", s = 150, c = "black")
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")
plt.show()
