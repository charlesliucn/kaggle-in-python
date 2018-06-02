# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

greeting = tf.constant("Hello Google Tensorflow!")
sess = tf.Session()
print(sess.run(greeting))
sess.close()

matrix1 = tf.constant([[3.0,3.0]])
matrix2 = tf.constant([[2.0],[2.0]])
product = tf.matmul(matrix1, matrix2)
linear = tf.add(product, tf.constant(2.0))

with tf.Session() as sess:
	res = sess.run(linear)
print(res)

# ====================

import tensorflow as tf
import numpy as np
import pandas as pd
train = pd.read_csv("breast-cancer-train.csv")
test = pd.read_csv("breast-cancer-test.csv")

x_train = np.float32(train[["Clump Thickness","Cell Size"]].T)
y_train = np.float32(train["Type"].T)
x_test = np.float32(test[["Clump Thickness","Cell Size"]].T)
y_test = np.float32(test["Type"].T)

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2],-1.0,1.0))
y = tf.matmul(W, x_train) + b
loss = tf.reduce_mean(tf.square(y - y_train))
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(1000):
	sess.run(train)
	if step % 200 == 0:
		print([step, sess.run(W), sess.run(b)])

test_negative = test.loc[test["Type"] == 0][["Clump Thickness","Cell Size"]]
test_positive = test.loc[test["Type"] == 1][["Clump Thickness","Cell Size"]]

import matplotlib.pyplot as plt
plt.scatter(test_negative["Clump Thickness"], test_negative["Cell Size"],
	marker = "o", s = 200, c = "red")
plt.scatter(test_positive["Clump Thickness"], test_positive["Cell Size"],
	marker = "x", s = 150, c = "black")
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")

lx = np.arange(0,12)
ly = (0.5 - sess.run(b) - lx * sess.run(W)[0][0]) / sess.run(W)[0][0]
plt.plot(lx, ly, color = "green")
plt.show()
