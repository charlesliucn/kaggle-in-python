# -*- coding:utf-8 -*-

# 导入数据集
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.DESCR)

# 训练集测试集分割
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
	iris.data, iris.target, test_size = 0.25, random_state = 33)

# 数据归一化/标准化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# 使用K近邻方法分类
from sklearn.neighbors import KNeighborsClassifier
knc =KNeighborsClassifier()
knc.fit(x_train, y_train)
knc_y_predict = knc.predict(x_test)

# 评价分类预测效果
from sklearn.metrics import classification_report
res = classification_report(y_test, knc_y_predict,
	target_names = iris.target_names)
print("Accuracy: %s" % knc.score(x_test, y_test))
print(res)
