# -*- coding:utf-8 -*-

# 导入数据
from sklearn.datasets import load_digits
digits = load_digits()
# print(digits.data.shape)
data = digits.data
target = digits.target

# 数据集分割成训练集和测试集
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
	data, target, test_size = 0.25, random_state = 33)

# 数据标准化/归一化
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)

# SVM分类预测
from sklearn.svm import LinearSVC
lsvc = LinearSVC()
lsvc.fit(x_train, y_train)
lsvc_y_predict = lsvc.predict(x_test)

# 计算SVM预测的结果
from sklearn.metrics import classification_report
res = classification_report(y_test, lsvc_y_predict, 
	target_names = digits.target_names.astype(str))
print("Accuracy: %s" % lsvc.score(x_test, y_test))
print(res)


