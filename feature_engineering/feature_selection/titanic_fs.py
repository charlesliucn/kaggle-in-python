# -*- coding:utf-8 -*-

# 导入数据
import pandas as pd
data_url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt"
titanic = pd.read_csv(data_url)
X = titanic.drop(["row.names","name","survived"], axis = 1)
y = titanic["survived"]

# 数据预处理
X["age"].fillna(X["age"].mean(), inplace = True)
X.fillna("UNKNOWN", inplace = True)

# 数据集分成训练集和测试集
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
	X, y, test_size = 0.25, random_state = 33)

# 进行特征提取
from sklearn.feature_extraction import DictVectorizer
dv = DictVectorizer()
x_train = dv.fit_transform(x_train.to_dict(orient = "record"))
x_test = dv.transform(x_test.to_dict(orient = "record"))
print(dv.feature_names_)

# 直接使用决策树进行分类
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train, y_train)
dt_score = dt.score(x_test, y_test)
print("Decision Tree without feature selection:")
print(dt_score)

# 特征筛选20%后决策树分类
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(
	feature_selection.chi2, percentile = 20)
x_train_fs = fs.fit_transform(x_train, y_train)
x_test_fs = fs.transform(x_test)
dt.fit(x_train_fs, y_train)
dt_score_fs = dt.score(x_test_fs, y_test)
print("Decision Tree after feature selection 20%:")
print(dt_score_fs)
# 交叉验证
from sklearn.cross_validation import cross_val_score
import numpy as np
pers = range(1,100,2)
dt_scores = []
for i in pers:
	tmp_fs = feature_selection.SelectPercentile(
		feature_selection.chi2, percentile = i)
	x_train_tmp = tmp_fs.fit_transform(x_train, y_train)
	x_test_tmp = tmp_fs.transform(x_test)
	dt.fit(x_train_tmp, y_train)
	tmp_score = cross_val_score(dt, x_train_tmp, y_train, cv = 5)
	dt_scores = np.append(dt_scores, tmp_score.mean())
print("Decision Tree Results:")
print(dt_scores)
opt_num = np.where(dt_scores == dt_scores.max())[0]
print(opt_num)
print("The optimal number of features:%d" % pers[opt_num[0]])
# 作图
import matplotlib.pyplot as plt
plt.plot(pers, dt_scores)
plt.xlabel("percentiles of features")
plt.ylabel("accuracy")
plt.show()
# 最优筛选方法
fs_final = feature_selection.SelectPercentile(
	feature_selection.chi2, percentile = pers[opt_num[0]])
x_train_final = fs_final.fit_transform(x_train, y_train)
x_test_final = fs_final.transform(x_test)
dt_final = DecisionTreeClassifier()
dt_final.fit(x_train_final, y_train)
final_score = dt_final.score(x_test_final, y_test)
print("Final Score:%f" % final_score)
# 分类结果
from sklearn.metrics import classification_report
dt_final_y_predict = dt_final.predict(x_test_final)
res = classification_report(y_test, dt_final_y_predict,
	target_names = ["died","survived"])
print(res)