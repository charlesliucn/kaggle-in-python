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

# 直接使用梯度Boosting进行分类
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier()
gbc.fit(x_train, y_train)
gbc_score = gbc.score(x_test, y_test)
print("GradientBoostingClassifier Score:")
print(gbc_score)

# 特征筛选20%后梯度Boosting分类
from sklearn import feature_selection
fs = feature_selection.SelectPercentile(
	feature_selection.chi2, percentile = 20)
x_train_fs = fs.fit_transform(x_train, y_train)
x_test_fs = fs.transform(x_test)
gbc.fit(x_train_fs, y_train)
gbc_score_fs = gbc.score(x_test_fs, y_test)
print("GradientBoostingClassifier after feature selection 20%:")
print(gbc_score_fs)

# 交叉验证评价模型指标
from sklearn.cross_validation import cross_val_score
import numpy as np
pers = range(1,100,2)
gbc_scores = []
for i in pers:
	tmp_fs = feature_selection.SelectPercentile(
		feature_selection.chi2, percentile = i)
	x_train_tmp = tmp_fs.fit_transform(x_train, y_train)
	x_test_tmp = tmp_fs.transform(x_test)
	gbc.fit(x_train_tmp, y_train)
	tmp_score = cross_val_score(gbc, x_train_tmp, y_train, cv = 5)
	gbc_scores = np.append(gbc_scores, tmp_score.mean())
print("GradientBoostingClassifier Results:")
print(gbc_scores)
opt_num = np.where(gbc_scores == gbc_scores.max())[0]
print(opt_num)
print("The optimal number of features:%d" % pers[opt_num[0]])
## 作图找到最好的percentile
import matplotlib.pyplot as plt
plt.plot(pers, gbc_scores)
plt.xlabel("percentiles of features")
plt.ylabel("accuracy")
plt.show()

# 最优特征筛选参数，然后查看模型结果
fs_final = feature_selection.SelectPercentile(
	feature_selection.chi2, percentile = pers[opt_num[0]])
x_train_final = fs_final.fit_transform(x_train, y_train)
x_test_final = fs_final.transform(x_test)
gbc_final = GradientBoostingClassifier()
gbc_final.fit(x_train_final, y_train)
final_score = gbc_final.score(x_test_final, y_test)
print("Final Score:%f" % final_score)
## 分类结果
from sklearn.metrics import classification_report
gbc_final_y_predict = gbc_final.predict(x_test_final)
res = classification_report(y_test, gbc_final_y_predict,
	target_names = ["died","survived"])
print(res)
