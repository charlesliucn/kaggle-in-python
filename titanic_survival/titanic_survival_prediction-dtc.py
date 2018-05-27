# -*- coding:utf-8 -*-

# 载入数据
import pandas as pd
titanic = pd.read_csv("titanic.csv")
print(titanic.head())
print(titanic.info())

# 数据预处理
X = titanic[["pclass","age","sex"]]
y = titanic["survived"]
print(X.info())
X["age"].fillna(X["age"].mean(), inplace = True)
print(X.info())

# 数据分成训练集和测试集
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
	X, y, test_size = 0.25, random_state = 33)

# 特征变换，变量形式变换
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)
x_train = vec.fit_transform(x_train.to_dict(orient = "record"))
print(vec.feature_names_)
x_test = vec.transform(x_test.to_dict(orient = "record"))

# 使用决策树进行分类
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
dtc_y_predict = dtc.predict(x_test)

# 分类效果评价
from sklearn.metrics import classification_report
res = classification_report(y_test, dtc_y_predict, 
	target_names = ["died", "survived"])
print("Acurracy: %s" % dtc.score(x_test, y_test))
print(res)