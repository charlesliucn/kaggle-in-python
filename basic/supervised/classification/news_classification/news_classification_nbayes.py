# -*- coding:utf-8 -*-

# 导入数据集
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset = "all")
print(len(news.data))
print(news.data[0])

# 将数据集分成训练集和测试集
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
	news.data, news.target, test_size = 0.25, random_state = 33)

# 将文本转换为向量特征
from sklearn.feature_extraction.text import CountVectorizer
vec = CountVectorizer()
x_train = vec.fit_transform(x_train)
x_test = vec.transform(x_test)

# 使用向量特征进行简单Bayes分类
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train, y_train)
mnb_y_predict = mnb.predict(x_test)

# 评价分类的效果
from sklearn.metrics import classification_report
res = classification_report(y_test, mnb_y_predict, 
	target_names = news.target_names)
print("Accuracy: %s" % mnb.score(x_test, y_test))
print(res)