# -*- coding:utf-8 -*-

import re
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

## 读入数据
train = pd.read_csv("./IMDB/labeledTrainData.tsv", delimiter = "\t")
test = pd.read_csv("./IMDB/testData.tsv", delimiter = "\t")
# print(train.head())
# print(test.head())

## 文本处理函数
def review_to_text(review, remove_stopwords):
	"""
	review: type str
	remove_stopwords: type boolean
	"""
	raw_text = BeautifulSoup(review, "html").get_text()
	letters = re.sub("[^a-zA-Z]", " ", raw_text)
	words = letters.lower().split()
	if remove_stopwords:
		stop_words = set(stopwords.words("english"))
		words = [w for w in words if w not in stop_words]
	return words

## 获取有标签训练数据和无标签测试数据
x_train = []
for review in train["review"]:
	x_train.append(" ".join(review_to_text(review, True)))
print(x_train)
y_train = train["sentiment"]
x_test = []
for review in test["review"]:
	x_test.append(" ".join(review_to_text(review, True)))


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV

pip_count = Pipeline([("count_vec", CountVectorizer(analyzer = "word")),
		("mnb", MultinomialNB())])
pip_tfidf = Pipeline([("tfidf_vec", TfidfVectorizer(analyzer = "word")),
		("mnb", MultinomialNB())])

params_count = {
	"count_vec__binary":[True, False],
	"count_vec__ngram_range":[(1,1),(1,2)],
	"mnb__alpha":[0.1,1.0,10.0]
}

params_tfidf = {
	"tfidf_vec__binary":[True, False],
	"tfidf_vec__ngram_range":[(1,1),(1,2)],
	"mnb__alpha":[0.1,1.0,10.0]
}

gs_count = GridSearchCV(pip_count, params_count, cv = 4,
	n_jobs = -1, verbose = 1)
gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv = 4,
	n_jobs = -1, verbose = 1)

gs_count.fit(x_train, y_train)
print(gs_count.best_score_)
print(gs_count.best_params_)
count_y_predict = gs_count.predict(x_test)

gs_tfidf.fit(x_train, y_train)
print(gs_tfidf.best_score_)
print(gs_tfidf.best_params_)
tfidf_y_predict = gs_tfidf.predict(x_test)

submission_count = pd.DataFrame({"id":test["id"],"sentiment":count_y_predict})
submission_tfidf = pd.DataFrame({"id":test["id"],"sentiment":tfidf_y_predict})
submission_count.to_csv("submission_count.csv")
submission_tfidf.to_csv("submission_tfidf.csv")
