# -*- coding:utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset = "all")

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
	news.data[:3000], news.target[:3000], 
	test_size = 0.25, random_state = 33)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import numpy as np

clf = Pipeline([	("vect",TfidfVectorizer(stop_words = "english", analyzer = "word")),
				("svc", SVC())
			  ])
params = {	"svc__gamma": np.logspace(-2,1,4),
			"svc__C":np.logspace(-1,1,3)}


from sklearn.grid_search import GridSearchCV
# 单线程网格搜索
gs = GridSearchCV(clf, params, verbose = 2,
	refit = True, cv = 3)
gs.fit(x_train, y_train)
print("Best Params:")
print(gs.best_params_)
print("Best Score:")
print(gs.best_score_)
print("Test Score:")
print(gs.score(x_test, y_test))

# 多线程网格搜索
gs_par = GridSearchCV(clf, params, verbose = 2,
	refit = True, cv = 3, n_jobs = -1)
gs_par.fit(x_train, y_train)
print("Best Params:")
print(gs_par.best_params_)
print("Best Score:")
print(gs_par.best_score_)
print("Test Score:")
print(gs_par.score(x_test, y_test))