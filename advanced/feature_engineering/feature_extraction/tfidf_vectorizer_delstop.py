# -*- coding:utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset = "all")

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
	news.data, news.target, test_size = 0.25, random_state = 33)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vec = TfidfVectorizer(analyzer = "word", stop_words = "english")
x_train_tfidf = tfidf_vec.fit_transform(x_train)
x_test_tfidf = tfidf_vec.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
mnb.fit(x_train_tfidf, y_train)
mnb_y_predict = mnb.predict(x_test_tfidf)
mnb_score = mnb.score(x_test_tfidf, y_test)

from sklearn.metrics import classification_report
res = classification_report(y_test, mnb_y_predict,
	target_names = news.target_names)
print("Naive Bayes based on tfidf:")
print("score: %s" % mnb_score)
print(res)