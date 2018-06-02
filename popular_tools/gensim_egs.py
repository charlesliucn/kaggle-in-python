# -*- coding:utf-8 -*-

from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset = "all")

from bs4 import BeautifulSoup
import nltk
import re

def news_to_sentences(news):
	news_text = BeautifulSoup(news).get_text()
	tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
	raw_sentences = tokenizer.tokenize(news_text)
	sentences = []
	for sent in raw_sentences:
		sent_proc = re.sub("[^a-zA-Z]"," ",sent.lower().strip())
		sentences.append(sent_proc.split())
	return sentences

sentences = []
for x in news.data:
	sentences += news_to_sentences(x)


from gensim.models import word2vec
num_features = 300
min_word_count = 20
num_workers = 2
context = 5
downsampling = 1e-3

model = word2vec.Word2Vec(sentences, workers = num_workers,
	size = num_features, min_count = min_word_count,
	window = context, sample = downsampling)

model.init_sims(replace = True)
print(model.most_similar("morning"))
print(model.most_similar("email"))