# -*- coding:utf-8 -*-

import re
import pandas as pd
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import nltk.data

train = pd.read_csv("./IMDB/labeledTrainData.tsv", delimiter = "\t")
test = pd.read_csv("./IMDB/testData.tsv", delimiter = "\t")
# print(train.head())
# print(test.head())

unlabeled_train = pd.read_csv("./IMDB/unlabeledTrainData.tsv",
	delimiter = "\t", quoting = 3)

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

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

def review_to_sentences(review, tokenizer):
	raw_sentences = tokenizer.tokenize(review.strip())
	sentences = []
	for raw_sentence in raw_sentences:
		if len(raw_sentence) > 0:
			sentences.append(review_to_text(raw_sentence, False))
	return sentences

corpora = []
for review in unlabeled_train["review"]:
	corpora += review_to_sentences(review.decode("utf-8"), tokenizer)

num_features = 300 # word vectors dimension
min_word_count = 20
num_workers = 2
context = 10
downsampling = 1e-3

from gensim.models import word2vec
print("Training word2vec model...")

model = word2vec.Word2Vec(corpora, workers = num_workers,
	size = num_features, min_count = min_word_count,
	window = context, sample  = downsampling)

model.init_sims(replace = True)

model.save("./IMDB/model.sav")

from gensim.models import Word2Vec
model = Word2Vec.load("./IMDB/model.sav")
model.most_similar("man")