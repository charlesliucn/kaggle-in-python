# -*- coding:utf-8 -*-

sent1 = "The cat is walking in the bedroom."
sent2 = "A dog was running across the kitchen."

from sklearn.feature_extraction.text import CountVectorizer
count_vec = CountVectorizer()
sents = [sent1, sent2]
arr = count_vec.fit_transform(sents).toarray()
print(arr)
print(count_vec.get_feature_names())

import nltk
stemmer = nltk.stem.PorterStemmer()

tokens1 = nltk.word_tokenize(sent1)
print(tokens1)
vocab1 = sorted(set(tokens1))
print(vocab1)
stem1 = [stemmer.stem(t) for t in tokens1]
print(stem1)
pos_tag1 = nltk.tag.pos_tag(tokens1)
print(pos_tag1)

tokens2 = nltk.word_tokenize(sent2)
print(tokens2)
vocab2 = sorted(set(tokens2))
print(vocab2)
stem2 = [stemmer.stem(t) for t in tokens2]
print(stem2)
pos_tag2 = nltk.tag.pos_tag(tokens2)
print(pos_tag2)
