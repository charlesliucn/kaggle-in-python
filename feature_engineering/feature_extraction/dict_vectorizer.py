# -*- coding:utf-8 -*-


## 字典型数据
measurements = [{"city":"Dubai","temperature":33.0},
{"city":"London","temperature":12.0},
{"city":"San Fransisco","temperature":18.0}]

## 特征提取
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
vec.fit_transform(measurements).toarray()
feature_names = vec.get_feature_names()
print(feature_names)