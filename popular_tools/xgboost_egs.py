# -*- coding:utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
url = "http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt"
titanic = pd.read_csv(url)
X = titanic[["pclass","age","sex"]]
y = titanic["survived"]

X["age"].fillna(X["age"].mean(), inplace = True)

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
	X, y, test_size = 0.25, random_state = 33)

from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer(sparse = False)

x_train = vec.fit_transform(x_train.to_dict(orient = "record"))
x_test = vec.transform(x_test.to_dict(orient = "record"))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
score = rfc.score(x_test, y_test)
print("RandomForestClassifier:")
print("score: %f" % score)

from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(x_train, y_train)
score2 = xgbc.score(x_test, y_test)
print("XGBClassifier:")
print("score: %f" % score2)

