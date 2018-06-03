# -*- coding:utf-8 -*-

import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print(train.info())
print(test.info())

myfeatures = ["Pclass","Sex","Age","Embarked",
	"SibSp","Parch","Fare"]
x_train = train[myfeatures]
x_test = test[myfeatures]
y_train = train["Survived"]

# Embarked
print(x_train["Embarked"].value_counts())
print(x_test["Embarked"].value_counts())
x_train["Embarked"].fillna("S", inplace = True)
x_test["Embarked"].fillna("S", inplace = True)
# Age
x_train["Age"].fillna(x_train["Age"].mean(), inplace = True)
x_test["Age"].fillna(x_test["Age"].mean(), inplace = True)
x_test["Fare"].fillna(x_test["Fare"].mean(), inplace = True)

from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse = False)
x_train = dict_vec.fit_transform(x_train.to_dict(orient = "record"))
x_test = dict_vec.transform(x_test.to_dict(orient = "record"))

from xgboost import XGBClassifier
xgbc = XGBClassifier()
xgbc.fit(x_train, y_train)
xgbc_y_predict = xgbc.predict(x_test)
xgbc_submission = pd.DataFrame({"PassengerId": test["PassengerId"],
	"Survived": xgbc_y_predict})
xgbc_submission.to_csv("nosearch_submission.csv", index = False)

from sklearn.grid_search import GridSearchCV
params = {"max_depth":range(2,7),
	"n_estimators":range(100,1100,200),
	"learning_rate":[0.05,0.1,0.25,0.5,1.0]}
xgbc_best = XGBClassifier()
gs = GridSearchCV(xgbc_best, params, n_jobs = -1, cv = 5, verbose = 1)
gs.fit(x_train, y_train)

print(gs.best_score_)
print(gs.best_params_)

xgbc_best_y_predict = gs.predict(x_test)
best_submission = pd.DataFrame({"PassengerId": test["PassengerId"],
	"Survived": xgbc_best_y_predict})
best_submission.to_csv("submission.csv", index = False)


