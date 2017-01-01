#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn import datasets
from pydotplus import graph_from_dot_data

# Load data
iris = datasets.load_iris()
features = iris.data
categories = iris.target

# Cross-Validation setting
X_train, X_test, y_train, y_test = train_test_split(features, categories, test_size=0.2, random_state=42)
cv_sets = ShuffleSplit(X_train.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
params = {'max_depth': np.arange(3,11)}

# Learning
def performance_metric(y_true, y_predict):
  score = f1_score(y_true, y_predict, average='micro')
  return score

classifier = DecisionTreeClassifier()
scoring_fnc = make_scorer(performance_metric)
grid = GridSearchCV(classifier, params, cv=cv_sets, scoring=scoring_fnc)
best_clf = grid.fit(X_train, y_train)

# Prediction result
print("Optimal models's parameter 'max_depth' : {} ".format(best_clf.best_estimator_.get_params()['max_depth']))
print("Classifiction sample : features = {0}, actual category = {1}, classification result = {2}".format(X_test[0], y_test[0], best_clf.predict(np.array([X_test[0]])[0])[0]))
print("Accuracy : {}".format(accuracy_score(y_test, best_clf.predict(X_test))))

# Output decision tree
dot_data = export_graphviz(best_clf.best_estimator_, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)

graph = graph_from_dot_data(dot_data)
graph.write_pdf('iris_clf_tree.pdf')
