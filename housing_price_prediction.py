#!/usr/bin/env python -W ignore::RuntimeWarning

import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, make_scorer, mean_squared_error
from sklearn.grid_search import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

# Load UCI housing data set
data = pd.read_csv('./housing.data', delim_whitespace=True, header=None)
prices = data[[13]]
prices.columns = ['MEDV']
features = data[[5]]
features.columns = ['RM']

# Cross-Validation setting
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.2, random_state=42)
cv_sets = ShuffleSplit(X_train.shape[0], n_iter = 10, test_size = 0.20, random_state = 0)
params = {'normalize': [False, True], 'fit_intercept': [False, True]}

# Learning
regressor = LinearRegression()
scoring_fnc = make_scorer(r2_score)
grid = GridSearchCV(regressor, params, cv=cv_sets, scoring=scoring_fnc)
best_reg = grid.fit(X_train, y_train)

# Prediction result
print("Prediction sample : room number = {0}, actual price = {1}, predicted price = {2}".format(X_test['RM'][0], y_test['MEDV'][0], best_reg.predict(X_test['RM'][0])[0][0]))
print("MSE : {}".format(mean_squared_error(y_test, best_reg.predict(X_test))))
