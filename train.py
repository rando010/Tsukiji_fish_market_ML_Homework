import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeRegressor

from joblib import dump

from preprocess import prep_data

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

from sklearn.metrics import mean_squared_error

fish_data = pd.read_csv("fish_participant.csv")

print(fish_data.head)

print(fish_data.dtypes)

X, y = prep_data(fish_data)

decisiontree = DecisionTreeRegressor()

cross_validate(
    decisiontree,
    X,
    y,
    scoring="neg_mean_squared_error",
    cv=KFold(random_state=123, shuffle=True),
)["test_score"].mean()

decisiontree.fit(X, y)

fish_data_holdout = pd.read_csv("fish_holdout_demo.csv")

X_hold, y_true = prep_data(fish_data_holdout)

y_predict = decisiontree.predict(X_hold)

print([y_true, y_predict])

print(mean_squared_error(y_true, y_predict))
