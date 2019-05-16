from collections import OrderedDict

import xgboost
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.keys())
print(boston.data.shape)

import pandas as pd

data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

import xgboost as xgb

from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

X, y = data.iloc[:,:-1],data.iloc[:,-1]

data_dmatrix = xgb.DMatrix(data=X,label=y)
from sklearn.model_selection import train_test_split

# should split by time!!!
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

cv_results.head()

# Visualize Boosting Trees and Feature Importance
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)
print(xg_reg)

