# import base packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import sklearn datasets for testing
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs

import utils.numeric_features as numeric
import utils.prep as prep

# Test scalers
print ('\nTesting Scalers...\n')

x = np.array([[-1000.1],
             [-200.2],
             [0.5],
             [200.6],
             [1000.1],
             [9000.9]])

range=(0,10)
x_scale, scaler = numeric.MinMaxScaler(x, range)
print ('MinMaxScaler with range:', range)
print (x_scale)
print ('Max: ', np.max(x_scale))
print ('Min: ', np.min(x_scale))

print ('\n')

x_scale, scaler = numeric.StandardScaler(x)
print ('StandardScaler')
print (x_scale)
print ('Mean: ', round(x_scale.mean()))
print ('Standard deviation: ', x_scale.std())

print ('\n')

x_scale, scaler = numeric.RobustScaler(x)
print ('RobustScaler')
print (x_scale)
print ('Mean: ', round(x_scale.mean()))
print ('Standard deviation: ', x_scale.std())

print ('\nEnd of Scaler testing...\n')

print ('\nTesting Transformations...\n')

x = np.array([[2,3],
             [2,3],
             [2,3]])

print ('PolynomialFeatures')
x_transformed, transformer = numeric.PolynomialFeatures(x, degree=2)
print ('\nOriginal:\n', x)
print ('\nPolynomial:\n', x_transformed, '\nusing ', transformer)
x_transformed, transformer = numeric.PolynomialFeatures(x, degree=2, interaction_only=True)
print ('\nInteraction-only Polynomial:\n', x_transformed, '\nusing ', transformer)

x = np.array([[-1000.1, 3],
             [-200.2, 5],
             [0.5, 8],
             [200.6, 4],
             [1000.1, 1],
             [9000.9, 12]])

df = pd.DataFrame(x,columns=['column1','column2'])
indices = numeric.DetectOutliers(df['column1'])
print ('\n\n', x, '\nOutliers for column 0:\n', indices)

x = np.array([[-1000.1, 3],
             [-200.2, np.NaN],
             [0.5, 8],
             [np.NaN, 4],
             [1000.1, np.NaN],
             [500.6, 4],
             [600.5, 5],
             [9000.9, 12]])

df = pd.DataFrame(x, columns=['column1', 'column2'])
print (prep.DisplayFeatureCompleteness(df))

print ('\n\nLinear Regression ------------------ ')

from sklearn.datasets import load_boston
boston = load_boston()

print ('Splitting...')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.25, random_state=33)

X_train, scalerX = numeric.StandardScaler(X_train)
y_train, scalerY = numeric.StandardScaler(y_train)

print ('Scaling...')
X_test = scalerX.transform(X_test)
y_test = scalerY.transform(y_test)

from sklearn import linear_model
import utils.models as models

print ('\nUsing SGDRegressor\n')
clf_sgd = linear_model.SGDRegressor(loss='squared_loss', penalty='l2')
models.train_and_evaluate_regression(clf_sgd, X_train, y_train)

from sklearn import svm

print ('\nUsing SVR with linear kernel\n')
clf_svr = svm.SVR(kernel='linear')
models.train_and_evaluate_regression(clf_svr, X_train, y_train)

print ('\nUsing SVR with poly kernel\n')
clf_svr = svm.SVR(kernel='poly')
models.train_and_evaluate_regression(clf_svr, X_train, y_train)

print ('\nUsing SVR with rbf kernel\n')
clf_svr = svm.SVR(kernel='rbf')
models.train_and_evaluate_regression(clf_svr, X_train, y_train)

print ('\nUsing ExtraTrees\n')
from sklearn import ensemble
clf_et=ensemble.ExtraTreesRegressor(n_estimators=50)
m = models.train_and_evaluate_regression(clf_et, X_train, y_train)
models.predict_and_evaluate_regression(m, X_test, y_test)

print ('\nUsing RandomForest\n')
from sklearn import ensemble
clf_et=ensemble.RandomForestRegressor(n_estimators=50)
m = models.train_and_evaluate_regression(clf_et, X_train, y_train)
models.predict_and_evaluate_regression(m, X_test, y_test)

print ('\nUsing AdaBoost\n')
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
clf_et=ensemble.AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=50)
m = models.train_and_evaluate_regression(clf_et, X_train, y_train)
models.predict_and_evaluate_regression(m, X_test, y_test)

print ('\nUsing GradientBoostingRegressor\n')
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
clf_et=ensemble.GradientBoostingRegressor(n_estimators=50)
m = models.train_and_evaluate_regression(clf_et, X_train, y_train)
models.predict_and_evaluate_regression(m, X_test, y_test)