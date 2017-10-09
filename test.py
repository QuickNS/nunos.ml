# import base packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import sklearn datasets for testing
from sklearn.datasets import make_regression
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs

import utils.numeric_features as utils

# Test scalers
print ('\nTesting Scalers...\n')

x = np.array([[-1000.1],
             [-200.2],
             [0.5],
             [200.6],
             [1000.1],
             [9000.9]])

range=(0,10)
x_scale, scaler = utils.MinMaxScaler(x, range)
print ('MinMaxScaler with range:', range)
print (x_scale)
print ('Max: ', np.max(x_scale))
print ('Min: ', np.min(x_scale))

print ('\n')

x_scale, scaler = utils.StandardScaler(x)
print ('StandardScaler')
print (x_scale)
print ('Mean: ', round(x_scale.mean()))
print ('Standard deviation: ', x_scale.std())

print ('\n')

x_scale, scaler = utils.RobustScaler(x)
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
x_transformed, transformer = utils.PolynomialFeatures(x, degree=2)
print ('\nOriginal:\n', x)
print ('\nPolynomial:\n', x_transformed, '\nusing ', transformer)
x_transformed, transformer = utils.PolynomialFeatures(x, degree=2, interaction_only=True)
print ('\nInteraction-only Polynomial:\n', x_transformed, '\nusing ', transformer)

x = np.array([[-1000.1, 3],
             [-200.2, 5],
             [0.5, 8],
             [200.6, 4],
             [1000.1, 1],
             [9000.9, 12]])

df = pd.DataFrame(x,columns=['column1','column2'])
indices = utils.DetectOutliers(df['column1'])
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
print (utils.DisplayFeatureCompleteness(df))

x_knn = utils.ImputeMissingValuesWithKNN(x)
print('Before: ', x)
print('\nAfter: ', x_knn)
