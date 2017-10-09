# load libraries
from sklearn import preprocessing
import numpy as np
import pandas as pd

def MinMaxScaler(x, range=(0,1)):
    # create scaler
    minmax_scale = preprocessing.MinMaxScaler(feature_range=range)
    # scale feature
    x_scale = minmax_scale.fit_transform(x)
    return x_scale, minmax_scale

def StandardScaler(x):
    # create scaler
    scaler = preprocessing.StandardScaler()
    #scale feature
    x_scale = scaler.fit_transform(x)
    return x_scale, scaler

def RobustScaler(x):
    # create scaler
    scaler = preprocessing.RobustScaler()
    #scale feature
    x_scale = scaler.fit_transform(x)
    return x_scale, scaler

def PolynomialFeatures(x, degree, include_bias=False, interaction_only=False):
    # create processor
    poly = preprocessing.PolynomialFeatures(degree=degree, 
                                            include_bias=include_bias, 
                                            interaction_only=interaction_only)
    # fit
    x_poly = poly.fit_transform(x)
    return x_poly, poly

def DetectOutliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    indices = np.where((x > upper_bound) | (x < lower_bound))
    return indices


def percentage(numerator, denomenator):
    if type(numerator) == pd.core.series.Series:
        return (numerator/denomenator*100).map('{:.1f}%'.format)
    
    elif type(numerator) == int or type(numerator) == float:
        return '{:.1f}%'.format(float(numerator)/float(denomenator)*100)
    
    else:
        print("check type")

def DisplayFeatureCompleteness(df):
    return percentage(df.count(), df.shape[0])

def ImputeMissingValuesWithKNN(x, k=5):
    from fancyImpute import KNN
    x_knn_imputed = KNN(k=k, verbose=0).complete(x)
    return x_knn_imputed