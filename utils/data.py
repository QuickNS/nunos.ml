import numpy as np
import pandas as pd
import math

def getCoefficientsDataFrame(feature_names, coefs):
    T = pd.DataFrame()
    T['feature'] = feature_names
    T['coefs'] = coefs
    T['weight'] = T['coefs'].apply(lambda x: math.fabs(x))
    return T.sort_values(by='weight', ascending=False)

def DetectOutliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    indices = x[(x > upper_bound) | (x < lower_bound)].index
    return indices

def DetectMissing(data):
    return data.isnull().sum()

def OneHotEncode(x):
    categorical = x.select_dtypes(exclude=[np.number])
    return pd.get_dummies(x, columns=categorical.columns)

def makeCategorical(x, feature, value_list):
    return x[feature].astype("category", categories = value_list)