import numpy as np
import pandas as pd

def OneHotEncode(x):
    categorical = x.select_dtypes(exclude=[np.number])
    return pd.get_dummies(x, columns=categorial.columns)

def makeCategorical(x, feature, value_list):
    return x[feature].astype("category", categories = value_list)