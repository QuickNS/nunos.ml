
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

def OneHotEncode(x):
    categorical = x.select_dtypes(exclude=[np.number])
    return pd.get_dummies(x, columns=categorical.columns)

def makeCategorical(x, feature, value_list):
    return x[feature].astype("category", categories = value_list)

def train_and_evaluate_regression(model, X_train, y_train, k_folds=10):
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring='neg_mean_squared_error')
    scores = np.sqrt(-scores)
    print ("R2 Score on training set: ", model.score(X_train, y_train))   
    print ("Average RMSE score using %i-fold crossvalidation: " %k_folds, np.mean(scores))
    return scores

def predict_and_evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print ("R2 score: %.5f" % r2_score(y_test, y_pred))
    print ("RMSE: %.5f" % math.sqrt(mean_squared_error(y_test, y_pred)))
    print("Coefficients:", model.coef_)
    return y_pred

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

def ApplyStandardScaler(data, scaler):
    from sklearn import preprocessing
    # create scaler
    scaler = preprocessing.StandardScaler()
    #scale feature
    x_scale = scaler.fit_transform(x)
    return x_scale, scaler

