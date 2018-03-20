from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import math

def train_and_evaluate_regression(model, X_train, y_train, k_folds=5):
    model.fit(X_train, y_train)
    print ("R2 Score on training set: ", model.score(X_train, y_train))
    cv = KFold(5, shuffle = True)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')
    print ("Average R2 score using %i-fold crossvalidation: " %k_folds, np.mean(scores))
    return model

def predict_and_evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print ("Root of Mean Squared Error: %.2f" % math.sqrt(mean_squared_error(y_test, y_pred)))
    print ("Variance score: %.2f" % r2_score(y_test, y_pred))
    return y_pred

