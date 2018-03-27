
import numpy as np
import pandas as pd
import math
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix

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
    return y_pred, model.coef_

def train_and_evaluate_classification(model, X_train, y_train, k_folds=10):
    model.fit(X_train, y_train)
    scores = cross_val_score(model, X_train, y_train, cv=k_folds, scoring='accuracy')
    print ("Accuracy score on traning data: ", model.score(X_train, y_train))   
    print ("Average accuracy score using %i-fold crossvalidation: " %k_folds, np.mean(scores))
    return scores

def predict_and_evaluate_classification(model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    print ("Accuracy score: %.5f" % r2_score(y_test, y_pred))
    c_report = classification_report(y_test, y_pred, labels)
    matrix = confusion_matrix(y_test, y_pred, labels)
    print("Classification Report:\n", c_report)
    print("\nConfusion Matrix:\n", matrix)
    return y_pred, c_report, matrix

