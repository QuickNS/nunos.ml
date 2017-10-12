import pandas as pd
import numpy as np
import math

##################################
## Model Parameters
##################################

remove_label_outliers = False
impute_strategy = 'mean'
scaling_method = 'standard'
do_linear_models = True
do_ensemble_models = True
cross_validation_k_folds = 8

##################################
## Reading data
##################################

X = pd.read_csv('data/train_values.csv', index_col=0)
Y = pd.read_csv('data/train_labels.csv', index_col=0)
X_score = pd.read_csv('data/test_values.csv', index_col=0)

##################################
## Remove features
##################################
def removeFeatures(x):
    return x

X = removeFeatures(X)
X_score = removeFeatures(X_score)

##################################
## Fixing categorical data
##################################

print ('\nTraining dataset shape:\n%d rows, %d features' %(X.shape[0], X.shape[1]))

def OneHotEncode(x):
    categorical = x.select_dtypes(exclude=[np.number])
    return pd.get_dummies(x, columns=categorical.columns)

def makeCategorical(x, feature, value_list):
    x[feature] = x[feature].astype("category", categories = value_list)
    return x

def fixCategoricalData(df):
    df = makeCategorical(df, 'school__degrees_awarded_highest', ["Non-degree-granting", "Certificate degree", "Associate degree", "Bachelor's degree", "Graduate degree"])
    df = makeCategorical(df, 'school__degrees_awarded_predominant', ["Not classified", "Predominantly certificate-degree granting", "Predominantly associate's-degree granting", "Predominantly bachelor's-degree granting", "Entirely graduate-degree granting"])
    df = makeCategorical(df, 'school__institutional_characteristics_level', ["2-year", "4-year", "Less-than-2-year"])
    df = makeCategorical(df, 'school__ownership', ["Public", "Private for-profit", "Private nonprofit"])
    df = makeCategorical(df, 'school__region_id', ["Plains (IA, KS, MN, MO, NE, ND, SD)", "New England (CT, ME, MA, NH, RI, VT)", "Southeast (AL, AR, FL, GA, KY, LA, MS, NC, SC, TN, VA, WV)", "Mid East (DE, DC, MD, NJ, NY, PA)", "Great Lakes (IL, IN, MI, OH, WI)", "Far West (AK, CA, HI, NV, OR, WA)", "Southwest (AZ, NM, OK, TX)", "Rocky Mountains (CO, ID, MT, UT, WY)", "Outlying Areas (AS, FM, GU, MH, MP, PR, PW, VI)"])
    df = makeCategorical(df, 'school__state', ["axc", "fga", "oly", "dmg", "hbt", "jgn", "kll", "xve", "dfy", "oon", "oli", "iqy", "qim", "shi", "ccg", "dkf", "ipu", "tbs", "luw", "pxv", "hww", "lff", "slp", "wjh", "idw", "ezv", "vvi", "zdl", "jsu", "hks", "bww", "fxt", "rxy", "cfi", "rse", "kus", "oub", "uah", "rya", "eyi", "wto", "gkt", "bkc", "znt", "slo", "hqy", "rgs", "cmz", "kdg", "pdh", "ahh", "twr", "xws", "por", "uuo", "nhl", "hmr", "jfm"])
    df = makeCategorical(df, 'school__main_campus', ["Main campus", "Not main campus"])
    df = makeCategorical(df, 'school__online_only', ["Not distance-education only", "nan", "Distance-education only"])
    return df

X = fixCategoricalData(X)
X_score = fixCategoricalData(X_score)

X = OneHotEncode(X)
X_score = OneHotEncode(X_score)

print ('Fixed categories!\nNew number of features %d ' % X.shape[1])

##################################
## Handling Outliers
##################################
def DetectOutliers(x):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 2)
    upper_bound = q3 + (iqr * 2)
    indices = x[(x > upper_bound) | (x < lower_bound)].index
    return indices

def RemoveOutliers(x, indexes):
    return x.drop(indexes, axis=0)

if (remove_label_outliers):
    outliers = DetectOutliers(Y['income'])
    print ('\nFound %d outliers for income:\n' % len(outliers))
    print ('Original statistics\n', Y['income'].describe())
    X = RemoveOutliers(X, outliers)
    Y = RemoveOutliers(Y, outliers)

    print ('\nRemoving values:\nNew observation count: ', X.shape[0])
    print ('Original statistics\n', Y['income'].describe())

##################################
## Filling missing values
##################################
from sklearn.preprocessing import Imputer

def percentage(numerator, denomenator):
    if type(numerator) == pd.core.series.Series:
        return (numerator/denomenator*100).map('{:.1f}%'.format)
    
    elif type(numerator) == int or type(numerator) == float:
        return '{:.1f}%'.format(float(numerator)/float(denomenator)*100)
    
    else:
        print("check type")

def DisplayFeatureCompleteness(df):
    nums = df.count()
    nums = nums[nums < df.shape[0]].sort_values(ascending=True)
    print ("Found %d features with missing value:\n" % nums.shape[0])
    if (nums.shape[0] > 0):
        print (percentage(nums, df.shape[0]).head(5))
        print ('...')
        print (percentage(nums, df.shape[0]).tail(5))

print ('\nFixing missing values:')
DisplayFeatureCompleteness(X)

columns = X.columns
imputer = Imputer(strategy=impute_strategy)
imputer.fit(X)
X = imputer.transform(X)
X_score = imputer.transform(X_score)

#rearrange DataFrame after transform
X = pd.DataFrame(X, columns=columns)
X_score = pd.DataFrame(X_score, columns=columns)

print ('\nAfter imputing using ''%s'':' % impute_strategy)
DisplayFeatureCompleteness(X)

##################################
## Scaling Features
##################################
from sklearn import preprocessing

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

print ('\nScaling features using %s:' % scaling_method)
print ('Sample feature before\n', X['school__faculty_salary'].describe())

if (scaling_method == 'standard'):
    X, scaler = StandardScaler(X)
else:
    X, scaler = RobustScaler(X)

X_score = scaler.transform(X_score)

#rearrange DataFrame after transform
X = pd.DataFrame(X, columns=columns)
X_score = pd.DataFrame(X_score, columns=columns)

print ('\nSample feature after\n', X['school__faculty_salary'].describe())

##################################
## Begin regression testing
##################################
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn import svm

def train_and_evaluate_regression(model, X_train, y_train, k_folds=cross_validation_k_folds):
    model.fit(X_train, y_train)
    print ("R2 Score on training set: ", model.score(X_train, y_train))
    cv = KFold(k_folds, shuffle = True)
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    cv_score = math.sqrt(math.fabs(np.mean(scores)))
    print ("Average RMSE using crossvalidation: ", cv_score)
    return model

def predict_and_evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print ("RMSE on Testing set: %.2f" % math.sqrt(mean_squared_error(y_test, y_pred)))
    print ("Variance score: %.2f" % r2_score(y_test, y_pred))
    return y_pred

def tryModel(model, X_train, y_train, X_test, y_test):
    m = train_and_evaluate_regression(model, X_train, y_train)
    predict_and_evaluate_regression(m, X_test, y_test)


# keep 30% of the data outside of training
#X_train, X_test, y_train, y_test = train_test_split(X, Y['income'], test_size=0.3)

#Run both Extra Trees and Adaboost

# select columns

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import math
from sklearn.model_selection import cross_val_score

y = Y['income']

param_grid = {"n_estimators":[500],
              "max_features":[0.3, 0.5, 1],
             }
extraEstimator = ExtraTreesRegressor()


grid = GridSearchCV(extraEstimator, param_grid,n_jobs=-1, cv=8, scoring='neg_mean_squared_error', verbose=4)
grid.fit(X, y)

print('Using ExtraTrees:')
print('Best Score:', grid.best_score_) 
print('Best Params:', grid.best_params_) 

y_pred = grid.predict(X)
print("Root of Mean squared error: %.2f"
      % math.sqrt(mean_squared_error(y, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y, y_pred))

test_results = grid.predict(X_score)

T = pd.DataFrame(test_results)
T.columns = ['income']
T.to_csv('ExtraTreeResults.csv')

### USING ADABOOST

param_grid = {"base_estimator__criterion":["gini", "entropy"],
              "base_estimator__max_features":[0.3, 0.5, 1],
              "learning_rate":[0.56, 0.64]}

adaBoostEstimator = AdaBoostRegressor(DecisionTreeRegressor())

grid2 = GridSearchCV(adaBoostEstimator, param_grid,n_jobs=-1, cv=8, scoring='neg_mean_squared_error', verbose=4)
grid2.fit(X, y)

print('Using AdaBoost:')
print('Best Score:', grid2.best_score_) 
print('Best Params:', grid2.best_params_) 

y_pred = grid2.predict(X)
print("Root of Mean squared error: %.2f"
      % math.sqrt(mean_squared_error(y, y_pred)))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y, y_pred))

test_results = grid2.predict(X_score)

T = pd.DataFrame(test_results)
T.columns = ['income']
T.to_csv('AdaBoostResults.csv')

    
#print ('\nUsing LinearRegressor')
#model = linear_model.LinearRegression()
#tryModel(model, X_train, y_train, X_test, y_test)
#
#print ('\nUsing Ridge regression')
#model = linear_model.Ridge()
#tryModel(model, X_train, y_train, X_test, y_test)

#print ('\nUsing SVR with linear kernel')
#model = svm.SVR(kernel='linear')
#tryModel(model, X_train, y_train, X_test, y_test)
##
#print ('\nUsing SVR with poly kernel')
#model = svm.SVR(kernel='poly')
#tryModel(model, X_train, y_train, X_test, y_test)
#

#print ('\nUsing SVR with rbf kernel')
#model = svm.SVR(kernel='rbf')
#tryModel(model, X_train, y_train, X_test, y_test)

#print ('\nUsing ExtraTrees')
#model=ensemble.ExtraTreesRegressor()
#tryModel(model, X_train, y_train, X_test, y_test)
#
#print ('\nUsing RandomForest')
#model=ensemble.RandomForestRegressor()
#tryModel(model, X_train, y_train, X_test, y_test)
#
#print ('\nUsing AdaBoost')
#model=ensemble.AdaBoostRegressor(DecisionTreeRegressor())
#tryModel(model, X_train, y_train, X_test, y_test)
#
#print ('\nUsing GradientBoostingRegressor')
#model=ensemble.GradientBoostingRegressor()
#tryModel(model, X_train, y_train, X_test, y_test)
#
#print ('\nUsing BaggingRegressor')
#model=ensemble.BaggingRegressor(DecisionTreeRegressor())
#tryModel(model, X_train, y_train, X_test, y_test)

# Plot correlations of income with all numeric variables

# bin income into 4 groups
# put it into X_analysis

# run multiple linear algorithms
# plot results and R2 score


#do feature selection
#sklearn feature_selection
	#- Using VarianceTreshold
	#- SelectKBest
	#- SelectPercentile

# do RFECV

#svc = SVC(kernel="linear")
#	# The "accuracy" scoring is proportional to the number of correct
#	# classifications
#	rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),
#              scoring='accuracy')
#rfecv.fit(X, y)
#
#print("Optimal number of features : %d" % rfecv.n_features_)
#
#Using LassoCV feature selection
#http://scikit-learn.org/stable/auto_examples/feature_selection/plot_select_from_model_boston.html#sphx-glr-auto-examples-feature-selection-plot-select-from-model-boston-py

#Tree-based feature selection using SelectFromModel
#
#>>> from sklearn.ensemble import ExtraTreesClassifier
#>>> from sklearn.datasets import load_iris
#>>> from sklearn.feature_selection import SelectFromModel
#>>> iris = load_iris()
#>>> X, y = iris.data, iris.target
#>>> X.shape
#(150, 4)
#>>> clf = ExtraTreesClassifier()
#>>> clf = clf.fit(X, y)
#>>> clf.feature_importances_  
#array([ 0.04...,  0.05...,  0.4...,  0.4...])
#>>> model = SelectFromModel(clf, prefit=True)
#>>> X_new = model.transform(X)
#>>> X_new.shape               
#(150, 2)
#
#
#Plotting importance of forest features
#http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html#sphx-glr-auto-examples-ensemble-plot-forest-importances-py
