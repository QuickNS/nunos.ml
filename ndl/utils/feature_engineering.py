# YOUR CODE GOES HERE
from sklearn.model_selection import KFold

def MeanEncodingKFold(data, columns, target, splits=5, shuffle=False):
    
    prior = data[target].mean() #global mean
 
    kf = KFold(n_splits=splits, shuffle=shuffle)
    splits = kf.split(data)

    for tr_ind, val_ind in splits:
        X_tr, X_val = data.iloc[tr_ind], data.iloc[val_ind]
        for col in columns:
            t_mean = X_val[col].map(X_tr.groupby(col)[target].mean())
            X_val[col + '_mean_target'] = t_mean
        data.iloc[val_ind] = X_val

    data[col + '_mean_target'].fillna(prior, inplace=True) #fill NaNs with global mean

    return data

def MeanEncodingLOO(data, columns, target):
    prior = data[target].mean() #global mean
    data[col + '_mean_target'] = data.groupby(columns)[target].transform(lambda x: (x.sum() - x) / (x.count() - 1))
    # Fill NaNs
    data[col + '_mean_target'].fillna(prior, inplace=True) #fill NaNs with global mean
    
    return data


