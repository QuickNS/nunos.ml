import pandas as pd
import numpy as np

def percentage(numerator, denomenator):
    if type(numerator) == pd.core.series.Series:
        return (numerator/denomenator*100).map('{:.1f}%'.format)
    
    elif type(numerator) == int or type(numerator) == float:
        return '{:.1f}%'.format(float(numerator)/float(denomenator)*100)
    
    else:
        print("check type")

def DisplayFeatureCompleteness(df):
    return percentage(df.count(), df.shape[0])
