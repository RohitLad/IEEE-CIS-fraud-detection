import numpy as np
import pandas as pd
from scipy import stats

def summary(df):
    summary = pd.DataFrame(df.types, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']

def reduce_memory_usage(df: pd.DataFrame, verbose=True,categories=[]):
    data_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_memory = df.memory_usage().sum()
    for column in df.columns:
        if column in categories:
            df[column] = df[column].fillna('Unavailable').astype('category')
        else:
            col_type = df[column].dtypes
            if col_type in data_types:
                c_min = df[column].min()
                c_max = df[column].max()
                if str(col_type)[:3] == 'int':
                    df[column] = reassign_int(c_min, c_max, df[column])
                else:
                    df[column] = reassign_float(c_min, c_max, df[column])
    end_memory = df.memory_usage().sum()
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_memory/1024**2, 100 * (start_memory - end_memory) / start_memory))
    return df

def reassign_int(c_min, c_max, val):
    
    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
        return val.astype(np.int8)
    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
        return val.astype(np.int16)
    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
        return val.astype(np.int32)
    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
        return val.astype(np.int64)

def reassign_float(c_min, c_max, val):
    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
        return val.astype(np.float16)
    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
        return val.astype(np.float32)
    else:
        return val.astype(np.float64)    

def calc_outliers(df_num, cut_factor):

    mean, std = np.mean(df_num), np.std(df_num)
    cut = std*cut_factor
    lower, upper = mean-cut, mean+cut

    lower_outliers = []
    higher_outliers = []
    removed_outliers = []
    for val in df_num:
        if val< lower:
            lower_outliers.append(val)
        elif val > upper:
            higher_outliers.append(val)
        else:
            removed_outliers.append(val)

    len_lower = len(lower_outliers)
    len_upper = len(higher_outliers)
    len_total = len_lower+len_upper
    len_removed = len(removed_outliers)
    print('Identified lowest outliers: %d' % len_lower)
    print('Identified higher outliers: %d' % len_upper)
    print('Total outliers: %d' % len_total)
    print('Total percentual of outliers: ', round((len_total/len_removed)*100,4))


def summarize(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary

def normalize(df):
    mean=df.mean()
    var=df.var()
    print(mean)
    print(var)
    return (df-mean)/var
