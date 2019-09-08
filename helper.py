import numpy as np
import pandas as pd

def summary(df):
    summary = pd.DataFrame(df.types, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']

def reduce_memory_usage(df: pd.DataFrame, verbose=True):
    data_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_memory = df.memory_usage().sum()
    for column in df.columns:
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