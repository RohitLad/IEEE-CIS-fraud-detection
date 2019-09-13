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
            df[column] = assign_category(df[column])
        else:
            col_type = str(df[column].dtypes)[:3]
            c_min = df[column].min()
            c_max = df[column].max()
            if col_type == 'int':
                df[column] = reassign_int(c_min, c_max, df[column])
            elif col_type == 'flo':
                df[column] = reassign_float(c_min, c_max, df[column])
    end_memory = df.memory_usage().sum()
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_memory/1024**2, 100 * (start_memory - end_memory) / start_memory))
    return df

def assign_category(df):
    return df.fillna('Unavailable').astype('category')

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
    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
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
    return (df-mean)/var, mean, var

def calc_group_mean(df,group_of,mean_of):
    return df.groupby(group_of)[mean_of].mean().to_dict()

def map_to_groups(df,group_of,map_dict,convert_type):
    return df[group_of].map(map_dict).astype(convert_type)

def ignore_categories_nan(threshold,df,categories):
    non_ignored_non_categories = []
    ignored_non_categories = []
    n_dataset = df.shape[0]
    for elem in categories:
        n_unknown = df[elem].isnull().sum()
        ratio = float(n_unknown/n_dataset)
        if ratio < threshold:
            non_ignored_non_categories.append(elem)
        else:
            ignored_non_categories.append(elem)
    return non_ignored_non_categories, ignored_non_categories

class min_max:
    def __init__(self, min, max):
        self.min = min
        self.max = max

def min_max_scale(df, categories, min_max_dict = None):
    override = False
    if min_max_dict is None:
        override = True
        min_max_dict = {}
    df_new = pd.DataFrame()
    for elem in categories:
        if override:
            min_max_dict[elem] = min_max(min=df[elem].min(),max=df[elem].max())
        df_new[elem] = df[elem]/(min_max_dict[elem].max - min_max_dict[elem].min)

    return df_new,min_max_dict