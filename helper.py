import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn import tree, metrics
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import time
from numba import jit
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

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

def make_logistic_model(X,y,X_test,model = None, random_state = 42, max_iter = 500):
    
    if model is None:
        model = LogisticRegression(random_state=random_state, solver='lbfgs', multi_class='multinomial', max_iter=max_iter).fit(X, y)
    y_test = model.predict(X_test)

    return y_test, model

def make_decisiontree_model(X,y,X_test,model = None):
    
    if model is None:
        model = tree.DecisionTreeClassifier().fit(X,y)
    y_test = model.predict(X_test)

    return y_test, model

@jit
def fast_auc(y_true, y_prob):
    
    y_true = np.asarray(y_true)
    y_true = y_true[np.argsort(y_prob)]
    nfalse = 0
    auc = 0
    n = len(y_true)
    for i in range(n):
        y_i = y_true[i]
        nfalse += (1 - y_i)
        auc += y_i * nfalse
    auc /= (nfalse * (n - nfalse))
    return auc

def eval_auc(y_true, y_pred):
    
    return 'auc', fast_auc(y_true, y_pred), True

def mse(y_true, y_pred):
    return 'custom MSE', mean_squared_error(y_true, y_pred), False

def test():
    print('LOL')

def train_model_classification(X, X_test, y, params, folds, model_type='lgb', eval_metric='auc', columns=None, plot_feature_importance=False, model=None,
                               verbose=10000, early_stopping_rounds=200, n_estimators=50000, splits=None, n_folds=3, averaging='usual', n_jobs=-1):
    
    """
    A function to train a variety of classification models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.
    
    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type
    
    """
    columns = X.columns if columns is None else columns
    n_splits = folds.n_splits if splits is None else n_folds
    X_test = X_test[columns]
    
    # to set up scoring parameters
    metrics_dict = {'auc': {'lgb_metric_name': mse,
                        'catboost_metric_name': 'AUC',
                        'sklearn_scoring_function': metrics.roc_auc_score},
                    }
    result_dict = {}
    
    if averaging == 'usual':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))
        
    elif averaging == 'rank':
        # out-of-fold predictions on train data
        oof = np.zeros((len(X), 1))

        # averaged predictions on train data
        prediction = np.zeros((len(X_test), 1))
    
    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()
    
    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}',flush=True)
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
            
        if model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, n_estimators=n_estimators, n_jobs = n_jobs)
            model.fit(X_train, y_train, 
                    eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                    verbose=verbose, early_stopping_rounds=early_stopping_rounds)
            
            y_pred_valid = model.predict_proba(X_valid)[:, 1]
            y_pred = model.predict_proba(X_test, num_iteration=model.best_iteration_)[:, 1]
            
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid)
            print(f'Fold {fold_n}. {eval_metric}: {score:.4f}.',flush=True)
            print('')
            
            y_pred = model.predict_proba(X_test)
        
        if averaging == 'usual':
            
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
            prediction += y_pred.reshape(-1, 1)

        elif averaging == 'rank':
                                  
            oof[valid_index] = y_pred_valid.reshape(-1, 1)
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))                  
            prediction += pd.Series(y_pred).rank().values.reshape(-1, 1)        
        
        if model_type == 'lgb' and plot_feature_importance:
            # feature importance
            fold_importance = pd.DataFrame()
            fold_importance["feature"] = columns
            fold_importance["importance"] = model.feature_importances_
            fold_importance["fold"] = fold_n + 1
            feature_importance = pd.concat([feature_importance, fold_importance], axis=0)

    prediction /= n_splits
    
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)),flush=True)
    
    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores
    
    if model_type == 'lgb':
        if plot_feature_importance:
            feature_importance["importance"] /= n_splits
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12))
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
            plt.title('LGB Features (avg over folds)')

            result_dict['feature_importance'] = feature_importance
            result_dict['top_columns'] = cols
        
    return result_dict


def Kfold_classifier(X,y,X_test,n_folds, model, random_state=42):

    columns = X.columns
    cv = KFold(n_splits=n_folds, random_state=random_state,shuffle=True)
    scores_valid = []
    y_preds = []
    for fold, (train_index, valid_index) in enumerate(cv.split(X)):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_valid = y[valid_index]

        model.fit(X_train,y_train)
        y_pred_valid = model.predict(X_valid)
        scores_valid.append(model.score(X_valid,y_valid))
        y_pred_test = model.predict(X_test)
        y_preds.append([y_pred_test])

    return model, scores_valid, y_preds