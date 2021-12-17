import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold


temp_df = pd.read_csv("./data/train.csv", index_col=0, nrows=5)

# Constants to select pandas columns efficiently
# all months abbreviations: jan, feb, mar, etc.
MONTHS = [m.lower() for m in pd.date_range(0, freq="M", periods=12).strftime("%b").to_list()]
# return all columns based on MONTH: train_df[COL_BY_MONTH["jan"]]
COL_BY_MONTH = {}
for month in MONTHS:
    COL_BY_MONTH[month] = [col for col in temp_df.columns if month in col]

# all features
FEATURES = [col for col in temp_df.columns if col != "LABELS"]
# all features with the month stripped: S2_B2_
COL_BASE = list({col[:-3] for col in FEATURES})
# return all columns based on FEATURE (e.g., across months): train_df[COL_BY_FEATURE["S2_B2_"]]
# alternatively, you can select the same df using: train_df[COL_BY_FEATURE[FEATURES[0]]]
COL_BY_FEATURE = {}
for feature in COL_BASE:
    COL_BY_FEATURE[feature] = [col for col in temp_df.columns if feature in col]


# aggregate same features over 3, 6, 12 months
def agg_over_months(train_df, agg_func=["mean"], freq=12, append_label=True):
    """
    agg_func: hold a list of pandas.aggregate functions
    Freq: the length of the window for aggregation: i.e, freq=6 -> two windows of 6 months
    """
    train_df = train_df.copy()
    agg_cols = []
    # iterate over all features
    for feature_id in range(len(COL_BASE)):
        # mask to only select columns associated with the feature
        feature_mask = COL_BY_FEATURE[COL_BASE[feature_id]]
        
        start_month = 0 # counter for iteration
        n_step = int(12/freq) # number of steps based on the freq parameter
        # iterate over n_steps
        for i in range(n_step):
            # for each aggregation function, apply the aggregate
            for agg in agg_func:
                # mask columns and aggregate
                agg_col = train_df[feature_mask].iloc[:, start_month:start_month+freq].aggregate(agg, axis=1)
                
                # name the new column and append to list
                name = f"{COL_BASE[feature_id]}{start_month}{start_month+freq}_{agg}"
                agg_cols.append(pd.Series(agg_col, name=name))
                                
                start_month += freq
    
    # add back the labels to the list of columns
    if "LABELS" in train_df.columns and append_label == True:
        agg_cols.append(train_df.LABELS)
    # reconstruct the dataframe from columns
    return pd.concat(agg_cols, axis=1)


def min_max_scaling(train_df, test_df):
    minmax_scaler = MinMaxScaler()
    feature_cols = [col for col in train_df.columns if col != "LABELS"]
    
    train_df[feature_cols] = minmax_scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = minmax_scaler.transform(test_df[feature_cols])
    return train_df, test_df


def rolling_window(train_df, agg_func=["mean"], window_size=3, append_label=True):
    train_df = train_df.copy()
    agg_cols = []
    # iterate over all features
    for feature_id in range(len(COL_BASE)):
        # mask to only select columns associated with the feature
        feature_mask = COL_BY_FEATURE[COL_BASE[feature_id]]
        
        for agg in agg_func:
            # mask columns and aggregate
            if agg == "mean":
                agg_col = train_df[feature_mask].rolling(window=window_size, axis=1).mean()
            elif agg == "std":
                agg_col = train_df[feature_mask].rolling(window=window_size, axis=1).std()
            elif agg == "min":
                agg_col = train_df[feature_mask].rolling(window=window_size, axis=1).min()
            elif agg == "max":
                agg_col = train_df[feature_mask].rolling(window=window_size, axis=1).max()

            # name the new column and append to list
            column_names = [f"{c}_roll_{window_size}_{agg}" for c in agg_col.columns]
            agg_col.columns = column_names
            agg_cols.append(agg_col)
                                
    
    # add back the labels to the list of columns
    if "LABELS" in train_df.columns and append_label == True:
        agg_cols.append(train_df.LABELS)
    # reconstruct the dataframe from columns
    return pd.concat(agg_cols, axis=1)


def standard_scaling(train_df, test_df):
    standard_scaler = StandardScaler()
    feature_cols = [col for col in train_df.columns if col != "LABELS"]
    
    train_df[feature_cols] = standard_scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols] = standard_scaler.transform(test_df[feature_cols])

    return train_df, test_df


def create_tensor(df):
    n_instances = df.shape[0]
    d_features = len(COL_BASE)
    t_timesteps = len(MONTHS)
    
    tensor = np.zeros((n_instances, d_features, t_timesteps))
    for t in range(t_timesteps):
        tensor[:, :, t] = df[COL_BY_MONTH[MONTHS[t]]].values
        
    return tensor


def cv_split(train_df, n_splits=3):
    x_df = train_df.loc[:, ~train_df.columns.isin(["LABELS"])]
    y_df = train_df.loc[:, "LABELS"]  
    
    eval_sets = {}
    kfold_cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for idx, (train_idx, test_idx) in enumerate(kfold_cv.split(x_df, y_df)):
        x_train, x_test = x_df.iloc[train_idx], x_df.iloc[test_idx]
        y_train, y_test = y_df.iloc[train_idx], y_df.iloc[test_idx]
        eval_sets[idx] = {"train":(x_train, y_train), "test":(x_test, y_test)}
    
    return eval_sets
