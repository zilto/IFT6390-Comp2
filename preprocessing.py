import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler



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


def create_tensor(df):
    n_instances = df.shape[0]
    d_features = len(COL_BASE)
    t_timesteps = len(MONTHS)
    
    tensor = np.zeros((n_instances, d_features, t_timesteps))
    for t in range(t_timesteps):
        tensor[:, :, t] = df[COL_BY_MONTH[MONTHS[t]]].values
        
    return tensor


# aggregate same features over 3, 6, 12 months
def agg_over_months(train_df, agg_func=["mean"], freq=12):
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
    # agg_cols.append(train_df.LABELS)
    # reconstruct the dataframe from columns
    return pd.concat(agg_cols, axis=1)

def min_max_scaling(train_df, test_df):
    minmax_scaler = MinMaxScaler()
    train_df[FEATURES] = minmax_scaler.fit_transform(train_df[FEATURES])
    test_df[FEATURES] = minmax_scaler.fit_transform(test_df[FEATURES])
    return train_df, test_df