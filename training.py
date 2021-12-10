import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

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