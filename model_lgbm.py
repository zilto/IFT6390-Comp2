import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score

import lightgbm as lgbm
from lightgbm import LGBMClassifier


# compute f1 score based on probabilities and threshold
def lgbm_f1(y_true, y_proba):
    # threshold is not passed as a parameter because LGBM callbacks need only 2 arguments
    threshold=0.5
    y_pred = np.where(y_proba > threshold, 1, 0)
    eval_score = f1_score(y_true, y_pred)
    return ("f1_score", eval_score, True)


# double f1 loss
def double_soft_f1_loss(y_true, y_pred):
    # inspired from: https://towardsdatascience.com/the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems-753902c0105d
    tp = np.sum(y_pred * y_true)
    fp = np.sum(y_pred * (1-y_true))
    fn = np.sum((1-y_pred) * y_true)
    tn = np.sum((1-y_pred) * (1-y_true))
    
    soft_f1_cls1 = 2*tp / (2*tp + fn + fp + 1e-16)
    soft_f1_cls0 = 2*tn / (2*tn + fn + fp + 1e-16)
    
    cost_cls1 = 1 - soft_f1_cls1
    cost_cls0 = 1 - soft_f1_cls0
    
    cost = 0.5 * (cost_cls1 + cost_cls0)
    macro_cost = np.mean(cost)
    return ("double_soft_f1_loss", macro_cost, True)


def train_lgbm(eval_sets, **hyperparams):
    # train a single LGBM classifier
    clf = LGBMClassifier(
        objective="binary",
        n_estimators=100,
        learning_rate=0.01,
        boosting_type="gbdt",
        subsample=0.5,
        subsample_freq=1,
        num_leaves=31,
        max_depth=-1,
        boost_from_average=False,
        n_jobs=8,     
    )
    
    clf.fit(
        eval_sets[0]["train"][0], eval_sets[0]["train"][1],
        eval_set=[(eval_sets[0]["test"][0], eval_sets[0]["test"][1])],
        eval_metric=["logloss", double_soft_f1_loss, lgbm_f1],
        verbose=False,
    )
    
    return clf


def train_best_lgbm(train_df, hyperparams):
    # train LGBM on the whole training set for final predictions
    # TODO merge with train_lgbm() and handle data split or not within function
    x_train = train_df.loc[:, ~train_df.columns.isin(["LABELS"])].values
    y_train = train_df.loc[:, "LABELS"].values
    
    clf = LGBMClassifier(objective="binary", verbose=-1, **hyperparams)
    clf.fit(
        x_train,
        y_train,
        eval_metric=["logloss", "double_soft_f1_loss"],
        verbose=False,
    )
    
    return clf


def lgbm_plot_evals(model):
    # Plot every eval metrics from the lightGBM model object
    sets = list(model.evals_result_.keys())
    n_sets = len(sets)
    metrics = list(model.evals_result_[sets[0]].keys())
    n_metrics = len(metrics)
    
    fig, ax = plt.subplots(1, n_metrics)
    
    for i, seti in enumerate(sets):
        for j, metric in enumerate(metrics):
            ax[j].set_title(f"{metric}")
            ax[j].plot(model.evals_result_[seti][metric], label=f"{seti}")
            