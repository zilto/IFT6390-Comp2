# LightGBM tuning
https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5

Hyperparameter to tune:
1. structure - max_depth: 3 to 12
2. structure - num_leaves: 2^(max_depth) (or 8 to 4096) -> conservative (20 to 3000)
3. structure - min_data_in_leaf: can try value in the 100s
4. accuracy - learning_rate: 0.01 to 0.3 (or lower)
5. accuracy - n_estimators: start with 100 (balance high n_estimators and low learning_rate)
6. overfitting - lambda_l1, lambda_l2: 0 to 100 (difficult to predict effect on training)
7. overfitting - min_gain_to_split: 0 to 15
8. overfitting - bagging_fraction, bagging_freq: 0 to 1
9. overfitting - feature_fraction: 0 to 1