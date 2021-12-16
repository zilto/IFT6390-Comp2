# Kaggle Competition 2 - CropHarvest

This repository contains the work realized by William Callaghan and Thierry Jean in the context of the graduate course IFT 6390 - Machine Learning Fundamentals of Fall 2021.

## Competition description
CropHarvest (Tseng et al., 2021) is a recently published dataset designed for agricultural research. The preprocessed subset received contains 60,000 data points, 18 features from satellite imaging, meteorological data, topographical profile, and vegetation, over 12 consecutive months (timesteps). We aim to train a binary classifier to predict if a location is a crop land or not.

## Approaches tried
- Gradient boosting machines using:
    - original features
    - feature aggregates over 3, 6, 9, 12 months
    - smoothed features using rolling windows of 3, 4, 6 months
- 1D Convolution neural network using:
    - with 2 fully connected layers and sigmoid activation
    - multiple kernel sizes (3, 4, 6), then concatenating representations

## How-to run the code
- The functions are implemented in the *preprocessing*, *utils*, and *model_* .py files
- The functions are imported in the runner Jupyter notebooks *gbm_runner.ipynb* and *conv1d_runner.ipynb*
- Running the cells sequentially should: load the data, preprocess it, optimize the model, train a model with optimal hyperparameters, make predictions on the test set, and save the predictions to an .csv file.

## Packages used
- numpy
- pandas
- matplotlib
- pytorch: deep learning framework
- xgboost: gradient boosting machine
- lgbm: gradient boosting machine
- sklearn: machine learning metrics and utils
- optuna: hyper parameter optimization
