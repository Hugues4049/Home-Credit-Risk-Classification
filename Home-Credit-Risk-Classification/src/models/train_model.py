import warnings

# %matplotlib inline
import pandas as pd
import os

# Import sklearn helper metrics and transformations
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# import library for hyperparameter optimization
# from hyperopt import STATUS_OKntity_from_dataframe

import sys

warnings.filterwarnings("ignore")
os.path.join(os.path.dirname(__file__), "../")
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
# from features.build_features import  feature_matrix

feature_matrix = pd.read_csv(
    "./Home-Credit-Risk-Classification/data/raw/feature_matrix.csv"
)

"""Data split and imbalance correction

First, we split the data into training/testing set in the ratio 75:25"""

# splitting application_train_newdf into train and test
train, test = train_test_split(feature_matrix, test_size=0.25, random_state=123)
print(len(train.columns))
# separating dependent and independent variables (no under/over sampling)
train_X = train[[i for i in train.columns if i not in ["SK_ID_CURR"] + ["TARGET"]]]
train_Y = train[["TARGET"]]

test_X = test[[i for i in test.columns if i not in ["SK_ID_CURR"] + ["TARGET"]]]
test_Y = test[["TARGET"]]


# Down-sample Majority Class

count = train["TARGET"].value_counts()
num_majority = count[0]
num_minority = count[1]

# Number of undersampled majority class 2 x minority class
num_undersample_majority = 2 * num_minority

# separating majority and minority classes
df_majority = train[train["TARGET"] == 0]
df_minority = train[train["TARGET"] == 1]

df_majority_undersampled = resample(
    df_majority, replace=False, n_samples=num_undersample_majority, random_state=123
)

df_undersampled = pd.concat([df_minority, df_majority_undersampled], axis=0)

# splitting dependent and independent variables

df_undersampled_X = df_undersampled[
    [i for i in df_undersampled.columns if i not in ["SK_ID_CURR"] + ["TARGET"]]
]
df_undersampled_Y = df_undersampled[["TARGET"]]

"""Training classifier models

First, we create a function for the classifier to train on data
predict using test data, and visualize the metrics"""
# Model function
