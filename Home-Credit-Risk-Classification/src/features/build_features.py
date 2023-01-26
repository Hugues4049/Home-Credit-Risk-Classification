from sys import displayhook
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pandas as pd
import seaborn as sns             
from timeit import default_timer as timer
import os
import random
import csv
import json
import itertools
import pprint
from pydash import at
import gc
import re

# import featuretools for automated feature engineering
import featuretools as ft 
from featuretools import selection

#Import sklearn helper metrics and transformations
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,roc_auc_score,classification_report,roc_curve,auc, f1_score

#Import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb

#import library for hyperparameter optimization
#from hyperopt import STATUS_OKntity_from_dataframe
from hyperopt import hp, tpe, Trials, fmin
from hyperopt.pyll.stochastic import sample




import os
import sys
os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
#from data.make_dataset import application_train
from visualization.visualize import  application_train




'''
 Feature Engineering

'''

'''Save/Load updated datasets'''

#We create a checkpoint to save the updated data files if needed

## Save the csv files as a checkpoint if necessary (high memory and time intensive)

save_files = False

if save_files == True:
    application_train.to_csv('application_train_updated.csv', index = False)
    

#This is followed by the option of loading the previously saved updated datafiles, in case we are restarting the notebook

# Load csv files from checkpoint if necessary
# Otherwise, copy to new df and clean up memory 

load_files = False  # Existing variables in memory

if load_files == False:
    application_train_new = application_train.copy()
    
    
    gc.enable()
    del application_train
    gc.collect()
    
else:
    application_train_new = pd.read_csv("application_train_updated.csv")
    

'''
Basic Feature Engineering (Replacing outliers, Imputing, One-hot encoding, Rescaling data)'''

#First, a function is created and run to replace the day outliers previously seen across the data

def replace_day_outliers(df):
    """Replace 365243 with np.nan in any columns with DAYS"""
    for col in df.columns:
        if "DAYS" in col:
            df[col] = df[col].replace({365243: np.nan})

    return df

# Replace all the day outliers
application_train_new = replace_day_outliers(application_train_new)

"""We create a function to remove columns which have missing values greater than 60%"""

#Function for removing columns with missing values more than 60%

def remove_missing_col(df):
    miss_data = pd.DataFrame((df.isnull().sum())*100/df.shape[0])
    miss_data_col=miss_data[miss_data[0]>60].index
    data_new  = df[[i for i in df.columns if i not in miss_data_col]]
    return data_new

"""Create a custom imputer function for both numerical and categorical variables"""


class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with median of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].median() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

"""One-hot encoding of categorical variables in the main training dataset is done"""

# one-hot encoding of categorical variables
base_case_train = pd.get_dummies(application_train_new)

"""After one-hot encoding, the index variable (SK_ID_CURR) is temporarily removed and 
   the main dataset has values imputed to replace missing samples, and the features are rescaled"""


   # Drop the SK_ID from the training data
skid_temp = application_train_new['SK_ID_CURR']
train = base_case_train.drop(columns = ['SK_ID_CURR'])
    
# Feature names
features = list(train.columns)

# Scale each feature to 0-1
scaler = MinMaxScaler(feature_range = (0, 1))

# Median imputation of missing values
train = DataFrameImputer().fit_transform(train)

## Repeat with the scaler
scaler.fit(train)
train = scaler.transform(train)
base_case_train = pd.DataFrame(data=train, columns=features)

print('Data shape: ', base_case_train.shape)

"""At this point, we can get the modified dataset for visual inspection. Looks ok!"""
print(base_case_train)

#The index variable is reattached to the dataset

base_case_train['SK_ID_CURR'] = skid_temp

print('Data shape: ', base_case_train.shape)

base_case_train.to_csv("./Home-Credit-Risk-Classification/data/interim/base_case_train.csv".format(pd.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")),index=False)

'''apply our limited domain knowlege to create few more variables, specifically ratios accounting for the credit income %, 
annuity income %, credit term, and fraction of years employed'''
#Domain knowledge

application_train_new['CREDIT_INCOME_PERCENT'] = application_train_new['AMT_CREDIT'] / application_train_new['AMT_INCOME_TOTAL']
application_train_new['ANNUITY_INCOME_PERCENT'] = application_train_new['AMT_ANNUITY'] / application_train_new['AMT_INCOME_TOTAL']
application_train_new['CREDIT_TERM'] = application_train_new['AMT_ANNUITY'] / application_train_new['AMT_CREDIT']
application_train_new['YEARS_EMPLOYED_PERCENT'] = application_train_new['YEARS_EMPLOYED'] / application_train_new['AGE']
displayhook(application_train_new.head(3))


''' Automated Feature Engineering using Featuretools'''
# Iterate through the columns and record the Boolean columns

def bool_type(df):

    col_type = {}

    for col in df:
        # If column is a number with only two values, encode it as a Boolean
        if (df[col].dtype != 'object') and (len(df[col].unique()) <= 2):
            col_type[col] = ft.variable_types.Boolean

    print('Number of boolean variables: ', len(col_type))
    return col_type

train_col_type = bool_type(application_train_new)
train_col_type['REGION_RATING_CLIENT'] = ft.variable_types.Ordinal
train_col_type['REGION_RATING_CLIENT_W_CITY'] = ft.variable_types.Ordinal


# Entity set with id applications
es = ft.EntitySet(id = 'clients')

# Entities with a unique index
es = es.entity_from_dataframe(entity_id = 'app', dataframe = application_train_new, index = 'SK_ID_CURR', variable_types = train_col_type)
print(es)


# List the primitives in a dataframe
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
displayhook(primitives[primitives['type'] == 'aggregation'].head(5))
displayhook(primitives[primitives['type'] == 'transform'].head(5))


# Default primitives from featuretools
default_agg_primitives =  ['sum', 'count', 'min', 'max', 'mean', 'mode']
default_trans_primitives =  ['diff', 'cum_sum', 'cum_mean', 'percentile']

# DFS with specified primitives
feature_names = ft.dfs(entityset = es, target_entity = 'app',
                       trans_primitives = default_trans_primitives,
                       agg_primitives=default_agg_primitives)
                       ##max_depth = 2, features_only=True)

print('%d Total Features' % len(feature_names))


# DFS with default primitives
print('hello je suis ')
feature_matrix, feature_names = ft.dfs(entityset = es, target_entity = 'app',
                                       trans_primitives = default_trans_primitives,
                                       agg_primitives=default_agg_primitives)
                                       ##max_depth = 2, features_only=False, verbose = True)

pd.options.display.max_columns = 3000
feature_matrix.head(5)
print('hello je suis la')
feature_matrix.to_csv("./Home-Credit-Risk-Classification/data/processed/feature_matrix.csv".format(pd.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")),index=False)
