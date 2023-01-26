# Please uncomment and install any missing packages 
# You can change the current environment first if needed - https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/

#!pip install sklearn
#!pip install warnings
#!pip install numpy
#!pip install matplotlib
#!pip install pandas
#!pip install seaborn
#!pip install timeit
#!pip install os
#!pip install random
#!pip install csv
#!pip install json
#!pip install itertools
#!pip install pprint
#!pip install pydash
#!pip install gc
#!pip install re
#!pip install featuretools
#!pip install xgboost
#!pip install lightgbm
#!pip install hyperopt

# Import required libraries/packages

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




print(os.getcwd())

# Load training data from main file

application_train = pd.read_csv("./Home-Credit-Risk-Classification/data/external/application_train.csv")
application_train.info(verbose=True, null_counts=True)

# Check dataset structures
print ("application_train     :",application_train.shape)
displayhook("application_train")
displayhook(application_train.head(3))
print("hello")