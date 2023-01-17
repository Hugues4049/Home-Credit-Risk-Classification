
import os
import sys
os.path.join(os.path.dirname(__file__), '../')
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from data.make_dataset import *




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

base_case_train.to_csv("./Home-Credit-Risk-Classification/data/interim/base_case_train.csv")