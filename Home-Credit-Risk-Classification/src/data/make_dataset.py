# Import required libraries/packages
from sys import displayhook
import warnings

# %matplotlib inline
import pandas as pd

# Import sklearn helper metrics and transformations
warnings.filterwarnings("ignore")
# Load training data from main file
application_train = pd.read_csv(
    "./Home-Credit-Risk-Classification/data/external/application_train.csv"
)
application_train.info(verbose=True, null_counts=True)
# Check dataset structures
print("application_train     :", application_train.shape)
displayhook("application_train")
displayhook(application_train.head(3))
