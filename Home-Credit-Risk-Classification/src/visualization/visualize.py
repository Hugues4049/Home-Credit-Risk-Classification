from sys import displayhook
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import sys


warnings.filterwarnings("ignore")
os.path.join(os.path.dirname(__file__), "../")
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from data.make_dataset import application_train

"""'Checking data imbalance"""
# Target data distribution
application_train["TARGET"].astype(int).plot.hist(color="forestgreen").set_xlabel(
    "Target value: 0 or 1"
)
count = application_train["TARGET"].value_counts()
num_repaid = count[0]
num_default = count[1]
print(
    "There are {} loans repaid on time (TARGET=0) and {} loans defaulted (TARGET=1) in the dataset".format(
        num_repaid, num_default
    )
)
# Checking missing data
# Missing data in application (main) dataset
# fig = plt.figure(figsize=(18,6))
# miss_data = pd.DataFrame((application_train.isnull().sum())*100/application_train.shape[0]).reset_index()
# miss_data["type"] = "application data"
# sns.set_style('darkgrid')
# ax = sns.pointplot("index",0,data=miss_data, color='coral')
# plt.xticks(rotation =90,fontsize =7)
# plt.title("Percentage of Missing values in application data")
# plt.ylabel("percentage missing")
# plt.xlabel("columns")
# plt.ylim((0,100))
# plt.show()

"""Examine feature correlations and distributions"""


def plot_bar_gen(feature, df=None, orientation_horizontal=True):
    if df is None:
        df = application_train
    else:
        df = df
    temp = df[feature].value_counts()
    df1 = pd.DataFrame({feature: temp.index, "Number of contracts": temp.values})
    # Calculate the percentage of target=1 per category value
    cat_perc = df[[feature, "TARGET"]].groupby([feature], as_index=False).mean()
    cat_perc.sort_values(by="TARGET", ascending=False, inplace=True)
    sns.set_color_codes("colorblind")
    if orientation_horizontal is True:
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        s1 = sns.barplot(y=feature, x="Number of contracts", data=df1)
        plt.subplot(122)
        s2 = sns.barplot(y=feature, x="TARGET", data=cat_perc)
        plt.xlabel("Fraction of loans defaulted", fontsize=12)
        plt.ylabel(feature, fontsize=12)

    else:
        plt.figure(figsize=(10, 18))
        plt.subplot(211)
        s1 = sns.barplot(x=feature, y="Number of contracts", data=df1)
        s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
        plt.subplot(212)
        s2 = sns.barplot(x=feature, y="TARGET", data=cat_perc)
        s2.set_xticklabels(s2.get_xticklabels(), rotation=90)
        plt.ylabel("Fraction of loans defaulted", fontsize=12)
        plt.xlabel(feature, fontsize=12)

    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.subplots_adjust(wspace=0.6)
    plt.show()


# Plot distribution of one feature with or without paid/default shown separately


def plot_distribution_gen(feature, df=None, separate_target=False):
    if df is None:
        df = application_train
    else:
        df = df
    if separate_target is False:
        plt.figure(figsize=(10, 6))
        plt.title("Distribution of %s" % feature)
        sns.distplot(df[feature].dropna(), color="red", kde=True, bins=100)
    else:
        t1 = df.loc[df["TARGET"] != 0]
        t0 = df.loc[df["TARGET"] == 0]

        plt.figure(figsize=(10, 6))
        plt.title("Distribution of %s" % feature)
        sns.set_style("whitegrid")
        sns.kdeplot(df.loc[df["TARGET"] == 0, feature], label="target == 0")
        sns.kdeplot(df.loc[df["TARGET"] == 1, feature], label="target == 1")

        plt.ylabel("Density plot", fontsize=12)
        plt.xlabel(feature, fontsize=12)
        plt.legend(
            loc="best", labels=["Loan repaid (TARGET=0)", "Loan defaulted (TARGET=1)"]
        )
        locs, labels = plt.xticks()
        plt.tick_params(axis="both", which="major", labelsize=12)
    plt.show()


# The function below can be used to identify outliers in the data distribution
def get_thresh(feature):
    """Outliers are usually > 3 standard deviations away from the mean."""
    ave = np.mean(application_train[feature])
    sdev = np.std(application_train[feature])
    threshold = round(ave + (3 * sdev), 2)
    print("Threshold for", feature, ":", threshold)
    return threshold


"""We can find the thresholds for outliers for the days employed and total income features, and replace the anomalous values with NaN"""
thresh_income = get_thresh("AMT_INCOME_TOTAL")
thresh_employment = get_thresh("DAYS_EMPLOYED")

anomalous_employment = application_train[application_train["DAYS_EMPLOYED"] > 0]
normal_employment = application_train[application_train["DAYS_EMPLOYED"] < 0]

print(
    "The non-anomalies default on %0.2f%% of loans"
    % (100 * normal_employment["TARGET"].mean())
)
print(
    "The anomalies default on %0.2f%% of loans"
    % (100 * anomalous_employment["TARGET"].mean())
)
print("There are %d anomalous days of employment" % len(anomalous_employment))
# Replace the anomalous values with nan

application_train["DAYS_EMPLOYED"].mask(
    application_train["DAYS_EMPLOYED"] > 0, inplace=True
)
application_train["AMT_INCOME_TOTAL"].mask(
    application_train["AMT_INCOME_TOTAL"] > thresh_income, inplace=True
)

"""The below function can be used to convert features which have days to years. The function can then be applied on all such features"""


def create_day_to_year(df, ls_cols, newcol):
    df[newcol] = round(np.abs(df[ls_cols[0]] / 365))
    df.drop(columns=ls_cols, inplace=True)
    return df


create_day_to_year(application_train, ["DAYS_BIRTH"], "AGE")
create_day_to_year(application_train, ["DAYS_EMPLOYED"], "YEARS_EMPLOYED")
create_day_to_year(application_train, ["DAYS_ID_PUBLISH"], "YEARS_ID_PUBLISH")
create_day_to_year(application_train, ["DAYS_REGISTRATION"], "YEARS_REGISTRATION")
# Create INCOME_BAND to group individuals per income range


def create_income_band(df):
    df.loc[(df.AMT_INCOME_TOTAL < 30000), "INCOME_BAND"] = 1
    df.loc[
        (df.AMT_INCOME_TOTAL >= 30000) & (df.AMT_INCOME_TOTAL < 65000), "INCOME_BAND"
    ] = 2
    df.loc[
        (df.AMT_INCOME_TOTAL >= 65000) & (df.AMT_INCOME_TOTAL < 95000), "INCOME_BAND"
    ] = 3
    df.loc[
        (df.AMT_INCOME_TOTAL >= 95000) & (df.AMT_INCOME_TOTAL < 130000), "INCOME_BAND"
    ] = 4
    df.loc[
        (df.AMT_INCOME_TOTAL >= 130000) & (df.AMT_INCOME_TOTAL < 160000), "INCOME_BAND"
    ] = 5
    df.loc[
        (df.AMT_INCOME_TOTAL >= 160000) & (df.AMT_INCOME_TOTAL < 190000), "INCOME_BAND"
    ] = 6
    df.loc[
        (df.AMT_INCOME_TOTAL >= 190000) & (df.AMT_INCOME_TOTAL < 220000), "INCOME_BAND"
    ] = 7
    df.loc[
        (df.AMT_INCOME_TOTAL >= 220000) & (df.AMT_INCOME_TOTAL < 275000), "INCOME_BAND"
    ] = 8
    df.loc[
        (df.AMT_INCOME_TOTAL >= 275000) & (df.AMT_INCOME_TOTAL < 325000), "INCOME_BAND"
    ] = 9
    df.loc[(df.AMT_INCOME_TOTAL >= 325000), "INCOME_BAND"] = 10
    return df


create_income_band(application_train)
application_train.drop(columns=["INCOME_BAND"], inplace=True)
displayhook(application_train.head(3))
