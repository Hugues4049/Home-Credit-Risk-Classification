import warnings
import numpy as np
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_curve,
    auc,
    f1_score,
)

# Import required libraries/packages

import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

# Import models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
import mlflow
import mlflow.sklearn
import sys
import logging
from urllib.parse import urlparse

os.path.join(os.path.dirname(__file__), "../")
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from models.train_model import *


warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Model function


def model_run(algorithm, dtrain_X, dtrain_Y, dtest_X, dtest_Y, cols=None):

    algorithm.fit(dtrain_X[cols], dtrain_Y)
    predictions = algorithm.predict(dtest_X[cols])
    name = (
        "./Home-Credit-Risk-Classification/predictions/predictions-from-{}.csv".format(
            algorithm
        )
    )
    name = name.split("(").pop(0) + ".csv"
    np.savetxt(name, predictions, delimiter=",")
    prediction_probabilities = algorithm.predict_proba(dtest_X[cols])[:, 1]
    accuracy = accuracy_score(dtest_Y, predictions)
    classify_metrics = classification_report(dtest_Y, predictions)
    f1 = f1_score(dtest_Y, predictions)
    fpr, tpr, thresholds = roc_curve(dtest_Y, prediction_probabilities)
    auc_score = auc(fpr, tpr)
    print(algorithm)
    print("Accuracy score : ", accuracy)
    print("F1 score : ", f1)
    print("AUC : ", auc_score)
    print("classification report :\n", classify_metrics)

    return accuracy, classify_metrics, fpr, tpr, auc_score, f1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    num_test = 5
    n_estimators_rfc = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth_rfc = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    max_depth_dtc = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    criterion_dtc = sys.argv[4] if len(sys.argv) > 4 else "entropy"

    with mlflow.start_run():
        """Model 1: Logistic Regression"""
        logit = LogisticRegression()
        (
            accuracy_logit,
            classify_metrics_logit,
            fpr_logit,
            tpr_logit,
            auc_score_logit,
            f1_logit,
        ) = model_run(
            logit, df_undersampled_X, df_undersampled_Y, test_X, test_Y, train_X.columns
        )

        """Model 2: Random Forest Classifier"""

        rfc = RandomForestClassifier(
            n_estimators=n_estimators_rfc, max_depth=max_depth_rfc, random_state=0
        )
        (
            accuracy_rfc,
            classify_metrics_rfc,
            fpr_rfc,
            tpr_rfc,
            auc_score_rfc,
            f1_rfc,
        ) = model_run(
            rfc, df_undersampled_X, df_undersampled_Y, test_X, test_Y, train_X.columns
        )

        """Model 3: Decision Tree Classifier"""
        dtc = DecisionTreeClassifier(
            criterion=criterion_dtc, max_depth=max_depth_dtc, random_state=42
        )
        (
            accuracy_dtc,
            classify_metrics_dtc,
            fpr_dtc,
            tpr_dtc,
            auc_score_dtc,
            f1_dtc,
        ) = model_run(
            dtc, df_undersampled_X, df_undersampled_Y, test_X, test_Y, train_X.columns
        )

        """Model 4:  XGBoost Classifier"""
        xgb = XGBClassifier()
        (
            accuracy_xgb,
            classify_metrics_xgb,
            fpr_xgb,
            tpr_xgb,
            auc_score_xgb,
            f1_xgb,
        ) = model_run(
            xgb, df_undersampled_X, df_undersampled_Y, test_X, test_Y, train_X.columns
        )

        """Model 5: Gradient Boosting Classifier"""
        gbc = GradientBoostingClassifier()
        (
            accuracy_gbc,
            classify_metrics_gbc,
            fpr_gbc,
            tpr_gbc,
            auc_score_gbc,
            f1_gbc,
        ) = model_run(
            gbc, df_undersampled_X, df_undersampled_Y, test_X, test_Y, train_X.columns
        )
        #
        classifier_names = [
            "Logistic Regression",
            "Random Forest",
            "Decision Tree",
            "XGBoost",
            "Gradient Boosting",
        ]
        accuracy_scores = [
            accuracy_logit,
            accuracy_rfc,
            accuracy_dtc,
            accuracy_xgb,
            accuracy_gbc,
        ]
        f1_scores = [f1_logit, f1_rfc, f1_dtc, f1_xgb, f1_gbc]
        auc_scores = [
            auc_score_logit,
            auc_score_rfc,
            auc_score_dtc,
            auc_score_xgb,
            auc_score_gbc,
        ]
        classify_metrics = [
            classify_metrics_logit,
            classify_metrics_rfc,
            classify_metrics_dtc,
            classify_metrics_xgb,
            classify_metrics_gbc,
        ]
        models = [logit, rfc, dtc, xgb, gbc]
        # logs
        mlflow.log_param("n_estimators", n_estimators_rfc)
        mlflow.log_param("max_depth_rfc", max_depth_rfc)
        mlflow.log_param("max_depth_dtc", max_depth_dtc)
        mlflow.log_param("criterion", criterion_dtc)
        mlflow.log_metric("accuracy_score_logit", accuracy_logit)
        mlflow.log_metric("f1_score_logit", f1_logit)

        mlflow.log_metric("accuracy_score_rfc", accuracy_rfc)
        mlflow.log_metric("f1_score_rfc", f1_rfc)

        mlflow.log_metric("accuracy_score_dtc", accuracy_dtc)
        mlflow.log_metric("f1_score_dtc", f1_dtc)

        mlflow.log_metric("accuracy_score_xgb", accuracy_xgb)
        mlflow.log_metric("f1_score_xgb", f1_xgb)

        mlflow.log_metric("accuracy_score_gbc", accuracy_gbc)
        mlflow.log_metric("f1_score_gbc", f1_gbc)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    sns.set_color_codes("colorblind")

    plt.figure(figsize=(15, 18))
    plt.subplot(311)
    plt.title("Classifier Comparison Scores: Accuracy, F1, ROC AUC")
    s1 = sns.barplot(x=classifier_names, y=accuracy_scores)
    s1.set_xticklabels(s1.get_xticklabels(), rotation=90)
    # s1.ylabel('accuracy scores', fontsize=12)
    plt.ylabel("Accuracy scores", fontsize=12)
    plt.subplot(312)
    s2 = sns.barplot(x=classifier_names, y=f1_scores)
    s2.set_xticklabels(s2.get_xticklabels(), rotation=90)
    # s2.ylabel('F1 scores', fontsize=12)
    plt.ylabel("F1 scores", fontsize=12)
    plt.subplot(313)
    s3 = sns.barplot(x=classifier_names, y=auc_scores)
    s3.set_xticklabels(s3.get_xticklabels(), rotation=90)
    plt.ylabel("AUC ROC scores", fontsize=12)

    plt.tick_params(axis="both", which="major", labelsize=12)
    plt.subplots_adjust(hspace=0.5)
    plt.show()

    ig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)

    ax.plot(
        fpr_logit,
        tpr_logit,
        label=[classifier_names[0], "AUC ROC :", round(auc_score_logit, 3)],
        linewidth=2,
        linestyle="dotted",
    )
    ax.plot(
        fpr_rfc,
        tpr_rfc,
        label=[classifier_names[1], "AUC ROC :", round(auc_score_rfc, 3)],
        linewidth=2,
        linestyle="dotted",
    )
    ax.plot(
        fpr_dtc,
        tpr_dtc,
        label=[classifier_names[2], "AUC ROC :", round(auc_score_dtc, 3)],
        linewidth=2,
        linestyle="dotted",
    )
    ax.plot(
        fpr_xgb,
        tpr_xgb,
        label=[classifier_names[4], "AUC ROC :", round(auc_score_xgb, 3)],
        linewidth=2,
        linestyle="dotted",
    )
    ax.plot(
        fpr_gbc,
        tpr_gbc,
        label=[classifier_names[5], "AUC ROC :", round(auc_score_gbc, 3)],
        linewidth=2,
        linestyle="dotted",
    )

    ax.plot([0, 1], [0, 1], linewidth=2, linestyle="dashed")
    plt.legend(loc="best")
    plt.title("ROC-CURVE & AREA UNDER CURVE")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.savefig(
        "/home/roland/Bureau/M2 DATA/Application of bigdata 2/Home-Credit-Risk-Classification/Figure_2.png"
    )
