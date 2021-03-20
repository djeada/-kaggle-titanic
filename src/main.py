import os
import errno
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from time import time

from calculate_stats import CalculateStats
from linear_regression import LinearRegression
from logisitic_regression import LogisticRegression
from multilayer_perceptron import MultilayerPerceptron
from random_forest import RandomForest
from svm import SVM
from gradient_boosting import GradientBoosting

DATASET_PATH = "../datasets/train.csv"
MODELS_DIR = "../models"
RESOURCES_PATH = "../resources/"


def render_mpl_table(
    data,
    col_width=3.0,
    row_height=0.625,
    font_size=14,
    header_color="#40466e",
    row_colors=["#f1f1f2", "w"],
    edge_color="w",
    bbox=[0, 0, 1, 1],
    header_columns=0,
    ax=None,
    **kwargs
):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array(
            [col_width, row_height]
        )
        fig, ax = plt.subplots(figsize=size)
        ax.axis("off")
    mpl_table = ax.table(
        cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    return ax.get_figure(), ax


def clean_data(path):
    data_frame = pd.read_csv(path)

    # fill missing data for age
    data_frame["Age"].fillna(data_frame["Age"].mean(), inplace=True)

    # convert to numeric
    mapping = {"male": 0, "female": 1}
    data_frame["Sex"] = data_frame["Sex"].replace(mapping.keys(), mapping.values())

    data_frame.drop(["Cabin", "Embarked", "Name", "Ticket"], axis=1, inplace=True)

    save_file_name = os.path.dirname(path) + os.sep + "titanic_cleaned.csv"
    data_frame.to_csv(save_file_name, encoding="utf-8", index=False)

    return save_file_name


def split_data(path):

    data_frame = pd.read_csv(path)

    x = data_frame.loc[:, data_frame.columns != "Survived"]
    y = data_frame.loc[:, data_frame.columns == "Survived"]

    train_test_data = train_test_split(x, y, test_size=0.2, stratify=y, random_state=1)

    dir_path = os.path.dirname(path) + os.sep

    paths = [
        dir_path + file_name
        for file_name in [
            "train_features.csv",
            "test_features.csv",
            "train_labels.csv",
            "test_labels.csv",
        ]
    ]

    for data, path in zip(train_test_data, paths):
        data.to_csv(path, index=False)

    return paths


def train_models(models, path, features_path, labels_path):

    return [model(path, features_path, labels_path).get_path() for model in models]


def compare_results(models_paths, save_path, features_path, labels_path):

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)

    results = {
        "Model": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "Latency": [],
    }

    for path in models_paths:
        model = joblib.load(path)

        start = time()
        prediction = model.predict(features)
        end = time()

        results["Model"].append(os.path.splitext(os.path.basename(path))[0])
        results["Accuracy"].append(round(accuracy_score(labels, prediction.round()), 3))
        results["Precision"].append(
            round(precision_score(labels, prediction.round()), 3)
        )
        results["Recall"].append(round(recall_score(labels, prediction.round()), 3))
        results["Latency"].append(round((end - start) * 1000, 1))

    df = pd.DataFrame(results)

    fig, ax = render_mpl_table(df, header_columns=0)
    fig.savefig(os.path.join(save_path, "model_comparison.png"))


def main():

    if not os.path.isfile(DATASET_PATH):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), DATASET_PATH)

    clean_data_path = clean_data(DATASET_PATH)

    CalculateStats(clean_data_path)
    train_features, test_feature, train_labels, test_labels = split_data(
        clean_data_path
    )

    models = [
        LinearRegression,
        LogisticRegression,
        MultilayerPerceptron,
        RandomForest,
        SVM,
        GradientBoosting,
    ]
    results_paths = train_models(models, MODELS_DIR, train_features, train_labels)
    compare_results(results_paths, RESOURCES_PATH, test_feature, test_labels)


if __name__ == "__main__":
    main()
