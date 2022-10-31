from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.models.gradient_boost import GradientBoost
from src.models.linear_regression import LinearRegression
from src.models.logisitic_regression import LogisticRegression
from src.models.multilayer_perceptron import MultilayerPerceptron
from src.models.random_forest import RandomForest
from src.models.svm import SVM
from src.preprocessing.clean_data import (
    clean_dataset,
    FillMissingValuesFilter,
    EncodeCategoricalVariablesFilter,
    DropFeaturesFilter,
)
from src.preprocessing.split_dataset import split_dataset

DATASET_PATH = "../data/train.csv"
TEST_DATASET_PATH = "../data/test.csv"

FEATURES_TO_DROP = ["Cabin", "Embarked", "Name", "Ticket", "PassengerId"]
LABELS_HEADERS = ["Survived"]


def main():
    print("Preprocessing...")
    output_dir = Path("../output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Cleaning dataset...")
    data_frame = pd.read_csv(DATASET_PATH)

    clean_data_frame = clean_dataset(
        data_frame,
        [
            DropFeaturesFilter(FEATURES_TO_DROP),
            EncodeCategoricalVariablesFilter(),
            FillMissingValuesFilter(),
        ],
    )

    print("Dataset cleaned.")

    # Split the dataset into train and test sets

    print("Splitting dataset...")
    y_dataset = clean_data_frame[LABELS_HEADERS]
    x_dataset = clean_data_frame.drop(LABELS_HEADERS, axis=1)

    dataset_split = split_dataset(x_dataset, y_dataset, save_to_file=True)

    print("Preprocessing finished.")

    ### Training (ready dataset -> train -> model)
    print("Training...")

    model_types = [
        LinearRegression,
        MultilayerPerceptron,
        RandomForest,
        SVM,
        LogisticRegression,
        GradientBoost,
    ]
    models = []

    for model_type in model_types:
        print(f"Training {model_type.__name__}...")
        model = model_type()
        model.fit(dataset_split.train_x, dataset_split.train_y)
        models.append(model)

    print("Training finished.")

    ### Testing (model -> test -> metrics)
    print("Testing...")
    scores = {}
    for model in models:
        print(f"Testing {model.__class__.__name__}...")
        predictions = model.predict(dataset_split.test_x)
        accuracy = accuracy_score(dataset_split.test_y.to_numpy(), predictions)
        precision = precision_score(dataset_split.test_y, predictions)
        recall = recall_score(dataset_split.test_y, predictions)
        scores[model.__class__.__name__] = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")

    print("Testing finished.")

    print("The best model is:")
    best_model = max(scores, key=lambda key: scores[key]["accuracy"])
    print(best_model)

    print("Saving model...")
    model.save(output_dir / f"{best_model}.model")
    print("Model saved.")

    test_data_frame = pd.read_csv(TEST_DATASET_PATH)
    test_data_frame = clean_dataset(
        test_data_frame,
        [
            DropFeaturesFilter(FEATURES_TO_DROP),
            EncodeCategoricalVariablesFilter(),
            FillMissingValuesFilter(),
        ],
    )
    test_data_frame = test_data_frame.drop(LABELS_HEADERS, axis=1)
    predictions = model.predict(test_data_frame)

    # load the test dataset again, because we need the PassengerId column
    test_data_frame = pd.read_csv(TEST_DATASET_PATH)
    test_data_frame["Survived"] = predictions
    test_data_frame[["PassengerId", "Survived"]].to_csv(
        output_dir / "submission.csv", index=False
    )


if __name__ == "__main__":
    main()
