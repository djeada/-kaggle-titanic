import joblib
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import GridSearchCV


class LogisticRegression:
    """
    Logistic Regression implementation using sklearn.
    """

    def __init__(
        self,
        parameters={"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]},
    ):
        logistic_regression = LR(max_iter=1000)
        self.model = GridSearchCV(
            logistic_regression, parameters, verbose=1, scoring="r2"
        )

    def fit(self, x, y):
        """
        Train the model on the given data.
        :param x: The input data.
        :param y: The output data.
        :return: The trained model.
        """
        self.model.fit(x, y)

    def predict(self, x):
        """
        Predict the labels for the given data.
        :param x: The input data.
        :return: The predicted labels.
        """
        return self.model.predict(x)

    def save(self, path):
        """
        Serialize the model to the given path.
        :param path: The path to save the model to.
        """
        joblib.dump(self.model, path)

    def load(self, path):
        """
        Load the model from the given path.
        :param path: The path to load the model from.
        """
        self.model = joblib.load(path)
