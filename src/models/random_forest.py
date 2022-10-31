import joblib
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.models.base_model import BaseModel


class RandomForest(BaseModel):
    """
    Random Forest implementation using sklearn.
    """

    def __init__(
        self,
        parameters={"n_estimators": [5, 30, 100], "max_depth": [3, 9, 27, 42, None]},
    ):
        random_forest = RandomForestClassifier()
        self.model = GridSearchCV(random_forest, parameters, cv=3)

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
