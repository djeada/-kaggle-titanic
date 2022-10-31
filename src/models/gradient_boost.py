import joblib
import os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from src.models.base_model import BaseModel


class GradientBoost(BaseModel):
    """
    Gradient Boosting implementation using sklearn GradientBoostingClassifier.
    """

    def __init__(
        self,
        parameters={
            "n_estimators": [10, 80, 200, 400],
            "max_depth": [2, 6, 12, 15],
            "learning_rate": [0.01, 0.1, 1, 10, 100],
        },
    ):
        self.encoder = LabelEncoder()
        gradient_boost = GradientBoostingClassifier()
        self.model = GridSearchCV(
            gradient_boost,
            parameters,
            verbose=1,
            scoring="r2",
        )

    def fit(self, x, y):
        """
        Train the model on the given data.
        :param x: The input data.
        :param y: The output data.
        :return: The trained model.
        """
        self.model.fit(x, self.encoder.fit_transform(y))

    def predict(self, x):
        """
        Predict the labels for the given data.
        :param x: The input data.
        :return: The predicted labels.
        """
        return self.encoder.inverse_transform(self.model.predict(x))

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
