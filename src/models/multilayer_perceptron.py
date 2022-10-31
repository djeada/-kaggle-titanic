import joblib
import os
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from src.models.base_model import BaseModel


class MultilayerPerceptron(BaseModel):
    """
    Multilayer Perceptron implementation using sklearn.
    """

    def __init__(
        self,
        parameters={
            "hidden_layer_sizes": [(10,), (50,), (100,)],
            "activation": ["logistic"],
            "solver": ["adam"],
            "learning_rate": ["adaptive"],
        },
    ):
        perceptron = MLPClassifier(max_iter=2000)
        self.model = GridSearchCV(perceptron, parameters, cv=3, scoring="accuracy")

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
