from typing import Optional

import pandas as pd
from sklearn.linear_model import LinearRegression

from datasets import Dataset
from models import Model


class LinearRegressionModel(Model):
    def __init__(self, hyper_parameters: Optional[dict] = None):
        super().__init__("linear_regression", hyper_parameters=hyper_parameters)

        self.model = LinearRegression()

    def fit(self, dataset: Dataset) -> "Model":
        X_train, y_train = dataset.get(split="train")

        self.model = self.model.fit(X_train, y_train)

        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        return self.model.predict(X)
