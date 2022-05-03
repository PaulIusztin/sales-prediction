import pandas as pd
import xgboost as xgb

from datasets import Dataset
from models import Model


class XGBoostModel(Model):
    def __init__(self, hyper_parameters: dict):
        super().__init__("xgboost", hyper_parameters=hyper_parameters)

        self.model = xgb.XGBRegressor(**self.hyper_parameters)

    def fit(self, dataset: Dataset) -> "Model":
        X_train, y_train = dataset.get(split="train")

        self.model = self.model.fit(X_train, y_train)

        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        return self.model.predict(X)
