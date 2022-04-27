import pandas as pd
import xgboost as xgb

from datasets import Dataset
from models import Model


class XGBoostModel(Model):
    HYPER_PARAMETERS = {
        "n_estimators": 100,
        "reg_lambda": 1,
        "gamma": 0,
        "max_depth": 3
    }

    def __init__(self):
        super().__init__("xgboost")

        self.model = xgb.XGBRegressor(**self.HYPER_PARAMETERS)

    @classmethod
    def from_config(cls, config: dict, *args, **kwargs) -> "Model":
        return cls()

    def fit(self, dataset: Dataset) -> "Model":
        X_train, y_train = dataset.get(split="train")

        self.model = self.model.fit(X_train, y_train)

        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        return self.model.predict(X)
