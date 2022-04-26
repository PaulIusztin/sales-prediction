import pandas as pd

from datasets import Dataset
from models.base import Model


class PersistenceModel(Model):
    def __init__(self, predict_column: str):
        super().__init__(name="persistence")

        self.predict_column = predict_column

    @classmethod
    def from_config(cls, config: dict, *args, **kwargs) -> "Model":
        return cls(**config)

    def fit(self, dataset: Dataset) -> "Model":
        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        return X[self.predict_column]
