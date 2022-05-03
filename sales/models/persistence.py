import pandas as pd

from datasets import Dataset
from models.base import Model


class PersistenceModel(Model):
    def __init__(self, hyper_parameters: dict, ):
        super().__init__(
            name="persistence",
            hyper_parameters=hyper_parameters,
            use_scaled_data=False
        )

        self.predict_column = self.hyper_parameters["predict_column"]

    def fit(self, dataset: Dataset) -> "Model":
        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        return X[self.predict_column]
