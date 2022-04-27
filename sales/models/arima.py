import pandas as pd

from datasets import Dataset
from models import Model


class ARIMAModel(Model):
    HYPER_PARAMETERS = {
        "order": (1, 1, 1),
    }

    def __init__(self):
        super().__init__("arima")

        self.model = None

    @classmethod
    def from_config(cls, config: dict, *args, **kwargs) -> "Model":
        return cls()

    def fit(self, dataset: Dataset) -> "Model":
        X_train, y_train = dataset.get(split="train")

        self.model = self.model.fit(X_train, y_train)

        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        return self.model.predict(X)
