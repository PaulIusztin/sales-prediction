import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

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
        _, y_train = dataset.get(split="train")

        # self.model = ARIMA(y_train, **self.HYPER_PARAMETERS)
        # self.model.fit()

        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        # return self.model.predict(X)
        return pd.Series()
