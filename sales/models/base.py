from abc import ABC, abstractmethod

import pandas as pd

from datasets import Dataset


class Model(ABC):
    def __init__(self, name: str):
        self.name = name

    @classmethod
    @abstractmethod
    def from_config(cls, config, *args, **kwargs) -> "Model":
        pass

    @abstractmethod
    def fit(self, dataset: Dataset) -> "Model":
        pass

    @abstractmethod
    def predict(self, X, *args, **kwargs) -> pd.Series:
        pass
