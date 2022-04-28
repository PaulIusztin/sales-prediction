from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from corecontrol.proxies import LoggerProxy

from datasets import Dataset


class Model(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger: Optional[LoggerProxy] = None

    def set_logger(self, logger: LoggerProxy):
        self.logger = logger

    @classmethod
    @abstractmethod
    def from_config(cls, config, *args, **kwargs) -> "Model":
        pass

    @abstractmethod
    def fit(self, dataset: Dataset) -> "Model":
        pass

    @abstractmethod
    def predict(self, X, *args, **kwargs) -> pd.Series:
        # TODO: Is is ok to let the return type be pd.Series? Maybe np.ndarray would be better.
        pass
