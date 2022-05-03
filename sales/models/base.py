from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from corecontrol.proxies import LoggerProxy

from datasets import Dataset


class Model(ABC):
    def __init__(self, name: str, hyper_parameters: Optional[dict] = None):
        self.name = name
        self.hyper_parameters = hyper_parameters or {}

        # TODO: Hook it somehow to the python logger so the logger wont be None.
        self.logger: Optional[LoggerProxy] = None

    def set_logger(self, logger: LoggerProxy):
        self.logger = logger

    @classmethod
    def from_config(cls, config: dict, *args, **kwargs) -> "Model":
        parameters = config["parameters"]

        return cls(**parameters)

    @abstractmethod
    def fit(self, dataset: Dataset) -> "Model":
        pass

    @abstractmethod
    def predict(self, X, *args, **kwargs) -> pd.Series:
        # TODO: Is is ok to let the return type be pd.Series? Maybe np.ndarray would be better.
        pass
