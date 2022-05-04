from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from corecontrol.proxies import LoggerProxy

from datasets import Dataset


class Model(ABC):
    def __init__(
            self,
            name: str,
            hyper_parameters: Optional[dict] = None,
            meta_parameters: Optional[dict] = None,
            use_scaled_data: bool = True
    ):
        self.name = name
        self.hyper_parameters = hyper_parameters or {}
        self.meta_parameters = meta_parameters or {}
        self.use_scaled_data = use_scaled_data

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
        pass

    def save(self, output_path: str):
        # TODO: Should we move save/load to a higher level of abstraction?
        pass

    def load(self, model_path: str):
        pass

    def plot(self, output_dir: str):
        pass
