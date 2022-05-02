from abc import ABC, abstractmethod

import pandas as pd


class Pipeline(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "Pipeline":
        pass

    @classmethod
    def get_class_state(cls) -> list:
        return list(cls.__dict__.keys())

    def get_state(self) -> list:
        return list(self.__dict__.keys())

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
