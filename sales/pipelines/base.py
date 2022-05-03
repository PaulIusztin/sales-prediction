from abc import ABC, abstractmethod

import pandas as pd


class Pipeline(ABC):
    @property
    def name(self):
        return f"{self.__class__.__name__}"

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict) -> "Pipeline":
        pass

    @classmethod
    def get_class_state(cls) -> list:
        return list(cls.__dict__.keys())

    def get_state(self) -> dict:
        return self.__dict__

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
