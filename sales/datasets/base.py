from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import pandas as pd
from hydra.utils import to_absolute_path

from pipelines import MonthPriceSalesPipeline


class Dataset(ABC):
    def __init__(
            self,
            data_dir: str,
            pipeline: MonthPriceSalesPipeline,
    ):
        self.data_dir = Path(to_absolute_path(data_dir))
        self.pipeline = pipeline

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_{self.pipeline.__class__.__name__}"

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict, data_dir: str, *args, **kwargs) -> "Dataset":
        pass

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get(self, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass
