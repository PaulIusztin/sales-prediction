from abc import ABC, abstractmethod
from typing import Tuple, Dict, Optional

import pandas as pd


class Dataset(ABC):
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
