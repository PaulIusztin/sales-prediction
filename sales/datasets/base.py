from abc import ABC, abstractmethod
from typing import Tuple, Dict

import pandas as pd


class Dataset(ABC):
    @abstractmethod
    def load(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        pass
