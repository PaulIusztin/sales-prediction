from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Any, Dict

import pandas as pd
from hydra.utils import to_absolute_path

from pipelines import MonthPriceSalesPipeline


class Dataset(ABC):
    def __init__(
            self,
            pipeline: MonthPriceSalesPipeline,
            root_dir: str,
            split_info: Dict[str, Any]
    ):
        self.pipeline = pipeline
        self.root_dir = Path(to_absolute_path(root_dir))
        self.split_info = split_info

    @property
    def name(self) -> str:
        return f"{self.__class__.__name__}_{self.pipeline.__class__.__name__}"

    @classmethod
    def from_config(cls, config: dict) -> "Dataset":
        pipeline_config = config["pipeline"]
        # TODO: Take pipeline class from registry.
        pipeline = MonthPriceSalesPipeline.from_config(pipeline_config)

        return cls(
            pipeline=pipeline,
            root_dir=config["root_dir"],
            **config["parameters"]
        )

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def get(self, split: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pass
