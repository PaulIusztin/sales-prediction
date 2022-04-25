import pandas as pd

from datasets.base import Dataset
from trainers.base import Trainer


class PersistenceModelTrainer(Trainer):
    def __init__(self, output_folder: str, dataset: Dataset):
        super().__init__("persistence", output_folder, dataset)

    def load_model(self):
        class PersistenceModel:
            def predict(self, x: pd.DataFrame) -> pd.Series:
                return x.item_sales_lag_1

        return PersistenceModel()

    def train(self):
        print("Done training persistence model...")
