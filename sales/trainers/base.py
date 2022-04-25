import datetime
from abc import abstractmethod
from pathlib import Path

from datasets.base import Dataset


class Trainer:
    def __init__(self, name: str, output_folder: str, dataset: Dataset):
        self.experiment_name = f"{name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.output_folder = Path(output_folder) / self.experiment_name
        Path(self.output_folder).mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {self.output_folder.absolute()}")

        self.dataset = dataset
        self.data = self.dataset.load()
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def train(self):
        pass
