import datetime
import logging
from pathlib import Path
from typing import List

from datasets import Dataset
from evaluators import RegressionEvaluator
from hooks.base import Hook
from models import model_registry
from models.base import Model


logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, models: List[Model], output_folder: str = "../outputs"):
        self.models = models
        self.hooks: List[Hook] = []
        self.evaluator = RegressionEvaluator()

        self.runner_session = f"experiments_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.output_folder = Path(output_folder) / self.runner_session
        logger.info(f"Runner output folder: {self.output_folder.absolute()}")

    @classmethod
    def from_config(cls, config: dict, output_folder: str = "../outputs") -> "Runner":
        models: List[Model] = []
        for model_config in config["models"]:
            model_class = model_config["name"]
            model_class = model_registry[model_class]
            model_parameters = model_config["parameters"]
            model = model_class.from_config(config=model_parameters)

            models.append(model)

        return cls(models=models, output_folder=output_folder)

    def register_hook(self, hook: Hook):
        self.hooks.append(hook)

    def run(self, dataset: Dataset) -> List[Model]:
        dataset.load()

        self.fit(dataset)
        self.test(dataset)

        return self.models

    def fit(self, dataset: Dataset):
        for hook in self.hooks:
            hook.before_fit(runner=self)
        for model in self.models:
            model.fit(dataset)
        for hook in self.hooks:
            hook.after_fit(runner=self)

    def test(self, dataset: Dataset) -> dict:
        X_test, y_test = dataset.get(split="test")

        results = {}
        for model in self.models:
            model_output_folder = self.output_folder / model.name
            Path(model_output_folder).mkdir(parents=True, exist_ok=True)

            y_pred = model.predict(X=X_test)
            results[model.name] = self.evaluator.compute(
                y_test,
                y_pred,
                plot=True,
                output_folder=str(model_output_folder)
            )

        return results
