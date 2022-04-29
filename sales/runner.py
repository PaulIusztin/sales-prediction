import datetime
import pprint
import logging

from pathlib import Path
from typing import List

from datasets import Dataset
from evaluators import RegressionEvaluator
from hooks.base import Hook
from hooks.corecontrol import CoreControlHook
from models import model_registry
from models.base import Model


logger = logging.getLogger(__name__)


class Runner:
    def __init__(self, models: List[Model], output_dir: str = "../outputs"):
        self.models = models
        self.hooks: List[Hook] = [CoreControlHook()]
        self.evaluator = RegressionEvaluator()

        self.session_name = f"experiments_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.output_dir = Path(output_dir) / self.session_name
        logger.info(f"Runner output folder: {self.output_dir.absolute()}")

    @classmethod
    def from_config(cls, config: dict, output_dir: str = "../outputs") -> "Runner":
        models: List[Model] = []
        for model_config in config["models"]:
            model_class = model_config["name"]
            model_class = model_registry[model_class]
            model_parameters = model_config["parameters"]
            model = model_class.from_config(config=model_parameters)

            models.append(model)

        return cls(models=models, output_dir=output_dir)

    def register_hook(self, hook: Hook):
        self.hooks.append(hook)

    def run(self, dataset: Dataset) -> List[Model]:
        for hook in self.hooks:
            hook.before_run(runner=self)

        dataset.load()

        self.fit(dataset)
        results = self.test(dataset)
        logger.info(f"Test results: {pprint.pformat(results, indent=4)}")

        for hook in self.hooks:
            hook.after_run(runner=self)

        return self.models

    def fit(self, dataset: Dataset):
        for hook in self.hooks:
            hook.before_fit(runner=self)
        for model in self.models:
            model.fit(dataset)
            # TODO: Find a better way to do generalize the plotting.
            if hasattr(model, "plot"):
                model_output_dir = self.output_dir / model.name
                Path(model_output_dir).mkdir(parents=True, exist_ok=True)

                model.plot(output_dir=str(model_output_dir))
        for hook in self.hooks:
            hook.after_fit(runner=self)

    def test(self, dataset: Dataset) -> dict:
        results = {}
        for model in self.models:
            model_output_dir = self.output_dir / model.name
            Path(model_output_dir).mkdir(parents=True, exist_ok=True)

            results[model.name] = self.evaluator.compute(
                model=model,
                dataset=dataset,
                plot=True,
                output_dir=str(model_output_dir)
            )

        return results
