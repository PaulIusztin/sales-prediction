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
    def __init__(self, models: List[Model], output_dir: str):
        self.models = models
        self.hooks: List[Hook] = [CoreControlHook()]
        self.evaluator = RegressionEvaluator()

        self.output_dir = Path(output_dir)
        logger.info(f"Runner output folder: {self.output_dir.absolute()}")

    @classmethod
    def from_config(cls, config: dict, output_dir: str = "./experiments") -> "Runner":
        models: List[Model] = []
        for model_config in config["models"]:
            model_class = model_config["name"]
            model_class = model_registry[model_class]
            model = model_class.from_config(config=model_config)

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
            logger.info(f"Fitting model: {model.name}")
            model.fit(dataset)

            logger.info(f"Plotting model: {model.name}")
            model_output_dir = self.output_dir / model.name
            model_output_dir.mkdir(parents=True, exist_ok=True)
            model.plot(output_dir=str(model_output_dir))

        for hook in self.hooks:
            hook.after_fit(runner=self)

    def test(self, dataset: Dataset) -> dict:
        results = {}
        for model in self.models:
            logger.info(f"Testing model: {model.name}")

            model_output_dir = self.output_dir / model.name
            model_output_dir.mkdir(parents=True, exist_ok=True)

            # TODO: Add a prediction line plot.
            model_results = self.evaluator.compute(
                model=model,
                dataset=dataset,
                plot=True,
                output_dir=str(model_output_dir)
            )
            for metric_name, metric_result in model_results.items():
                if model.logger is not None:
                    model.logger.report_scalar(
                        title=f"Test/{metric_name.upper()}",
                        series=model.name,
                        value=metric_result,
                        iteration=1
                    )
            results[model.name] = model_results

        return results
