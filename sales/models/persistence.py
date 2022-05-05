import logging
import pprint

import pandas as pd

from datasets import Dataset
from models.base import Model


logger = logging.getLogger(__name__)


class PersistenceModel(Model):
    def __init__(self, hyper_parameters: dict, ):
        from evaluators import RegressionEvaluator

        super().__init__(
            name="persistence",
            hyper_parameters=hyper_parameters,
            use_scaled_data=False
        )

        self.predict_column = self.hyper_parameters["predict_column"]
        self.evaluator = RegressionEvaluator()

    def fit(self, dataset: Dataset) -> "Model":
        train_results = self.evaluator.compute(
            model=self,
            dataset=dataset,
            split="train",
            plot=False,
            output_dir=None
        )
        validation_results = self.evaluator.compute(
            model=self,
            dataset=dataset,
            split="validation",
            plot=False,
            output_dir=None
        )
        logger.info(f"Train results: {pprint.pformat(train_results, indent=4)}")
        logger.info(f"Validation results: {pprint.pformat(validation_results, indent=4)}")
        if self.logger is not None:
            for split_name, results in (("train", train_results), ("validation", validation_results)):
                for metric_name, metric_value in results.items():
                    self.logger.report_scalar(
                        title=metric_name.upper(),
                        series=f"LinearRegression/{split_name}",
                        value=metric_value,
                        iteration=0
                    )

        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        return X[self.predict_column]
