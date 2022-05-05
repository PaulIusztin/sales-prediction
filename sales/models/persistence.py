import pandas as pd

from datasets import Dataset
from models.base import Model


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
        if self.logger is not None:
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
            for split_name, results in (("train", train_results), ("validation", validation_results)):
                for metric_name, metric_value in results.items():
                    self.logger.report_scalar(
                        title=metric_name.upper(),
                        series=f"PersistenceModel/{split_name}",
                        value=metric_value,
                        iteration=0
                    )

        return self

    def predict(self, X, *args, **kwargs) -> pd.Series:
        return X[self.predict_column]
