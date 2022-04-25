import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import metrics

from trainers.base import Trainer


class MonthSalesTester:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

        self.output_folder = trainer.output_folder
        self.results_file = self.trainer.output_folder.parent / "month_sales_results.csv"
        self.model = self.trainer.model
        self.data = self.trainer.data

    def test(self):
        print(f"\nTesting {self.model.__class__.__name__}")

        x_test, y_test_gt = self.data["test"]
        y_test_predicted = self.model.predict(x_test)
        r2_score = metrics.r2_score(y_test_gt, y_test_predicted)
        rmse = metrics.mean_squared_error(y_test_gt, y_test_predicted, squared=False)
        results = pd.DataFrame({"model": [self.model.__class__.__name__], "r2": [r2_score], "rmse": [rmse]})
        self.persist_results(results)

        print(f"Results:")
        print(results)
        print("\n")

        plt.figure(figsize=(10, 10))
        plt.ylabel('Predicted')
        sns.regplot(x=y_test_gt, y=y_test_predicted, fit_reg=True, scatter_kws={"s": 100})
        plt.savefig(self.output_folder / "test_prediction.png")

    def persist_results(self, results_df: pd.DataFrame):
        if self.results_file.exists():
            old_results_df = pd.read_csv(self.results_file)
            results_df = pd.concat([old_results_df, results_df])

        results_df.to_csv(self.results_file, index=False)
