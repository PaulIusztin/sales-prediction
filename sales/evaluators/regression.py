import json
from pathlib import Path
from typing import Optional

import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import metrics


class RegressionEvaluator:
    def compute(self, y_true, y_pred, plot: bool = True, output_folder: Optional[str] = None) -> dict:
        r2 = metrics.r2_score(y_true, y_pred)
        rmse = metrics.mean_squared_error(y_true, y_pred, squared=False)
        results = {
            "r2": r2,
            "rmse": rmse
        }

        if output_folder is not None:
            results_file = Path(output_folder) / "results.json"
            with open(results_file, "w") as f:
                json.dump(results, f)

        if plot:
            assert output_folder is not None, "If you want to plot, you need to specify an output folder."

            self.plot(y_true, y_pred, output_folder=output_folder)

        return results

    def plot(self, y_true, y_pred, output_folder: str):
        plt.figure(figsize=(10, 10))
        plt.ylabel("Predicted")
        sns.regplot(x=y_true, y=y_pred, fit_reg=True, scatter_kws={"s": 100})
        plt.savefig(Path(output_folder) / "regplot.png")
        plt.clf()
