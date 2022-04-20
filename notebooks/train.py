import datetime
from pathlib import Path
from typing import Tuple, Dict

import lightgbm
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn import metrics

import data


class LightGBMTrainer:
    FEATURES = [
        {
            "name": "time",
        },
        {
            "name": "revenue"
        },
        {
            "name": "is_new_item"
        },
        {
            "name": "is_first_shop_transaction"
        },
        {
            "name": "category_sales"
        },
        {
            "name": "daily_lags",
            "parameters": {
                "columns": ["item_cnt_day", "shop_category_cnt_day"],
                "lags": [1, 2, 4, 5, 10, 20, 30, 35, 65, 100],
                "fill_value": 0
            }
        }
    ]
    DROP_COLUMNS = [
        "item_name",
        "item_category_name",
        "shop_name",
        "city_name",
        "date",
        "date_block_num"
    ]
    DROP_FROM = datetime.datetime(year=2013, month=4, day=11)  # Drop the first 100 days of data, because of the lags.
    HYPER_PARAMETERS = {
        "num_leaves": 966,
        "cat_smooth": 45.01680827234465,
        "min_child_samples": 27,
        "min_child_weight": 0.021144950289224463,
        "max_bin": 214,
        "learning_rate": 0.01,
        "subsample_for_bin": 300000,
        "min_data_in_bin": 7,
        "colsample_bytree": 0.8,
        "subsample": 0.6,
        "subsample_freq": 5,
        "n_estimators": 8000,
    }
    CATEGORICAL_FEATURES = [
        "item_id",
        "shop_id",
        "city_id",
        "country_part",
        "item_category_id",
    ]
    META_FEATURES = {
        "early_stopping": 30
    }

    def __init__(self):
        self.experiment_name = f"lightgbm_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        self.output_folder = Path("../outputs") / self.experiment_name
        self.output_folder.mkdir(parents=True, exist_ok=True)
        print(f"Output folder: {self.output_folder.absolute()}")

        self.setup()
        self.data = self.load_data()
        self.model = self.load_model()

    def setup(self):
        pass

    def load_data(self) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        feature_names = [f["name"] for f in self.FEATURES]
        df = data.try_load_from_cache(features=feature_names)
        if df is None:
            df = data.load()
            df = data.pre_clean(df)
            df = data.add_features(df=df, features=self.FEATURES)
            data.save_to_cache(df, features=feature_names)
        x, y = data.split_features_labels(df, label_columns=["item_cnt_day"])
        splits = data.split(x=x, y=y)
        splits = data.scale(splits)

        for split_name, (x, y) in splits.items():
            x, y = data.post_clean(
                x, y,
                x_drop_columns=self.DROP_COLUMNS,
                drop_from=self.DROP_FROM
            )
            splits[split_name] = (x, y)

        return splits

    def load_model(self):
        return lightgbm.LGBMRegressor(**self.HYPER_PARAMETERS)

    def train(self):
        x_train, y_train = self.data["train"]

        eval_set = [self.data["train"], self.data["validation"]]
        categorical_features = [c for c in self.CATEGORICAL_FEATURES if c in x_train.columns]

        self.model = self.model.fit(
            x_train,
            y_train,
            eval_set=eval_set,
            eval_metric=["rmse"],
            verbose=100,
            categorical_feature=categorical_features,
            early_stopping_rounds=self.META_FEATURES["early_stopping"]
        )

        lightgbm.plot_importance(
            self.model,
            figsize=(20, 50),
            height=0.7,
            importance_type="gain",
            max_num_features=50
        )
        plt.savefig(self.output_folder / "feature_importance.png")

    def test(self):
        x_test, y_test_gt = self.data["test"]

        y_test_predicted = self.model.predict(x_test)
        print(f"Test R2 Score: {metrics.r2_score(y_test_gt, y_test_predicted)}")
        print(f"Test RMSE: {metrics.mean_squared_error(y_test_gt, y_test_predicted, squared=False)}")

        plt.figure(figsize=(10, 10))
        plt.ylabel('Predicted')
        sns.regplot(x=y_test_gt, y=y_test_predicted, fit_reg=True, scatter_kws={"s": 100})
        plt.savefig(self.output_folder / "test_prediction.png")


if __name__ == "__main__":
    trainer = LightGBMTrainer()
    trainer.train()
    trainer.test()
