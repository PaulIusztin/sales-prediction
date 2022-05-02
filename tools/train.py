import hydra
from omegaconf import DictConfig

from datasets.month import MonthPriceSalesDataset
from runner import Runner


@hydra.main(config_path="../configs", config_name="kaggle")
def train(config: DictConfig) -> None:
    config = {**config}

    data_dir = config["path"]["dataset"]

    dataset = MonthPriceSalesDataset.from_config(
        config=config,
        data_dir=data_dir
    )
    runner = Runner.from_config(config=config)
    runner.run(dataset)


if __name__ == "__main__":
    train()
