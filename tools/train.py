import hydra
from omegaconf import DictConfig

import utils
from datasets.month import MonthPriceSalesDataset
from runner import Runner


@hydra.main(config_path="../configs", config_name="kaggle")
def train(config: DictConfig) -> None:
    config = utils.omega_conf_to_dict(config)

    data_dir = config["path"]["dataset"]

    # TODO: Take dataset class from registry.
    dataset = MonthPriceSalesDataset.from_config(
        config=config,
        data_dir=data_dir
    )
    runner = Runner.from_config(config=config)
    runner.run(dataset)


if __name__ == "__main__":
    train()
