import hydra
from omegaconf import DictConfig

import utils
from datasets.month import MonthPriceSalesDataset
from runner import Runner


@hydra.main(config_path="../configs", config_name="kaggle")
def train(config: DictConfig) -> None:
    config = utils.omega_conf_to_dict(config)

    # TODO: Take dataset class from registry.
    dataset = MonthPriceSalesDataset.from_config(config=config["dataset"])
    runner = Runner.from_config(config=config["runner"])
    runner.run(dataset)


if __name__ == "__main__":
    train()
