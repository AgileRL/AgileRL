import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

from agilerl.training.train_ilql import train

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../"))


@hydra.main(config_path="configs/wordle", config_name="train_iql")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
