import hydra
from omegaconf import DictConfig, OmegaConf
from train_ilql import train


@hydra.main(config_path="configs", config_name="train_iql")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
