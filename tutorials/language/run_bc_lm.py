import hydra
from omegaconf import DictConfig, OmegaConf
from train_bc_lm import train

# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../"))


@hydra.main(config_path="configs", config_name="train_bc")
def main(cfg: DictConfig):
    cfg = OmegaConf.to_container(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
