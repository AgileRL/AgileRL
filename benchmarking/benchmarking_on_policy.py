import gymnasium as gym
import torch

from agilerl.models import PPOSpec, TrainingSpec
from agilerl.models.hpo import (
    MutationSpec,
    RLHyperparameter,
    TournamentSelectionSpec,
)
from agilerl.models.networks import MlpSpec, NetworkSpec
from agilerl.training.trainer import LocalTrainer
from agilerl.utils.utils import make_vect_envs


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "LunarLander-v3"
    num_envs = 16

    print("============ AgileRL ============")
    print(f"DEVICE: {device}, ENV: {env_name}")

    env = make_vect_envs(make_env=lambda **_kw: gym.make(env_name), num_envs=num_envs)

    algorithm = PPOSpec(
        lr=0.0003,
        net_config=NetworkSpec(
            latent_dim=128,
            encoder_config=MlpSpec(hidden_size=[128]),
            head_config=MlpSpec(hidden_size=[128]),
        ),
    )

    training = TrainingSpec(
        max_steps=2_000_000,
        pop_size=4,
        evo_steps=10_000,
        eval_loop=3,
        target_score=250,
    )

    mutation = MutationSpec(
        rl_hp_selection={
            "lr": RLHyperparameter(min=1e-5, max=1e-2),
            "batch_size": RLHyperparameter(min=8, max=1024),
            "learn_step": RLHyperparameter(min=64, max=2048),
        },
        rand_seed=42,
    )

    tournament = TournamentSelectionSpec()

    trainer = LocalTrainer(
        algorithm=algorithm,
        environment=env,
        training=training,
        mutation=mutation,
        tournament=tournament,
        device=str(device),
    )

    _trained_pop, _pop_fitnesses = trainer.train(
        env_name=env_name,
        tensorboard=True,
        tensorboard_log_dir="tensorboard_logs",
    )

    if str(device) == "cuda":
        torch.cuda.empty_cache()

    env.close()


if __name__ == "__main__":
    main()
