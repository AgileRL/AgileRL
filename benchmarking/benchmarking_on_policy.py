import gymnasium as gym
import torch

from agilerl.models import PPOSpec, TrainingSpec
from agilerl.models.hpo import (
    MutationProbabilities,
    MutationSpec,
    RLHyperparameter,
    TournamentSelectionSpec,
)
from agilerl.models.networks import MlpSpec, NetworkSpec
from agilerl.trainer import LocalTrainer
from agilerl.utils.utils import make_vect_envs


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_name = "LunarLander-v3"
    num_envs = 16

    print("============ AgileRL ============")
    print(f"DEVICE: {device}, ENV: {env_name}")

    env = make_vect_envs(make_env=lambda **_kw: gym.make(env_name), num_envs=num_envs)

    algorithm = PPOSpec(
        learn_step=128,
        num_envs=num_envs,
        lr=0.0003,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=4,
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
        eval_loop=1,
        target_score=250,
    )

    mutation = MutationSpec(
        probabilities=MutationProbabilities(
            no_mut=0.4,
            arch_mut=0.2,
            new_layer=0.2,
            params_mut=0.2,
            act_mut=0.0,
            rl_hp_mut=0.2,
        ),
        rl_hp_selection={
            "lr": RLHyperparameter(min=1e-5, max=1e-2),
            "batch_size": RLHyperparameter(min=8, max=1024),
            "learn_step": RLHyperparameter(min=64, max=2048),
        },
        mutation_sd=0.1,
        rand_seed=42,
    )

    tournament = TournamentSelectionSpec(
        tournament_size=2,
        elitism=True,
    )

    trainer = LocalTrainer(
        algorithm=algorithm,  # "PPO"
        environment=env,
        training=training,
        mutation=mutation,
        tournament=tournament,
        env_name=env_name,
        tensorboard=True,
        tensorboard_log_dir="tensorboard_logs",
        device=str(device),
    )

    trained_pop, _pop_fitnesses = trainer.train()

    if str(device) == "cuda":
        torch.cuda.empty_cache()

    env.close()


if __name__ == "__main__":
    main()
