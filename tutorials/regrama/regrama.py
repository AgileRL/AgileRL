# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""ReGraMa dormant-neuron resets with PPO on LunarLander-v3.

Runnable companion to the ReGraMa documentation tutorial. It builds the environment,
population, selection strategy and mutations in Python and hands them to
train_on_policy, which runs the ordinary evolutionary HPO loop.

The manifest alternative is two lines (see ``configs/training/ppo/ppo_regrama.yaml``)::

    from agilerl import LocalTrainer

    population, fitnesses = LocalTrainer.from_manifest(
        "configs/training/ppo/ppo_regrama.yaml"
    ).train()
"""

import torch

from agilerl.algorithms import PPO
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import make_vect_envs

if __name__ == "__main__":
    print("===== AgileRL ReGraMa (PPO) Demo =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a vectorised environment and read the observation/action spaces off it
    num_envs = 16
    env = make_vect_envs("LunarLander-v3", num_envs=num_envs)
    observation_space = env.single_observation_space
    action_space = env.single_action_space

    # A ReLU MLP with two 64-node hidden layers. ReGraMa works with any activation though.
    net_config = {"head_config": {"hidden_size": [64, 64], "activation": "ReLU"}}

    # RL hyperparameters that mutation is allowed to tune, with their search ranges
    hp_config = HyperparameterConfig(
        lr=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=1024),
        learn_step=RLParameter(min=256, max=8192, grow_factor=1.5, shrink_factor=0.75),
        ent_coef=RLParameter(min=1e-3, max=1e-1, grow_factor=1.0, shrink_factor=0.9),
    )

    # Algorithm hyperparameters
    init_hp = {
        "batch_size": 128,
        "lr": 0.001,
        "learn_step": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "action_std_init": 0.6,
        "clip_coef": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "update_epochs": 4,
        "num_envs": num_envs,
        "hp_config": hp_config,
        "net_config": net_config,
    }

    population_size = 4
    pop = PPO.population(
        size=population_size,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        **init_hp,
    )

    # Selection is unchanged from any other AgileRL run: ReGraMa is a stage of the
    # parameter mutation, so it composes with tournament or multi-frequency selection
    tournament = TournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=population_size,
    )

    # The four lines that differ from a stock run: ReGraMa's targeted dormant-neuron
    # resets, at the given score threshold, replace both the amplified and reset bands
    mutations = Mutations(
        no_mutation=0.4,
        architecture=0.2,
        new_layer_prob=0.2,
        parameters=0.2,
        activation=0.0,
        rl_hp=0.2,
        mutation_sd=0.1,
        rand_seed=42,
        device=device,
        regrama_param_mut=True,
        super_param_mut=False,
        reset_param_mut=False,
        dormant_threshold=0.01,
    )

    save_path = "ReGraMa_PPO_trained_agent.pt"

    # train_on_policy orchestrates the whole evolutionary run
    print("Training...")
    trained_pop, pop_fitnesses = train_on_policy(
        env=env,
        env_name="LunarLander-v3",
        algo="PPO",
        pop=pop,
        init_hp=init_hp,
        max_steps=200_000,
        evo_steps=10_240,
        eval_steps=None,
        eval_loop=1,
        selection_strategy=tournament,
        mutation=mutations,  # ReGraMa is configured on the mutation object
        wb=False,  # set True to log to Weights & Biases
        save_elite=True,  # save the best agent at the end of training
        elite_path=save_path,
    )

    env.close()

    # The two returned values line up agent-for-agent: the population that came out of the
    # final evolution step, and each agent's most recent evaluation score
    print("\n===== Final population =====")
    for agent, fitness in zip(trained_pop, pop_fitnesses, strict=True):
        print(f"  agent {agent.index:>3} | fitness {fitness:>8.1f}")
