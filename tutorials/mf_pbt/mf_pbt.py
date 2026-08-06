# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Multi-Frequency PBT (MF-PBT) with PPO on LunarLander-v3.

Runnable companion to the documentation tutorial
(``docs/tutorials/mf_pbt/mf_pbt_tutorial.rst``). It builds the environment,
population, selection strategy and mutations in Python and hands them to
``train_on_policy``, which runs the evolutionary HPO loop with multi-frequency
selection instead of the default tournament selection. Multi-frequency selection is a
drop-in replacement for tournament selection, so only the selection object changes; the
operator is algorithm-agnostic and the same wiring works for DQN, DDPG, IPPO, bandits and
offline algorithms.

The manifest alternative is two lines (see ``configs/training/ppo/ppo_mfpbt.yaml``)::

    from agilerl import LocalTrainer

    population, fitnesses = LocalTrainer.from_manifest(
        "configs/training/ppo/ppo_mfpbt.yaml"
    ).train()
"""

import torch

from agilerl.algorithms import PPO
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.multi_frequency import MultiFrequencySelection
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_on_policy import train_on_policy
from agilerl.utils.utils import make_vect_envs

if __name__ == "__main__":
    print("===== AgileRL MF-PBT (PPO) Demo =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a vectorised environment and read the observation/action spaces off it
    num_envs = 16
    env = make_vect_envs("LunarLander-v3", num_envs=num_envs)
    observation_space = env.single_observation_space
    action_space = env.single_action_space

    # A simple MLP with two 64-node hidden layers
    net_config = {"head_config": {"hidden_size": [64, 64]}}

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

    # MF-PBT needs a reasonably large population; 16 agents give two subpopulations of 8
    population_size = 16
    pop = PPO.population(
        size=population_size,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
        **init_hp,
    )

    # The only line that differs from a tournament-selection run: build a
    # MultiFrequencySelection instead of a TournamentSelection. Its arguments map
    # one-to-one onto the manifest's `selection_strategy` block. Subpopulations are
    # tagged automatically (by population index) on the first call to `select`, so we do
    # not tag the agents ourselves. `seed` seeds the RNG that picks which winner replaces
    # each loser; the manifest path derives it from `mutation.rand_seed` instead
    mf_selection = MultiFrequencySelection(
        population_size=population_size,
        n_subpopulations=2,
        n_winners=2,
        n_survivors=0,
        n_open_for_migration=2,
        n_losers=4,
        evolution_frequency_ratios=[1, 5],
        seed=42,
    )

    # Mutation is unchanged from any other AgileRL run
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
    )

    save_path = "MFPBT_PPO_trained_agent.pt"

    # train_on_policy orchestrates the whole evolutionary run: it collects rollouts, trains
    # each agent, evaluates the population and runs the selection-plus-mutation step every
    # evo_steps. Passing selection_strategy=mf_selection is the single switch that makes it
    # MF-PBT; swapping in a TournamentSelection runs tournament HPO with the same call. Each
    # evolution cycle it calls mf_selection.select(...), which returns the winner-clone
    # indices, and mutates only those; the per-frequency scheduling is internal to select
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
        selection_strategy=mf_selection,  # MF-PBT; the one line that differs from tournament HPO
        mutation=mutations,
        wb=False,  # set True to log to Weights & Biases
        save_elite=True,  # save the best agent at the end of training
        elite_path=save_path,
    )

    env.close()
