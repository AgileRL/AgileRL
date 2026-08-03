# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""This tutorial shows how to train an MATD3 agent on the simple speaker listener multi-particle environment.

Authors: Michael (https://github.com/mikepratt1), Nickua (https://github.com/nicku-a)
"""

import os

import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from tensordict import TensorDictBase

from agilerl.algorithms import MATD3
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.data import MultiAgentTransition
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.population import Population
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    make_multi_agent_vect_envs,
    run_selection_and_mutation,
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Online Multi-Agent Demo =====")

    num_envs = 8

    # Network configuration
    net_config = {
        "latent_dim": 64,
        "encoder_config": {
            "hidden_size": [64],
        },
        "head_config": {
            "hidden_size": [64],
        },
    }

    # Algorithm hyperparameters
    init_hp = {
        "O_U_noise": True,
        "expl_noise": 0.1,
        "mean_noise": 0.0,
        "theta": 0.15,
        "dt": 0.01,
        "batch_size": 128,
        "lr_actor": 0.0001,
        "lr_critic": 0.001,
        "gamma": 0.95,
        "learn_step": 100,
        "tau": 0.01,
        "policy_freq": 2,
    }

    def make_env():
        return simple_speaker_listener_v4.parallel_env(continuous_actions=True)

    env = make_multi_agent_vect_envs(env=make_env, num_envs=num_envs)

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]

    # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=512),
        learn_step=RLParameter(
            min=20,
            max=200,
            grow_factor=1.5,
            shrink_factor=0.75,
        ),
    )

    # Create a population ready for evolutionary hyper-parameter optimisation
    population_size = 4
    pop = MATD3.population(
        size=population_size,
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=env.agents,
        net_config=net_config,
        hp_config=hp_config,
        device=device,
        **init_hp,
    )

    # Configure the multi-agent replay buffer
    memory = ReplayBuffer(
        max_size=100_000,
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=population_size,  # Population size
    )

    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        no_mutation=0.2,  # Probability of no mutation
        architecture=0.2,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        mutation_sd=0.1,  # Mutation strength
        rand_seed=1,
        device=device,
    )

    # Define training loop parameters
    max_steps = 2_000_000  # Max steps (default: 2000000)
    learning_delay = 0  # Steps before starting learning
    evo_steps = 10_000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes

    pbar = default_progress_bar(max_steps)

    # Initialize loggers and population wrapper
    loggers = init_loggers(
        algo="MATD3",
        env_name="simple_speaker_listener_v4",
        pbar=pbar,
        verbose=True,
    )

    population = Population(
        agents=pop,
        loggers=loggers,
    )

    # Pre-training mutation
    population.update(mutations.mutation(population.agents, pre_training_mut=True))

    # TRAINING LOOP
    while population.all_below(max_steps):
        for agent in population.agents:  # Loop through population
            agent.set_training_mode(True)
            agent.init_training_step()

            obs, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0

            for idx_step in range(evo_steps // num_envs):
                # Get next action from agent
                action, raw_action = agent.get_action(
                    obs=obs,
                    infos=info,
                )

                # Act in environment
                next_obs, reward, termination, truncation, info = env.step(action)

                scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
                steps += num_envs

                # Save experiences to replay buffer
                transition: TensorDictBase = MultiAgentTransition(
                    obs=obs,
                    action=raw_action,
                    reward=reward,
                    next_obs=next_obs,
                    done=termination,
                )
                transition = transition.to_tensordict()
                transition.batch_size = [num_envs]
                memory.add(transition)

                # Learn according to learning frequency
                # Handle learn steps > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)

                # Handle num_envs > learn step; learn multiple times per step in env
                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)

                obs = next_obs

                # Calculate scores and reset noise for finished episodes
                reset_noise_indices = []
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(
                    zip(term_array, trunc_array, strict=False),
                ):
                    if np.any(d) or np.any(t):
                        completed_episode_scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)

                agent.reset_action_noise(reset_noise_indices)

            agent.add_scores(completed_episode_scores)
            agent.finalize_training_step(steps)
            pbar.update(evo_steps // population.size)

        population.increment_evo_step()

        # Evaluate population
        for agent in population.agents:
            agent.test(
                env,
                max_steps=eval_steps,
                loop=eval_loop,
            )

        population.report_metrics(clear=True)

        # Tournament selection and population mutation
        population.update(
            run_selection_and_mutation(
                tournament,
                population=population.agents,
                mutation=mutations,
                env_name="simple_speaker_listener_v4",
                algo="MATD3",
                save_elite=True,
                elite_path="./models/MATD3",
            ),
        )

    # Save the trained algorithm
    path = "./models/MATD3"
    filename = "MATD3_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    population.agents[0].save_checkpoint(save_path)

    population.finish()
    pbar.close()
    env.close()
