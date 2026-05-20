"""This tutorial shows how to train an MADDPG agent on the space invaders atari environment.

Authors: Michael (https://github.com/mikepratt1), Nick (https://github.com/nicku-a)
"""

import os

import numpy as np
import supersuit as ss
import torch
from pettingzoo.atari import space_invaders_v2
from tensordict import TensorDictBase

from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.data import MultiAgentTransition
from agilerl.components.replay_buffer import MultiAgentReplayBuffer
from agilerl.population import Population
from agilerl.utils.utils import (
    default_progress_bar,
    init_loggers,
    make_multi_agent_vect_envs,
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_envs = 8

    # Network configuration
    net_config = {
        "latent_dim": 128,
        "encoder_config": {
            "channel_size": [32, 32],
            "kernel_size": [3, 3],
            "stride_size": [1, 1],
        },
        "head_config": {"hidden_size": [128]},
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
        "learn_step": 50,
        "tau": 0.01,
    }

    # Define the space invaders environment as a parallel environment
    def make_env():
        env = space_invaders_v2.parallel_env()
        env = ss.frame_skip_v0(env, 4)
        env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)
        return ss.reshape_v0(env, (4, 84, 84))

    # Environment processing for image based observations
    env = make_multi_agent_vect_envs(env=make_env, num_envs=num_envs)

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]

    # Create a population ready for evolutionary hyper-parameter optimisation
    population_size = 1
    pop = MADDPG.population(
        size=population_size,
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=env.agents,
        net_config=net_config,
        device=device,
        **init_hp,
    )

    # Configure the multi-agent replay buffer
    memory = MultiAgentReplayBuffer(
        100_000,
        device=device,
    )

    # Define training loop parameters
    max_steps = 2_000_000  # Max steps (default: 2000000)
    learning_delay = 500  # Steps before starting learning
    evo_steps = 10_000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes

    pbar = default_progress_bar(max_steps)

    # Initialize loggers and population wrapper
    loggers = init_loggers(
        algo="MADDPG",
        env_name="space_invaders_v2",
        pbar=pbar,
        verbose=True,
    )

    population = Population(
        agents=pop,
        loggers=loggers,
    )

    # TRAINING LOOP
    while population.all_below(max_steps):
        for agent in population.agents:
            agent.set_training_mode(True)
            agent.init_training_step()

            obs, info = env.reset()  # Reset environment at start of episode
            completed_episode_scores = []
            scores = np.zeros((num_envs, len(env.agents)))
            steps = 0

            for idx_step in range(evo_steps // num_envs):
                # Get next action from agent and take a step in the environment
                action, raw_action = agent.get_action(obs=obs, infos=info)
                next_obs, reward, termination, truncation, info = env.step(action)

                scores += np.array(list(reward.values())).transpose()
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
                sum_scores=False,
            )

        population.report_metrics(clear=True)

    # Save the trained algorithm
    path = "./models/MADDPG"
    filename = "MADDPG_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    population.agents[0].save_checkpoint(save_path)

    population.finish()
    pbar.close()
    env.close()
