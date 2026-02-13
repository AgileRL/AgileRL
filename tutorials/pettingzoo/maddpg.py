"""This tutorial shows how to train an MADDPG agent on the space invaders atari environment.

Authors: Michael (https://github.com/mikepratt1), Nick (https://github.com/nicku-a)
"""

import os
from copy import deepcopy

import numpy as np
import supersuit as ss
import torch
from pettingzoo.atari import space_invaders_v2

from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.utils.utils import (
    create_population,
    default_progress_bar,
    make_multi_agent_vect_envs,
)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the network configuration
    NET_CONFIG = {
        "latent_dim": 128,
        "encoder_config": {
            "channel_size": [32, 32],  # CNN channel size
            "kernel_size": [3, 3],  # CNN kernel size
            "stride_size": [1, 1],  # CNN stride size
        },
        "head_config": {"hidden_size": [128]},  # Actor head hidden size
    }

    # Define the initial hyperparameters
    INIT_HP = {
        "POPULATION_SIZE": 1,
        "ALGO": "MADDPG",  # Algorithm
        "BATCH_SIZE": 128,  # Batch size
        "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.1,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.15,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "LR_ACTOR": 0.0001,  # Actor learning rate
        "LR_CRITIC": 0.001,  # Critic learning rate
        "GAMMA": 0.95,  # Discount factor
        "MEMORY_SIZE": 100_000,  # Max memory buffer size
        "LEARN_STEP": 50,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
    }

    num_envs = 8

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

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["AGENT_IDS"] = env.agents

    # Create a population ready for evolutionary hyper-parameter optimisation
    agent: MADDPG = create_population(
        INIT_HP["ALGO"],
        observation_spaces,
        action_spaces,
        NET_CONFIG,
        INIT_HP,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=num_envs,
        device=device,
    )[0]

    agent.set_training_mode(True)

    # Configure the multi-agent replay buffer
    field_names = ["obs", "action", "reward", "next_obs", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # Define training loop parameters
    agent_ids = deepcopy(env.agents)
    max_steps = 2_000_000  # Max steps (default: 2000000)
    learning_delay = 500  # Steps before starting learning
    training_steps = 10_000  # Frequency at which we evaluate training score
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes
    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = default_progress_bar(max_steps)
    while np.less(agent.steps[-1], max_steps):
        obs, info = env.reset()  # Reset environment at start of episode
        scores = np.zeros((num_envs, len(agent_ids)))
        completed_episode_scores = []
        steps = 0
        for idx_step in range(training_steps // num_envs):
            # Get next action from agent and take a step in the environment
            action, raw_action = agent.get_action(obs=obs, infos=info)
            next_obs, reward, termination, truncation, info = env.step(action)

            scores += np.array(list(reward.values())).transpose()
            total_steps += num_envs
            steps += num_envs
            pbar.update(num_envs)

            # Save experiences to replay buffer
            memory.save_to_memory(
                obs,
                raw_action,
                reward,
                next_obs,
                termination,
                is_vectorised=True,
            )

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
            elif len(memory) >= agent.batch_size and memory.counter > learning_delay:
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
            for idx, (d, t) in enumerate(zip(term_array, trunc_array, strict=False)):
                if np.any(d) or np.any(t):
                    completed_episode_scores.append(scores[idx])
                    agent.scores.append(scores[idx])
                    scores[idx] = 0
                    reset_noise_indices.append(idx)

            agent.reset_action_noise(reset_noise_indices)

        agent.steps[-1] += steps

        # Evaluate population
        fitness = agent.test(
            env,
            max_steps=eval_steps,
            loop=eval_loop,
            sum_scores=False,
        )
        pop_episode_scores = np.array(completed_episode_scores)
        mean_scores = np.mean(pop_episode_scores, axis=0)

        pbar.write(
            f"--- Global steps {total_steps} ---\n"
            f"Steps {agent.steps[-1]}\n"
            f"Scores: {np.mean(mean_scores)}\n"
            f"Fitnesses: {np.mean(fitness)}\n"
            f"5 fitness avgs: {np.mean(agent.fitness[-5:])}",
        )

        # Update step counter
        agent.steps.append(agent.steps[-1])

    # Save the trained algorithm
    path = "./models/MADDPG"
    filename = "MADDPG_trained_agent.pt"
    os.makedirs(path, exist_ok=True)
    save_path = os.path.join(path, filename)
    agent.save_checkpoint(save_path)

    pbar.close()
    env.close()
