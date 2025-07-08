# AgileRL On-policy (RNN/MLP) Demo
#
# This script demonstrates how to use recurrent neural networks (RNNs) or MLPs with PPO to solve the Pendulum-v1
# environment with masked angular velocities.

from typing import List

import gymnasium as gym
import numpy as np
import torch
from tqdm import trange

from agilerl.algorithms import PPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.rollouts.on_policy import collect_rollouts_recurrent
from agilerl.utils.utils import create_population, make_vect_envs
from benchmarking.benchmarking_recurrent import MaskVelocityWrapper


def run_demo():
    # --- Setup Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Toggle this to True for RNN (LSTM), False for MLP
    recurrent = True  # <--- CHANGE THIS TO ENABLE/DISABLE RECURRENT

    # --- Create Environment and Population ---
    num_envs = 4  # Can be higher for faster training

    if recurrent:
        NET_CONFIG = {
            "encoder_config": {
                "hidden_state_size": 64,  # LSTM hidden state size
                "num_layers": 1,
                "max_seq_len": 1024,
            },
        }
    else:
        NET_CONFIG = {
            "encoder_config": {
                "hidden_size": [64],
            },
        }

    # Hyperparameters
    INIT_HP = {
        "POP_SIZE": 1,  # Population size
        "BATCH_SIZE": 256,
        "LEARN_STEP": 1024,
        "LR": 1e-3,
        "GAMMA": 0.9,
        "GAE_LAMBDA": 0.95,
        "CLIP_COEF": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 1.0,
        "MAX_GRAD_NORM": 0.5,
        "UPDATE_EPOCHS": 4,
        "SHARE_ENCODERS": True,
        "USE_ROLLOUT_BUFFER": True,
        "RECURRENT": recurrent,
        "ACTION_STD_INIT": 0.6,
        "TARGET_KL": None,
    }

    def make_env():
        return MaskVelocityWrapper(gym.make("Pendulum-v1"))

    env = make_vect_envs(
        make_env=make_env, num_envs=num_envs, should_async_vector=False
    )
    single_test_env = gym.vector.SyncVectorEnv([make_env])

    observation_space = env.single_observation_space
    action_space = env.single_action_space

    pop: List[PPO] = create_population(
        algo="PPO",
        observation_space=observation_space,
        action_space=action_space,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    # --- Setup Evolution Components ---
    eval_loop = 3
    tournament = TournamentSelection(
        tournament_size=2,
        elitism=True,
        population_size=INIT_HP["POP_SIZE"],
        eval_loop=eval_loop,
    )

    mutations = Mutations(
        no_mutation=0.4,
        architecture=0,
        new_layer_prob=0.0,
        parameters=0.2,
        activation=0,
        rl_hp=0.2,
        mutation_sd=0.1,
        activation_selection=["ReLU", "ELU", "GELU"],
        mutate_elite=True,
        rand_seed=1,
        device=device,
    )

    # --- Training Loop (Performance-Flamegraph Style) ---
    max_steps = 1_000_000 // len(pop)
    required_score = 0.95
    evo_steps = num_envs * INIT_HP["LEARN_STEP"] * 1
    eval_steps = None

    total_steps = 0
    training_complete = False

    print("Training...")
    pbar = trange(max_steps * len(pop), unit="step")
    while (
        np.less([agent.steps[-1] for agent in pop], max_steps).all()
        and not training_complete
    ):
        for agent in pop:
            collect_rollouts_recurrent(agent, env)
            agent.learn()
            total_steps += agent.learn_step * num_envs
            agent.steps[-1] += agent.learn_step * num_envs
            pbar.update(agent.learn_step * num_envs // len(pop))

        # Evaluate and evolve
        if total_steps % evo_steps == 0:
            fitnesses = [
                agent.test(
                    single_test_env,
                    max_steps=eval_steps,
                    loop=eval_loop,
                )
                for agent in pop
            ]
            mean_scores = [
                round(np.mean(agent.fitness[-eval_loop:]), 1) for agent in pop
            ]
            print(f"--- Global steps {total_steps} ---")
            print(f"Steps {[agent.steps[-1] for agent in pop]}")
            print(f"Scores: {mean_scores}")
            print(f"Fitnesses: {['%.2f' % fitness for fitness in fitnesses]}")
            print(
                f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop]}"
            )

            if any(score >= required_score for score in mean_scores):
                print(
                    f"\nAgent achieved required score {required_score}. Stopping training."
                )
                elite, _ = tournament.select(pop)
                training_complete = True
                break

            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)
            for agent in pop:
                agent.steps.append(agent.steps[-1])

    pbar.close()
    env.close()

    # --- Evaluate Best Agent ---
    print("Evaluating best agent...")

    if not training_complete:
        fitnesses = [
            agent.test(
                single_test_env,
                max_steps=eval_steps,
                loop=eval_loop,
                vectorized=True,
            )
            for agent in pop
        ]
        elite, _ = tournament.select(pop)

    # --- Run a few episodes and print results ---
    print("Running a few episodes with the best agent:")
    total_steps = 0
    episode_rewards = []

    for episode in range(20):
        obs, _ = single_test_env.reset()
        done = np.array([False])
        episode_reward = 0
        episode_steps = 0
        if recurrent:
            hidden_state = elite.get_initial_hidden_state(1)

        while not done[0]:
            if recurrent:
                action, _, _, _, hidden_state = elite.get_action(
                    obs, hidden_state=hidden_state
                )
            else:
                action, _, _, _, _ = elite.get_action(obs)
            obs, reward, terminated, truncated, _ = single_test_env.step(action)
            done = np.logical_or(terminated, truncated)
            episode_reward += reward[0]
            episode_steps += 1
        print(
            f"Episode {episode + 1}: Reward: {episode_reward}, Steps: {episode_steps}"
        )
        total_steps += episode_steps
        episode_rewards.append(episode_reward)

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    avg_steps = total_steps / len(episode_rewards)
    print(f"Average Reward: {avg_reward:.2f}, Average Steps: {avg_steps:.2f}")

    print("Demo complete.")


if __name__ == "__main__":
    run_demo()
