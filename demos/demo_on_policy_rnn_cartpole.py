# AgileRL On-policy (RNN/MLP) Demo
#
# This script demonstrates how to use recurrent neural networks (RNNs) or MLPs with PPO to solve the Pendulum-v1
# environment with masked angular velocities.

from typing import List

import gymnasium as gym
import numpy as np
import torch

from agilerl.algorithms import PPO
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.rollouts.on_policy import collect_rollouts, collect_rollouts_recurrent
from agilerl.typing import BPTTSequenceType
from agilerl.utils.utils import create_population, default_progress_bar, make_vect_envs
from agilerl.wrappers.agent import RSNorm
from benchmarking.benchmarking_recurrent import MaskVelocityWrapper


def run_demo():
    # --- Setup Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Toggle this to True for RNN (LSTM), False for MLP
    recurrent = True  # <--- CHANGE THIS TO ENABLE/DISABLE RECURRENT
    active_collect = collect_rollouts if not recurrent else collect_rollouts_recurrent

    # --- Create Environment and Population ---
    num_envs = 16  # Can be higher for faster training
    learn_step = 1024 * num_envs

    if recurrent:
        NET_CONFIG = {
            "latent_dim": 64,
            "encoder_config": {
                "hidden_state_size": 64,  # LSTM hidden state size
                "num_layers": 1,
            },
            "head_config": {
                "hidden_size": [128],
            },
        }
    else:
        NET_CONFIG = {
            "latent_dim": 64,
            "encoder_config": {
                "hidden_size": [64],
            },
            "head_config": {
                "hidden_size": [128],
            },
        }

    # Hyperparameters
    INIT_HP = {
        "POP_SIZE": 2,  # Population size
        "BATCH_SIZE": 256,
        "LEARN_STEP": learn_step,
        "LR": 3e-4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_COEF": 0.2,
        "ENT_COEF": 0.001,
        "VF_COEF": 0.2,
        "MAX_GRAD_NORM": 0.5,
        "UPDATE_EPOCHS": 4,
        "SHARE_ENCODERS": False,  # Use same LSTM for actor and critic
        "USE_ROLLOUT_BUFFER": True,
        "RECURRENT": recurrent,
        "ACTION_STD_INIT": 0.6,
        "TARGET_KL": None,
        "BPTT_SEQUENCE_TYPE": BPTTSequenceType.CHUNKED,
        "MAX_SEQ_LEN": 16,  # max sequence length for truncated BPTT
    }

    def make_env():
        return MaskVelocityWrapper(gym.make("CartPole-v1"))

    env = make_vect_envs(
        make_env=make_env, num_envs=num_envs, should_async_vector=False
    )
    single_test_env = gym.vector.SyncVectorEnv([make_env])

    observation_space = env.single_observation_space
    action_space = env.single_action_space

    # Configuration for RL hyperparameter mutations
    hp_config = HyperparameterConfig(
        lr=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=128, max=1024),
        # In general we want the entropy to decay over time
        ent_coef=RLParameter(min=0.0001, max=0.001, grow_factor=1.0, shrink_factor=0.9),
    )

    pop: List[PPO] = create_population(
        algo="PPO",
        observation_space=observation_space,
        action_space=action_space,
        hp_config=hp_config,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    # Normalize observations using running statistics
    pop = [RSNorm(agent) for agent in pop]

    # --- Setup Evolution Components ---
    eval_loop = 10
    tournament = TournamentSelection(
        tournament_size=1,
        elitism=True,
        population_size=INIT_HP["POP_SIZE"],
        eval_loop=eval_loop,
    )

    mutations = Mutations(
        no_mutation=0.4,
        architecture=0,
        new_layer_prob=0.0,
        parameters=0.0,
        activation=0,
        rl_hp=0.2,
        mutation_sd=0.1,
        activation_selection=["ReLU", "ELU", "GELU"],
        mutate_elite=False,
        rand_seed=1,
        device=device,
    )

    # --- Training Loop (Performance-Flamegraph Style) ---
    max_steps = 5_000_000
    required_score = 500
    evo_steps = INIT_HP["LEARN_STEP"] * 5
    eval_steps = None

    total_steps = 0
    training_complete = False

    print("Training...")
    pbar = default_progress_bar(max_steps)
    while (
        np.less([agent.steps[-1] for agent in pop], max_steps).all()
        and not training_complete
    ):
        pop_episode_scores = []
        for agent in pop:
            steps = 0
            completed_episodes = []
            last_obs, last_done, last_scores, last_info = None, None, None, None
            for _ in range(-(evo_steps // -agent.learn_step)):
                # Collect rollouts and save in buffer
                episode_scores, last_obs, last_done, last_scores, last_info = (
                    active_collect(
                        agent,
                        env,
                        last_obs=last_obs,
                        last_done=last_done,
                        last_scores=last_scores,
                        last_info=last_info,
                    )
                )

                agent.learn()  # Learn from rollout buffer

                # Update step counter and scores
                total_steps += agent.learn_step
                steps += agent.learn_step
                agent.steps[-1] += agent.learn_step
                completed_episodes += episode_scores

            pop_episode_scores.append(
                round(np.mean(completed_episodes), 2)
                if len(completed_episodes) > 0
                else "0 completed episodes"
            )

            pbar.update(steps // len(pop))

        fitnesses = [
            agent.test(
                single_test_env,
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]

        pbar.write(
            f"--- Global steps {total_steps} ---\n"
            f"Steps: {[agent.steps[-1] for agent in pop]}\n"
            f"Scores: {pop_episode_scores}\n"
            f"Fitnesses: {['%.2f' % fitness for fitness in fitnesses]}\n"
            f"5 fitness avgs: {['%.2f' % np.mean(agent.fitness[-5:]) for agent in pop]}\n"
            f"Mutations: {[agent.mut for agent in pop]}\n"
        )

        if any(score >= required_score for score in pop_episode_scores):
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
