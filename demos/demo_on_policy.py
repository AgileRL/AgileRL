from typing import List

import numpy as np
import torch

from agilerl.algorithms import PPO
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.rollouts.on_policy import collect_rollouts
from agilerl.utils.utils import create_population, default_progress_bar, make_vect_envs

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')

if __name__ == "__main__":
    print("===== AgileRL On-policy Demo =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "encoder_config": {
            "hidden_size": [32, 32],  # Actor hidden size
        }
    }

    INIT_HP = {
        "POP_SIZE": 6,  # Population size
        "BATCH_SIZE": 128,  # Batch size
        "LR": 1e-3,  # Learning rate
        "LEARN_STEP": 128,  # Learning frequency
        "GAMMA": 0.99,  # Discount factor
        "GAE_LAMBDA": 0.95,  # Lambda for general advantage estimation
        "ACTION_STD_INIT": 0.6,  # Initial action standard deviation
        "CLIP_COEF": 0.2,  # Surrogate clipping coefficient
        "ENT_COEF": 0.01,  # Entropy coefficient
        "VF_COEF": 0.5,  # Value function coefficient
        "MAX_GRAD_NORM": 0.5,  # Maximum norm for gradient clipping
        "TARGET_KL": None,  # Target KL divergence threshold
        "UPDATE_EPOCHS": 4,  # Number of policy update epochs
        "USE_ROLLOUT_BUFFER": True,  # Use rollout buffer
    }

    num_envs = 16
    env = make_vect_envs("LunarLander-v3", num_envs=num_envs)  # Create environment

    observation_space = env.single_observation_space
    action_space = env.single_action_space
    pop: List[PPO] = create_population(
        algo="PPO",  # Algorithm
        observation_space=observation_space,  # Observation space
        action_space=action_space,  # Action space
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
        num_envs=num_envs,  # Number of vectorized envs
        device=device,
    )

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        mutation_sd=0.1,  # Mutation strength  # Network architecture
        rand_seed=1,  # Random seed
        device=device,
    )

    max_steps = 200000  # Max steps
    evo_steps = 10000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes

    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = default_progress_bar(max_steps)
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            last_obs, last_done, last_scores, last_info = None, None, None, None
            steps = 0
            completed_episodes = []
            for _ in range(-(evo_steps // -agent.learn_step)):
                # Collect rollouts and save in buffer
                episode_scores, last_obs, last_done, last_scores, last_info = (
                    collect_rollouts(
                        agent,
                        env,
                        last_obs=last_obs,
                        last_done=last_done,
                        last_scores=last_scores,
                        last_info=last_info,
                    )
                )

                loss = agent.learn()  # Learn from rollout buffer

                # Update step counter and scores
                total_steps += agent.learn_step
                steps += agent.learn_step
                agent.steps[-1] += agent.learn_step
                completed_episodes += episode_scores

            pop_episode_scores.append(
                np.mean(completed_episodes)
                if len(completed_episodes) > 0
                else "0 completed episodes"
            )
            pbar.update(steps // len(pop))

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
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
        )

        # Tournament selection and population mutation
        # elite, pop = tournament.select(pop)
        # pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    pbar.close()
    env.close()
