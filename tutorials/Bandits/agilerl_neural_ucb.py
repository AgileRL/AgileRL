"""This tutorial shows how to train an NeuralUCB agent on the IRIS dataset.

Authors: Nick (https://github.com/nicku-a)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from gymnasium import spaces
from scipy.ndimage import gaussian_filter1d
from tensordict import TensorDict
from ucimlrepo import fetch_ucirepo

from agilerl.algorithms import NeuralUCB
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population, default_progress_bar
from agilerl.wrappers.learning import BanditEnv

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "latent_dim": 64,
        "encoder_config": {"hidden_size": [64]},
        "head_config": {"hidden_size": [64]},
    }

    INIT_HP = {
        "POPULATION_SIZE": 4,  # Population size
        "BATCH_SIZE": 64,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 1.0,  # Scaling factor
        "LAMBDA": 1.0,  # Regularization factor
        "REG": 0.000625,  # Loss regularization factor
        "LEARN_STEP": 2,  # Learning frequency
    }

    # Fetch data  https://archive.ics.uci.edu/
    iris = fetch_ucirepo(id=53)
    features = iris.data.features
    targets = iris.data.targets

    env = BanditEnv(features, targets)  # Create environment
    context_dim = env.context_dim
    action_dim = env.arms

    # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr=RLParameter(min=6.25e-5, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        learn_step=RLParameter(
            min=1, max=10, dtype=int, grow_factor=1.5, shrink_factor=0.75
        ),
    )

    observation_space = spaces.Box(
        low=features.values.min(), high=features.values.max(), shape=context_dim
    )
    action_space = spaces.Discrete(action_dim)
    pop: list[NeuralUCB] = create_population(
        algo="NeuralUCB",  # Algorithm
        observation_space=observation_space,  # Observation space
        action_space=action_space,  # Action space
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        hp_config=hp_config,  # Hyperparameter configuration
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        device=device,
    )

    memory = ReplayBuffer(
        max_size=10000,  # Max replay buffer size
        device=device,
    )

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0.2,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        mutation_sd=0.1,  # Mutation strength
        mutate_elite=False,  # Mutate best agent in population  # Network architecture
        rand_seed=1,  # Random seed
        device=device,
    )

    max_steps = 2500  # Max steps per episode
    episode_steps = 500  # Steps in episode
    evo_steps = 500  # Evolution frequency
    eval_steps = 500  # Evaluation steps per episode
    eval_loop = 1  # Number of evaluation episodes
    evo_count = 0

    regret = [[0] for _ in pop]
    score = [[0] for _ in pop]
    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = default_progress_bar(max_steps)
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        for i, agent in enumerate(pop):  # Loop through population
            losses = []
            context = env.reset()  # Reset environment at start of episode
            for idx_step in range(episode_steps):
                # Get next action from agent
                action = agent.get_action(context)
                next_context, reward = env.step(action)  # Act in environment

                # Save experience to replay buffer
                transition = TensorDict(
                    {
                        "obs": context[action],
                        "reward": reward,
                    },
                ).float()
                transition = transition.unsqueeze(0)
                transition.batch_size = [1]

                # Save experience to replay buffer
                memory.add(transition)

                # Learn according to learning frequency
                if len(memory) >= agent.batch_size:
                    for _ in range(agent.learn_step):
                        # Sample replay buffer
                        # Learn according to agent's RL algorithm
                        experiences = memory.sample(agent.batch_size)
                        loss = agent.learn(experiences)
                        losses.append(loss)

                context = next_context
                score[i].append(reward)
                regret[i].append(regret[i][-1] + 1 - reward)

            total_steps += episode_steps
            agent.steps[-1] += episode_steps

        pbar.update(episode_steps)

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
            f"Steps {[agent.steps[-1] for agent in pop]}\n"
            f"Regret: {[regret[i][-1] for i in range(len(pop))]}\n"
            f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}\n'
            f"Mutations: {[agent.mut for agent in pop]}"
        )

        if pop[0].steps[-1] // evo_steps > evo_count:
            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)
            evo_count += 1

    # Plot the results
    plt.figure()
    for i, agent_regret in enumerate(regret):
        plt.plot(
            np.linspace(0, total_steps, len(agent_regret)),
            agent_regret,
            label=f"NeuralUCB: Agent {i}",
        )
    plt.xlabel("Training Step")
    plt.ylabel("Regret")
    plt.legend()
    plt.savefig("NeuralUCB-IRIS-regret.png")
    plt.close()

    plt.figure()
    for i, agent_score in enumerate(score):
        smoothed_score = gaussian_filter1d(agent_score, sigma=80)
        plt.plot(
            np.linspace(0, total_steps, len(smoothed_score)),
            smoothed_score,
            label=f"NeuralUCB: Agent {i}",
        )
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("NeuralUCB-IRIS-reward.png")
    plt.close()
