"""This tutorial shows how to train an NeuralTS agent on the PenDigits dataset with evolutionary HPO.

Authors: Nick (https://github.com/nicku-a)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from ucimlrepo import fetch_ucirepo

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from agilerl.wrappers.learning import BanditEnv

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "hidden_size": [128],  # Actor hidden size
    }

    INIT_HP = {
        "POPULATION_SIZE": 4,  # Population size
        "BATCH_SIZE": 64,  # Batch size
        "LR": 0.001,  # Learning rate
        "GAMMA": 1.0,  # Scaling factor
        "LAMBDA": 1.0,  # Regularization factor
        "REG": 0.0625,  # Loss regularization factor
        "LEARN_STEP": 2,  # Learning frequency
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
    }

    # Fetch data  https://archive.ics.uci.edu/
    pendigits = fetch_ucirepo(id=81)
    features = pendigits.data.features
    targets = pendigits.data.targets

    env = BanditEnv(features, targets)  # Create environment
    context_dim = env.context_dim
    action_dim = env.arms

    pop = create_population(
        algo="NeuralTS",  # Algorithm
        state_dim=context_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=None,  # One-hot encoding
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        device=device,
    )

    field_names = ["context", "reward"]
    memory = ReplayBuffer(
        memory_size=10000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    mutations = Mutations(
        algo="NeuralTS",  # Algorithm
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0.2,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        rl_hp_selection=["lr", "batch_size"],  # Learning HPs to choose from
        mutation_sd=0.1,  # Mutation strength
        mutate_elite=False,  # Mutate best agent in population
        arch=NET_CONFIG["arch"],  # Network architecture
        rand_seed=1,  # Random seed
        device=device,
    )

    max_steps = 2500  # Max steps per episode
    episode_steps = 500  # Steps in episode
    evo_steps = 1000  # Evolution frequency
    eval_steps = 500  # Evaluation steps per episode
    eval_loop = 1  # Number of evaluation episodes
    print("Training...")

    regret = [[0] for _ in pop]
    score = [[0] for _ in pop]
    total_steps = 0
    evo_count = 0

    # TRAINING LOOP
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        for i, agent in enumerate(pop):  # Loop through population
            losses = []
            context = env.reset()  # Reset environment at start of episode
            for idx_step in range(episode_steps):
                # Get next action from agent
                action = agent.get_action(context)
                next_context, reward = env.step(action)  # Act in environment

                # Save experience to replay buffer
                memory.save_to_memory(context[action], reward)

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

        fitnesses = [
            agent.test(
                env,
                swap_channels=INIT_HP["CHANNELS_LAST"],
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        print(f"Regret: {[regret[i][-1] for i in range(len(pop))]}")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')

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
            label=f"NeuralTS: Agent {i}",
        )
    plt.xlabel("Training Step")
    plt.ylabel("Regret")
    plt.legend()
    plt.savefig("NeuralTS-PenDigits-regret.png")

    plt.figure()
    for i, agent_score in enumerate(score):
        smoothed_score = gaussian_filter1d(agent_score, sigma=80)
        plt.plot(
            np.linspace(0, total_steps, len(smoothed_score)),
            smoothed_score,
            label=f"NeuralTS: Agent {i}",
        )
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("NeuralTS-PenDigits-reward.png")
