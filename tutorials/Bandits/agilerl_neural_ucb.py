"""This tutorial shows how to train an NeuralUCB agent on the IRIS dataset.

Authors: Nick (https://github.com/nicku-a)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange
from ucimlrepo import fetch_ucirepo

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.utils.utils import initialPopulation
from agilerl.wrappers.learning import BanditEnv

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "h_size": [128],  # Actor hidden size
    }

    INIT_HP = {
        "POPULATION_SIZE": 1,  # Population size
        "BATCH_SIZE": 64,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 1.0,  # Scaling factor
        "LAMBDA": 1.0,  # Regularization factor
        "REG": 0.000625,  # Loss regularization factor
        "LEARN_STEP": 1,  # Learning frequency
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
    }

    # Fetch data  https://archive.ics.uci.edu/
    iris = fetch_ucirepo(id=53)
    features = iris.data.features
    targets = iris.data.targets

    env = BanditEnv(features, targets)  # Create environment
    context_dim = env.context_dim
    action_dim = env.arms

    pop = initialPopulation(
        algo="NeuralUCB",  # Algorithm
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
        action_dim=action_dim,  # Number of agent actions
        memory_size=10000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )

    max_episodes = 50  # Max training episodes
    max_steps = 50  # Max steps per episode

    evo_epochs = 2  # Evolution frequency
    evo_loop = 1  # Number of evaluation episodes

    print("Training...")

    regret = [[0] for _ in pop]

    total_steps = 0

    # TRAINING LOOP
    for idx_epi in trange(max_episodes):
        for i, agent in enumerate(pop):  # Loop through population
            score = 0
            losses = []
            context = env.reset()  # Reset environment at start of episode
            for idx_step in range(max_steps):
                # Get next action from agent
                action = agent.getAction(context)
                next_context, reward = env.step(action)  # Act in environment

                # Save experience to replay buffer
                memory.save2memory(context[action], reward)

                # Learn according to learning frequency
                if (
                    memory.counter % agent.learn_step == 0
                    and len(memory) >= agent.batch_size
                ):
                    for _ in range(2):
                        experiences = memory.sample(
                            agent.batch_size
                        )  # Sample replay buffer
                        # Learn according to agent's RL algorithm
                        loss = agent.learn(experiences)
                        losses.append(loss)

                context = next_context
                score += reward
                regret[i].append(regret[i][-1] + 1 - reward)

            total_steps += max_steps

        # Now evolve population if necessary
        if (idx_epi + 1) % evo_epochs == 0:
            # Evaluate population
            fitnesses = [
                agent.test(
                    env,
                    swap_channels=INIT_HP["CHANNELS_LAST"],
                    max_steps=max_steps,
                    loop=evo_loop,
                )
                for agent in pop
            ]

            print(f"Episode {idx_epi+1}/{max_episodes}")
            print(f"Regret: {[regret[i][-1] for i in range(len(pop))]}")

    # Plot the results
    for i, agent_regret in enumerate(regret):
        plt.plot(
            np.linspace(0, total_steps, len(agent_regret)),
            agent_regret,
            label=f"NeuralUCB: Agent {i}",
        )
        plt.xlabel("Training Step")
        plt.ylabel("Regret")
        plt.legend()
        plt.savefig("NeuralUCB-IRIS.png")
