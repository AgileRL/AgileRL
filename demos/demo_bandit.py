import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import wandb
from tqdm import trange

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.utils.utils import initialPopulation

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


class IRIS:
    def __init__(self):
        self.arm = 3
        self.dim = (12,)
        self.data = pd.read_csv("../data/iris/iris.csv")
        self.prev_reward = np.zeros(self.arm)

    def _new_state_and_target_action(self):
        r = random.randint(0, 149)
        if 0 <= r <= 49:
            target = 0
        elif 50 <= r <= 99:
            target = 1
        else:
            target = 2
        rand = self.data.loc[r]
        x = np.zeros(4)
        for i in range(1, 5):
            x[i - 1] = rand[i]
        X_n = []
        for i in range(3):
            front = np.zeros(4 * i)
            back = np.zeros(4 * (2 - i))
            new_d = np.concatenate((front, x, back), axis=0)
            X_n.append(new_d)
        X_n = np.array(X_n)
        return X_n, target

    def step(self, k):
        # Calculate reward from action in previous state
        reward = self.prev_reward[k]

        # Now decide on next state
        next_state, target = self._new_state_and_target_action()

        # Save reward for next call to step()
        next_reward = np.zeros(self.arm)
        next_reward[target] = 1
        self.prev_reward = next_reward
        return next_state, reward

    def reset(self):
        next_state, target = self._new_state_and_target_action()
        next_reward = np.zeros(self.arm)
        next_reward[target] = 1
        self.prev_reward = next_reward
        return next_state


if __name__ == "__main__":
    print("===== AgileRL Bandit Demo =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "h_size": [128],  # Actor hidden size
    }

    INIT_HP = {
        "POPULATION_SIZE": 1,  # Population size
        "BATCH_SIZE": 64,  # Batch size
        "LR": 1e-4,  # Learning rate
        "GAMMA": 1.0,  # Scaling factor
        "LAMBDA": 1.0,  # Regularization factor
        "REG": 1.0,  # Loss regularization factor
        "LEARN_STEP": 1,  # Learning frequency
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
    }

    env = IRIS()  # Create environment
    context_dim = env.dim
    action_dim = env.arm

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

    # pop = [NeuralUCB(context_dim, action_dim, batch_size=INIT_HP["BATCH_SIZE"], lr=INIT_HP["LR"], device=device)]

    field_names = ["context", "reward"]
    memory = ReplayBuffer(
        action_dim=action_dim,  # Number of agent actions
        memory_size=10000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )

    # tournament = TournamentSelection(
    #     tournament_size=2,  # Tournament selection size
    #     elitism=True,  # Elitism in tournament selection
    #     population_size=INIT_HP["POPULATION_SIZE"],  # Population size
    #     evo_step=1,
    # )  # Evaluate using last N fitness scores

    # mutations = Mutations(
    #     algo="DQN",  # Algorithm
    #     no_mutation=0.4,  # No mutation
    #     architecture=0.2,  # Architecture mutation
    #     new_layer_prob=0.2,  # New layer mutation
    #     parameters=0.2,  # Network parameters mutation
    #     activation=0,  # Activation layer mutation
    #     rl_hp=0.2,  # Learning HP mutation
    #     rl_hp_selection=["lr", "batch_size"],  # Learning HPs to choose from
    #     mutation_sd=0.1,  # Mutation strength
    #     arch=NET_CONFIG["arch"],  # Network architecture
    #     rand_seed=1,  # Random seed
    #     device=device,
    # )

    tournament = mutation = None

    max_episodes = 100  # Max training episodes
    max_steps = 100  # Max steps per episode

    evo_epochs = 5  # Evolution frequency
    evo_loop = 1  # Number of evaluation episodes

    print("Training...")

    regret = [[0] for _ in pop]

    wandb.init(
        # set the wandb project where this run will be logged
        project="AgileRL-Bandits",
        name="NeuralUCB-{}".format(datetime.now().strftime("%m%d%Y%H%M%S")),
        # track hyperparameters and run metadata
        config=INIT_HP,
    )

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

            wandb_dict = {
                "global_step": total_steps,
                "train/loss": np.mean(losses),
                "train/score": score,
                "train/regret": regret[0][-1],
            }
            wandb.log(wandb_dict)

        # # Now evolve population if necessary
        # if (idx_epi + 1) % evo_epochs == 0:
        #     # Evaluate population
        #     fitnesses = [
        #         agent.test(
        #             env,
        #             swap_channels=INIT_HP["CHANNELS_LAST"],
        #             max_steps=max_steps,
        #             loop=evo_loop,
        #         )
        #         for agent in pop
        #     ]

        #     print(f"Episode {idx_epi+1}/{max_episodes}")
        #     print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        #     print(
        #         f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}'
        #     )

        #     # # Tournament selection and population mutation
        #     # elite, pop = tournament.select(pop)
        #     # pop = mutations.mutation(pop)
