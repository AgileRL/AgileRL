import random

import numpy as np
import pandas as pd
import torch
import yaml

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_bandits import train_bandits
from agilerl.utils.utils import initialPopulation, printHyperparams

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


def main(INIT_HP, MUTATION_PARAMS, NET_CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("===== AgileRL Bandit Benchmarking =====")
    print(f"DEVICE: {device}")
    print(INIT_HP)
    print(MUTATION_PARAMS)
    print(NET_CONFIG)

    env = IRIS()  # Create environment
    context_dim = env.dim
    action_dim = env.arm

    if INIT_HP["CHANNELS_LAST"]:
        context_dim = (context_dim[2], context_dim[0], context_dim[1])

    field_names = ["context", "action"]
    memory = ReplayBuffer(
        action_dim, INIT_HP["MEMORY_SIZE"], field_names=field_names, device=device
    )
    tournament = TournamentSelection(
        INIT_HP["TOURN_SIZE"],
        INIT_HP["ELITISM"],
        INIT_HP["POP_SIZE"],
        INIT_HP["EVO_EPOCHS"],
    )
    mutations = Mutations(
        algo=INIT_HP["ALGO"],
        no_mutation=MUTATION_PARAMS["NO_MUT"],
        architecture=MUTATION_PARAMS["ARCH_MUT"],
        new_layer_prob=MUTATION_PARAMS["NEW_LAYER"],
        parameters=MUTATION_PARAMS["PARAMS_MUT"],
        activation=MUTATION_PARAMS["ACT_MUT"],
        rl_hp=MUTATION_PARAMS["RL_HP_MUT"],
        rl_hp_selection=MUTATION_PARAMS["RL_HP_SELECTION"],
        mutation_sd=MUTATION_PARAMS["MUT_SD"],
        arch=NET_CONFIG["arch"],
        rand_seed=MUTATION_PARAMS["RAND_SEED"],
        device=device,
    )

    agent_pop = initialPopulation(
        algo=INIT_HP["ALGO"],
        state_dim=context_dim,
        action_dim=action_dim,
        one_hot=None,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=INIT_HP["POP_SIZE"],
        device=device,
    )

    trained_pop, pop_fitnesses = train_bandits(
        env,
        INIT_HP["ENV_NAME"],
        INIT_HP["ALGO"],
        agent_pop,
        memory=memory,
        INIT_HP=INIT_HP,
        MUT_P=MUTATION_PARAMS,
        swap_channels=INIT_HP["CHANNELS_LAST"],
        n_episodes=INIT_HP["EPISODES"],
        evo_epochs=INIT_HP["EVO_EPOCHS"],
        evo_loop=1,
        target=INIT_HP["TARGET_SCORE"],
        tournament=tournament,
        mutation=mutations,
        wb=INIT_HP["WANDB"],
    )

    printHyperparams(trained_pop)

    if str(device) == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    with open("../configs/training/neural_ucb.yaml") as file:
        bandit_config = yaml.safe_load(file)
    INIT_HP = bandit_config["INIT_HP"]
    MUTATION_PARAMS = bandit_config["MUTATION_PARAMS"]
    NET_CONFIG = bandit_config["NET_CONFIG"]
    main(INIT_HP, MUTATION_PARAMS, NET_CONFIG)
