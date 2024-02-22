from datetime import datetime

import numpy as np
import torch
from tqdm import trange
from ucimlrepo import fetch_ucirepo

import wandb
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation
from agilerl.wrappers.learning import BanditEnv

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "h_size": [128],  # Actor hidden size
    }

    INIT_HP = {
        "POPULATION_SIZE": 4,  # Population size
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

    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        evo_step=1,
    )  # Evaluate using last N fitness scores

    mutations = Mutations(
        algo="NeuralUCB",  # Algorithm
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.5,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0.2,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        rl_hp_selection=["lr", "batch_size"],  # Learning HPs to choose from
        mutation_sd=0.1,  # Mutation strength
        arch=NET_CONFIG["arch"],  # Network architecture
        rand_seed=1,  # Random seed
        device=device,
    )

    max_episodes = 50  # Max training episodes
    max_steps = 100  # Max steps per episode

    evo_epochs = 2  # Evolution frequency
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

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)
