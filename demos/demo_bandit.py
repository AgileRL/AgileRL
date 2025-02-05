from datetime import datetime

import numpy as np
import torch
import wandb
from gymnasium import spaces
from tqdm import trange
from ucimlrepo import fetch_ucirepo

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import create_population
from agilerl.wrappers.learning import BanditEnv

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


if __name__ == "__main__":
    print("===== AgileRL Bandit Demo =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "encoder_config": {
            "encoder_config": {"hidden_size": [128]}  # Actor hidden size
        }
    }

    INIT_HP = {
        "BATCH_SIZE": 64,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 1.0,  # Scaling factor
        "LAMBDA": 1.0,  # Regularization factor
        "REG": 0.000625,  # Loss regularization factor
        "LEARN_STEP": 2,  # Learning frequency
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
        "POP_SIZE": 4,  # Population size
    }

    # Fetch data  https://archive.ics.uci.edu/
    iris = fetch_ucirepo(id=53)
    features = iris.data.features
    targets = iris.data.targets

    env = BanditEnv(features, targets)  # Create environment
    context_dim = env.context_dim
    action_dim = env.arms

    observation_space = spaces.Box(
        low=features.values.min(), high=features.values.max()
    )
    action_space = spaces.Discrete(env.arms)
    pop = create_population(
        algo="NeuralUCB",  # Algorithm
        observation_space=observation_space,  # State dimension
        action_space=action_space,  # Action dimension
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POP_SIZE"],  # Population size
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
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )
    mutations = Mutations(
        algo="NeuralUCB",  # Algorithm
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.5,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0.2,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        rl_hp_selection=["lr", "batch_size"],  # Learning HPs to choose from
        mutation_sd=0.1,  # Mutation strength  # Network architecture
        rand_seed=1,  # Random seed
        device=device,
    )

    max_steps = 10000  # Max steps per episode
    episode_steps = 500  # Steps in episode
    evo_steps = 500  # Evolution frequency
    eval_steps = 500  # Evaluation steps per episode
    eval_loop = 1  # Number of evaluation episodes

    print("Training...")

    wandb.init(
        # set the wandb project where this run will be logged
        project="AgileRL-Bandits",
        name="NeuralUCB-{}".format(datetime.now().strftime("%m%d%Y%H%M%S")),
        # track hyperparameters and run metadata
        config=INIT_HP,
    )

    total_steps = 0
    evo_count = 0

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps, unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent_idx, agent in enumerate(pop):  # Loop through population
            score = 0
            losses = []
            context = env.reset()  # Reset environment at start of episode
            for idx_step in range(episode_steps):
                if INIT_HP["CHANNELS_LAST"]:
                    context = obs_channels_to_first(context)
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
                score += reward
                agent.regret.append(agent.regret[-1] + 1 - reward)

            agent.scores.append(score)
            pop_episode_scores.append(score)
            agent.steps[-1] += episode_steps
            total_steps += episode_steps
            pbar.update(episode_steps // len(pop))

            wandb_dict = {
                "global_step": total_steps,
                "train/loss": np.mean(losses),
                "train/score": score,
                "train/mean_regret": np.mean([agent.regret[-1] for agent in pop]),
            }
            wandb.log(wandb_dict)

        # Evaluate population
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
        print(f"Regret: {[agent.regret[-1] for agent in pop]}")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(
            f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
        )

        if pop[0].steps[-1] // evo_steps > evo_count:
            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)
            evo_count += 1

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    pbar.close()
    env.close()
