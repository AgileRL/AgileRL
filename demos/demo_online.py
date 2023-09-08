import numpy as np
import torch
from tqdm import trange

from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import initialPopulation, makeVectEnvs

# !Note: If you are running this demo without having installed agilerl,
# uncomment and place the following above agilerl imports:

# import sys
# sys.path.append('../')


if __name__ == "__main__":
    print("===== AgileRL Online Demo =====")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NET_CONFIG = {
        "arch": "mlp",  # Network architecture
        "h_size": [32, 32],  # Actor hidden size
    }

    INIT_HP = {
        "POPULATION_SIZE": 4,  # Population size
        "DOUBLE": True,  # Use double Q-learning
        "BATCH_SIZE": 128,  # Batch size
        "LR": 1e-3,  # Learning rate
        "GAMMA": 0.99,  # Discount factor
        "LEARN_STEP": 1,  # Learning frequency
        "TAU": 1e-3,  # For soft update of target network parameters
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "CHANNELS_LAST": False,
    }

    env = makeVectEnvs("LunarLander-v2", num_envs=8)  # Create environment
    try:
        state_dim = env.single_observation_space.n  # Discrete observation space
        one_hot = True  # Requires one-hot encoding
    except Exception:
        state_dim = env.single_observation_space.shape  # Continuous observation space
        one_hot = False  # Does not require one-hot encoding
    try:
        action_dim = env.single_action_space.n  # Discrete action space
    except Exception:
        action_dim = env.single_action_space.shape[0]  # Continuous action space

    if INIT_HP["CHANNELS_LAST"]:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    pop = initialPopulation(
        algo="DQN",  # Algorithm
        state_dim=state_dim,  # State dimension
        action_dim=action_dim,  # Action dimension
        one_hot=one_hot,  # One-hot encoding
        net_config=NET_CONFIG,  # Network configuration
        INIT_HP=INIT_HP,  # Initial hyperparameters
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        device=device,
    )

    field_names = ["state", "action", "reward", "next_state", "done"]
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
        algo="DQN",  # Algorithm
        no_mutation=0.4,  # No mutation
        architecture=0.2,  # Architecture mutation
        new_layer_prob=0.2,  # New layer mutation
        parameters=0.2,  # Network parameters mutation
        activation=0,  # Activation layer mutation
        rl_hp=0.2,  # Learning HP mutation
        rl_hp_selection=["lr", "batch_size"],  # Learning HPs to choose from
        mutation_sd=0.1,  # Mutation strength
        arch=NET_CONFIG["arch"],  # Network architecture
        rand_seed=1,  # Random seed
        device=device,
    )

    max_episodes = 1000  # Max training episodes
    max_steps = 500  # Max steps per episode

    # Exploration params
    eps_start = 1.0  # Max exploration
    eps_end = 0.1  # Min exploration
    eps_decay = 0.995  # Decay per episode
    epsilon = eps_start

    evo_epochs = 5  # Evolution frequency
    evo_loop = 1  # Number of evaluation episodes

    print("Training...")

    # TRAINING LOOP
    for idx_epi in trange(max_episodes):
        for agent in pop:  # Loop through population
            state = env.reset()[0]  # Reset environment at start of episode
            score = 0
            for idx_step in range(max_steps):
                # Get next action from agent
                action = agent.getAction(state, epsilon)
                next_state, reward, done, _, _ = env.step(action)  # Act in environment

                # Save experience to replay buffer
                if INIT_HP["CHANNELS_LAST"]:
                    memory.save2memoryVectEnvs(
                        state, action, reward, np.moveaxis(next_state, [3], [1]), done
                    )
                else:
                    memory.save2memoryVectEnvs(state, action, reward, next_state, done)

                # Learn according to learning frequency
                if (
                    memory.counter % agent.learn_step == 0
                    and len(memory) >= agent.batch_size
                ):
                    experiences = memory.sample(
                        agent.batch_size
                    )  # Sample replay buffer
                    # Learn according to agent's RL algorithm
                    agent.learn(experiences)

                state = next_state
                score += reward

        # Update epsilon for exploration
        epsilon = max(eps_end, epsilon * eps_decay)

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
            print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
            print(
                f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}'
            )

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)

    env.close()
