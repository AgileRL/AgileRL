from agilerl.utils.utils import makeVectEnvs, initialPopulation
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.replay_data import ReplayDataset
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from accelerate import Accelerator
import numpy as np
from torch.utils.data import DataLoader
from tqdm import trange

if __name__ == '__main__':

    print('===== AgileRL Online Distributed Demo =====')

    print('Loading accelerator...')
    accelerator = Accelerator()

    NET_CONFIG = {
        'arch': 'mlp',       # Network architecture
        'h_size': [32, 32],  # Actor hidden size
    }

    INIT_HP = {
        'POPULATION_SIZE': 4,   # Population size
        'DOUBLE': True,         # Use double Q-learning in DQN or CQN
        'BATCH_SIZE': 128,      # Batch size
        'LR': 1e-3,             # Learning rate
        'GAMMA': 0.99,          # Discount factor
        'LEARN_STEP': 1,        # Learning frequency
        'TAU': 1e-3,            # For soft update of target network parameters
        'POLICY_FREQ': 2,       # DDPG target network update frequency vs policy network
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'CHANNELS_LAST': False
    }

    env = makeVectEnvs('LunarLander-v2', num_envs=8)   # Create environment
    try:
        state_dim = env.single_observation_space.n          # Discrete observation space
        one_hot = True                                      # Requires one-hot encoding
    except Exception:
        state_dim = env.single_observation_space.shape      # Continuous observation space
        one_hot = False                                     # Does not require one-hot encoding
    try:
        action_dim = env.single_action_space.n             # Discrete action space
    except Exception:
        action_dim = env.single_action_space.shape[0]      # Continuous action space

    if INIT_HP['CHANNELS_LAST']:
        state_dim = (state_dim[2], state_dim[0], state_dim[1])

    pop = initialPopulation(algo='DQN',                 # Algorithm
                            state_dim=state_dim,        # State dimension
                            action_dim=action_dim,      # Action dimension
                            one_hot=False,              # One-hot encoding
                            net_config=NET_CONFIG,      # Network configuration
                            INIT_HP=INIT_HP,            # Initial hyperparameters
                            population_size=INIT_HP['POPULATION_SIZE'], # Population size
                            accelerator=accelerator)    # Accelerator

    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(action_dim=action_dim,    # Number of agent actions
                          memory_size=10000,        # Max replay buffer size
                          field_names=field_names)  # Field names to store in memory
    replay_dataset = ReplayDataset(memory, INIT_HP['BATCH_SIZE'])
    replay_dataloader = DataLoader(replay_dataset)
    replay_dataloader = accelerator.prepare(replay_dataloader)

    tournament = TournamentSelection(tournament_size=2,  # Tournament selection size
                                     elitism=True,      # Elitism in tournament selection
                                     population_size=INIT_HP['POPULATION_SIZE'],  # Population size
                                     evo_step=1)        # Evaluate using last N fitness scores

    mutations = Mutations(algo='DQN',                           # Algorithm
                          no_mutation=0.4,                      # No mutation
                          architecture=0.2,                     # Architecture mutation
                          new_layer_prob=0.2,                   # New layer mutation
                          parameters=0.2,                       # Network parameters mutation
                          activation=0,                         # Activation layer mutation
                          rl_hp=0.2,                            # Learning HP mutation
                          # Learning HPs to choose from
                          rl_hp_selection=['lr', 'batch_size'],
                          mutation_sd=0.1,                      # Mutation strength
                          # Network architecture
                          arch=NET_CONFIG['arch'],
                          rand_seed=1,                          # Random seed
                          accelerator=accelerator)              # Accelerator)

    max_episodes = 1000  # Max training episodes
    max_steps = 500     # Max steps per episode

    # Exploration params
    eps_start = 1.0     # Max exploration
    eps_end = 0.1       # Min exploration
    eps_decay = 0.995   # Decay per episode
    epsilon = eps_start

    evo_epochs = 5      # Evolution frequency
    evo_loop = 1        # Number of evaluation episodes

    print('Training...')

    # TRAINING LOOP
    for idx_epi in trange(max_episodes):
        for agent in pop:   # Loop through population
            state = env.reset()[0]  # Reset environment at start of episode
            score = 0
            for idx_step in range(max_steps):
                # Get next action from agent
                action = agent.getAction(state, epsilon)
                next_state, reward, done, _, _ = env.step(
                    action)   # Act in environment

                # Save experience to replay buffer
                memory.save2memoryVectEnvs(
                    state, action, reward, next_state, done)

                # Learn according to learning frequency
                if memory.counter % agent.learn_step == 0 and len(
                        memory) >= agent.batch_size:
                    # Sample dataloader
                    replay_dataset.batch_size = agent.batch_size
                    experiences = next(iter(replay_dataloader))
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
                    swap_channels=False,
                    max_steps=max_steps,
                    loop=evo_loop) for agent in pop]

            print(f'Episode {idx_epi+1}/{max_episodes}')
            print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
            print(f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}')

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)
