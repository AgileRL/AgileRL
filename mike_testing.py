import gymnasium as gym
import torch
from agilerl.utils.utils import makeVectEnvs, initialPopulation
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.algorithms.td3 import TD3
from agilerl.algorithms.ddpg import DDPG

import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.show()

def tournament_test():
    # 1. Device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 2. Create environment and vectorise
    env = makeVectEnvs('LunarLanderContinuous-v2', num_envs=4)

    # 3. Set-up state_dim and action_dim variables
    try:
        state_dim = env.single_observation_space.n          # Discrete observation space
        one_hot = True                                      # Requires one-hot encoding
    except:
        state_dim = env.single_observation_space.shape      # Continuous observation space
        one_hot = False                                     # Does not require one-hot encoding
    try:
        action_dim = env.single_action_space.n              # Discrete action space
    except:
        action_dim = env.single_action_space.shape[0]       # Continuous action space

    # 4. Set-up the hyperparameters and network configuration
    # Refer to utils.utils.initialPopulation for algo + relevant HPs
    NET_CONFIG = {
        'arch': 'mlp',       # Network architecture
        'h_size': [32, 32],  # Actor hidden size
    }
    INIT_HP = {
        'BATCH_SIZE': 128,      # Batch size
        'LR': 1e-3,             # Learning rate
        'GAMMA': 0.99,          # Discount factor
        'LEARN_STEP': 1,        # Learning frequency
        'TAU': 1e-3,            # For soft update of target network parameters
        'POLICY_FREQ':2,
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        'CHANNELS_LAST': False
    }
    
    # 5. Create a population of DDPG algos
    pop = initialPopulation(algo='TD3',            # Algorithm
                            state_dim=state_dim,    # State dimension
                            action_dim=action_dim,  # Action dimension
                            one_hot=one_hot,        # One-hot encoding
                            net_config=NET_CONFIG,  # Network configuration
                            INIT_HP=INIT_HP,        # Initial hyperparameters
                            population_size=3,      # Population size
                            device=torch.device(device))    

    # 6. Create the replay buffer
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(action_dim=action_dim,    # Number of agent actions
                          memory_size=10000,        # Max replay buffer size
                          field_names=field_names,  # Field names to store in memory
                          device=torch.device(device))  
    
    # 7. Create a tournament object
    tournament = TournamentSelection(tournament_size=2, # Tournament selection size
                                     elitism=True,      # Elitism in tournament selection
                                     population_size=3, # Population size
                                     evo_step=1)        # Evaluate using last N fitness scores
    
    # 8. Create a mutations object
    mutations = Mutations(algo='TD3',                           # Algorithm
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
                          device=torch.device(device))

    # 9 . Train the sucker 
    max_episodes = 100  # Max training episodes
    max_steps = 500     # Max steps per episode

    # Exploration params
    eps_start = 1.0     # Max exploration
    eps_end = 0.1       # Min exploration
    eps_decay = 0.995   # Decay per episode
    epsilon = eps_start

    evo_epochs = 5      # Evolution frequency
    evo_loop = 1        # Number of evaluation episodes

    print('===== AgileRL Demo =====')
    print('Verbose off. Add a progress bar to view training progress more frequently.')
    print('Training...')

    # TRAINING LOOP
    for idx_epi in range(max_episodes):
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
                    experiences = memory.sample(
                        agent.batch_size)  # Sample replay buffer
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
            print(
                f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}')

            # Tournament selection and population mutation
            elite, pop = tournament.select(pop)
            pop = mutations.mutation(pop)



def agent_test():
    env = makeVectEnvs('LunarLanderContinuous-v2', num_envs=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_action = float(env.single_action_space.high[0])


    try:
        state_dim = env.single_observation_space.n          # Discrete observation space
        one_hot = True                                      # Requires one-hot encoding
    except:
        state_dim = env.single_observation_space.shape      # Continuous observation space
        one_hot = False                                     # Does not require one-hot encoding
    try:
        action_dim = env.single_action_space.n              # Discrete action space
    except:
        action_dim = env.single_action_space.shape[0]       # Continuous action space

    NET_CONFIG = {
      'arch': 'mlp',      # Network architecture
      'h_size': [400, 300]  # Network hidden size
    }

    agent = TD3(state_dim=state_dim,
                action_dim=action_dim,
                one_hot=False,
                max_action=max_action,
                index=0,
                net_config=NET_CONFIG,
                batch_size=100,
                tau=0.005,
                lr = 0.001,
                device=torch.device(device))
    
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        action_dim, 100000, field_names=field_names, device=torch.device(device))

    np.random.seed(0)
    score_history = []
    episodes = 1000
    for i in range(episodes):
        done = [False]
        score = 0
        state = env.reset()[0]

        while not done[0]:
            action = agent.getAction(state)
            next_state, reward, done, _, _ = env.step(action)
            memory.save2memoryVectEnvs(state, action, reward, next_state, done)
                    
            # Learn according to learning frequency
            if memory.counter % agent.learn_step == 0 and len(
                    memory) >= agent.batch_size:
                experiences = memory.sample(
                    agent.batch_size)   # Sample replay buffer
                # Learn according to agent's RL algorithm
                agent.learn(experiences)

            score += reward 
            state = next_state
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        print('episode', i, 'score', avg_score,
          'average score', avg_score)

    x = [i+1 for i in range(episodes)]
    plot_learning_curve(x, score_history)

if __name__ == "__main__":
    agent_test()