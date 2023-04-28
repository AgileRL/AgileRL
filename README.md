# AgileRL
<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/222710068-e09a4e3c-368c-458a-9e01-b68674806887.png height="120">
</p>
<p align="center"><b>Reinforcement learning streamlined.</b><br>Easier and faster reinforcement learning with RLOps. Visit our <a href="https://agilerl.com">website</a>. View <a href="https://agilerl.readthedocs.io/en/latest/">documentation</a>.<br>Join the <a href="https://discord.gg/eB8HyTA2ux">Discord Server</a> to collaborate.</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/agilerl/badge/?version=latest)](https://agilerl.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/agilerl)](https://pypi.python.org/pypi/agilerl/)
[![Discord](https://dcbadge.vercel.app/api/server/eB8HyTA2ux?style=flat)](https://discord.gg/eB8HyTA2ux)

</div>

This is a Deep Reinforcement Learning library focused on improving development by introducing RLOps - MLOps for reinforcement learning.
  
This library is initially focused on reducing the time taken for training models and hyperparameter optimization (HPO) by pioneering evolutionary HPO techniques for reinforcement learning.<br>
Evolutionary HPO has been shown to drastically reduce overall training times by automatically converging on optimal hyperparameters, without requiring numerous training runs.<br>
We are constantly adding more algorithms, with a view to add hierarchical and multi-agent algorithms soon.

### Benchmarks

Reinforcement learning algorithms and libraries are usually benchmarked once the optimal hyperparameters for training are known, but it often takes hundreds or thousands of experiments to discover these. This is unrealistic and does not reflect the true, total time taken for training. What if we could remove the need to conduct all these prior experiments?

In the charts below, a single AgileRL run, which automatically tunes hyperparameters, is benchmarked against Optuna's multiple training runs traditionally required for hyperparameter optimization, demonstrating the real time savings possible. Global steps is the sum of every step taken by any agent in the environment, including across an entire population.

<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/227481592-27a9688f-7c0a-4655-ab32-90d659a71c69.png height="500">
</p>
<p align="center">AgileRL offers an order of magnitude speed up in hyperparameter optimization vs popular reinforcement learning training frameworks combined with Optuna. Remove the need for multiple training runs and save yourself hours.</p>

## Get Started
Install as a package with pip: 
```bash
pip install agilerl
```
Or install in development mode: (Recommended due to nascent nature of this library)
```bash
git clone https://github.com/AgileRL/AgileRL.git && cd AgileRL
pip install -r requirements.txt
```
If using ILQL on Wordle, download and unzip data.zip <a href="https://drive.google.com/drive/folders/13LFspsFQ-7XIlFjnsZttKf4nfVDlnmW2?usp=sharing">here</a>. 

Demo:
```bash
python demo.py
```

## Algorithms implemented (more coming soon!)
  * DQN
  * DDPG
  * ILQL

## Train an agent
Before starting training, there are some meta-hyperparameters and settings that must be set. These are defined in <code>INIT_HP</code>, for general parameters, and <code>MUTATION_PARAMS</code>, which define the evolutionary probabilities, and <code>NET_CONFIG</code>, which defines the network architecture. For example:
```python
INIT_HP = {
    'ENV_NAME': 'LunarLander-v2',   # Gym environment name
    'ALGO': 'DQN',                  # Algorithm
    'CHANNELS_LAST': False,         # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    'BATCH_SIZE': 256,              # Batch size
    'LR': 1e-3,                     # Learning rate
    'EPISODES': 2000,               # Max no. episodes
    'TARGET_SCORE': 200.,           # Early training stop at avg score of last 100 episodes
    'GAMMA': 0.99,                  # Discount factor
    'MEMORY_SIZE': 10000,           # Max memory buffer size
    'LEARN_STEP': 1,                # Learning frequency
    'TAU': 1e-3,                    # For soft update of target parameters
    'TOURN_SIZE': 2,                # Tournament size
    'ELITISM': True,                # Elitism in tournament selection
    'POP_SIZE': 6,                  # Population size
    'EVO_EPOCHS': 20,               # Evolution frequency
    'POLICY_FREQ': 2,               # Policy network update frequency
    'WANDB': True                   # Log with Weights and Biases
}
```
```python
MUTATION_PARAMS = {
    # Relative probabilities
    'NO_MUT': 0.4,                              # No mutation
    'ARCH_MUT': 0.2,                            # Architecture mutation
    'NEW_LAYER': 0.2,                           # New layer mutation
    'PARAMS_MUT': 0.2,                          # Network parameters mutation
    'ACT_MUT': 0,                               # Activation layer mutation
    'RL_HP_MUT': 0.2,                           # Learning HP mutation
    'RL_HP_SELECTION': ['lr', 'batch_size'],    # Learning HPs to choose from
    'MUT_SD': 0.1,                              # Mutation strength
    'RAND_SEED': 1,                             # Random seed
}
```
```python
NET_CONFIG = {
    'arch': 'mlp',      # Network architecture
    'h_size': [32, 32], # Actor hidden size
}
```
First, use <code>utils.initialPopulation</code> to create a list of agents - our population that will evolve and mutate to the optimal hyperparameters.
```python
from agilerl.utils import makeVectEnvs, initialPopulation
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = makeVectEnvs(env_name=INIT_HP['ENV_NAME'], num_envs=16)
try:
    state_dim = env.single_observation_space.n          # Discrete observation space
    one_hot = True                                      # Requires one-hot encoding
except:
    state_dim = env.single_observation_space.shape      # Continuous observation space
    one_hot = False                                     # Does not require one-hot encoding
try:
    action_dim = env.single_action_space.n             # Discrete action space
except:
    action_dim = env.single_action_space.shape[0]      # Continuous action space

if INIT_HP['CHANNELS_LAST']:
    state_dim = (state_dim[2], state_dim[0], state_dim[1])

agent_pop = initialPopulation(algo=INIT_HP['ALGO'],                 # Algorithm
                              state_dim=state_dim,                  # State dimension
                              action_dim=action_dim,                # Action dimension
                              one_hot=one_hot,                      # One-hot encoding
                              net_config=NET_CONFIG,                # Network configuration
                              INIT_HP=INIT_HP,                      # Initial hyperparameters
                              population_size=INIT_HP['POP_SIZE'],  # Population size
                              device=torch.device("cuda"))
```
Next, create the tournament, mutations and experience replay buffer objects that allow agents to share memory and efficiently perform evolutionary HPO.
```python
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
import torch

field_names = ["state", "action", "reward", "next_state", "done"]
memory = ReplayBuffer(action_dim=action_dim,                # Number of agent actions
                      memory_size=INIT_HP['MEMORY_SIZE'],   # Max replay buffer size
                      field_names=field_names,              # Field names to store in memory
                      device=torch.device("cuda"))

tournament = TournamentSelection(tournament_size=INIT_HP['TOURN_SIZE'], # Tournament selection size
                                 elitism=INIT_HP['ELITISM'],            # Elitism in tournament selection
                                 population_size=INIT_HP['POP_SIZE'],   # Population size
                                 evo_step=INIT_HP['EVO_EPOCHS'])        # Evaluate using last N fitness scores

mutations = Mutations(algo=INIT_HP['ALGO'],                                 # Algorithm
                      no_mutation=MUTATION_PARAMS['NO_MUT'],                # No mutation
                      architecture=MUTATION_PARAMS['ARCH_MUT'],             # Architecture mutation
                      new_layer_prob=MUTATION_PARAMS['NEW_LAYER'],          # New layer mutation
                      parameters=MUTATION_PARAMS['PARAMS_MUT'],             # Network parameters mutation
                      activation=MUTATION_PARAMS['ACT_MUT'],                # Activation layer mutation
                      rl_hp=MUTATION_PARAMS['RL_HP_MUT'],                   # Learning HP mutation
                      rl_hp_selection=MUTATION_PARAMS['RL_HP_SELECTION'],   # Learning HPs to choose from
                      mutation_sd=MUTATION_PARAMS['MUT_SD'],                # Mutation strength
                      arch=NET_CONFIG['arch'],                              # Network architecture
                      rand_seed=MUTATION_PARAMS['RAND_SEED'],               # Random seed
                      device=torch.device("cuda"))
```
The easiest training loop implementation is to use our <code>training.train()</code> function. It requires the <code>agent</code> have functions <code>getAction()</code> and <code>learn().</code>
```python
from agilerl.training.train import train

trained_pop, pop_fitnesses = train(env=env,                             # Gym-style environment
                                   env_name=INIT_HP['ENV_NAME'],        # Environment name
                                   algo=INIT_HP['ALGO'],                # Algorithm
                                   pop=agent_pop,                       # Population of agents
                                   memory=memory,                       # Replay buffer
                                   swap_channels=False,                 # Swap image channel from last to first
                                   n_episodes=INIT_HP['EPISODES'],      # Max number of training episodes
                                   evo_epochs=INIT_HP['EVO_EPOCHS'],    # Evolution frequency
                                   evo_loop=1,                          # Number of evaluation episodes per agent
                                   target=INIT_HP['TARGET_SCORE'],      # Target score for early stopping
                                   tournament=tournament,               # Tournament selection object
                                   mutation=mutations,                  # Mutations object
                                   wb=INIT_HP['WANDB'],                 # Weights and Biases tracking
                                   device=torch.device("cuda"))
```

### Custom Training Loop
Alternatively, use a custom training loop. Combining all of the above:

```python
from agilerl.utils import makeVectEnvs, initialPopulation
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
import gymnasium as gym
import numpy as np
import torch

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
            'CHANNELS_LAST': False  # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
          }

pop = initialPopulation(algo='DQN',             # Algorithm
                        state_dim=(8,),         # State dimension
                        action_dim=4,           # Action dimension
                        one_hot=False,          # One-hot encoding
                        net_config=NET_CONFIG,  # Network configuration
                        INIT_HP=INIT_HP,        # Initial hyperparameters
                        population_size=6,      # Population size
                        device=torch.device("cuda"))

field_names = ["state", "action", "reward", "next_state", "done"]
memory = ReplayBuffer(action_dim=4,             # Number of agent actions
                      memory_size=10000,        # Max replay buffer size
                      field_names=field_names,  # Field names to store in memory
                      device=torch.device("cuda"))

tournament = TournamentSelection(tournament_size=2, # Tournament selection size
                                 elitism=True,      # Elitism in tournament selection
                                 population_size=6, # Population size
                                 evo_step=1)        # Evaluate using last N fitness scores

mutations = Mutations(algo='DQN',                           # Algorithm
                      no_mutation=0.4,                      # No mutation
                      architecture=0.2,                     # Architecture mutation
                      new_layer_prob=0.2,                   # New layer mutation
                      parameters=0.2,                       # Network parameters mutation
                      activation=0,                         # Activation layer mutation
                      rl_hp=0.2,                            # Learning HP mutation
                      rl_hp_selection=['lr', 'batch_size'], # Learning HPs to choose from
                      mutation_sd=0.1,                      # Mutation strength
                      arch=NET_CONFIG['arch'],              # Network architecture
                      rand_seed=1,                          # Random seed
                      device=torch.device("cuda"))

max_episodes = 1000 # Max training episodes
max_steps = 500     # Max steps per episode

# Exploration params
eps_start = 1.0     # Max exploration
eps_end = 0.1       # Min exploration
eps_decay = 0.995   # Decay per episode
epsilon = eps_start

evo_epochs = 5      # Evolution frequency
evo_loop = 1        # Number of evaluation episodes

env = makeVectEnvs('LunarLander-v2', num_envs=16)   # Create environment

# TRAINING LOOP
for idx_epi in range(max_episodes):
    for agent in pop:   # Loop through population
        state = env.reset()[0]  # Reset environment at start of episode
        score = 0
        for idx_step in range(max_steps):
            action = agent.getAction(state, epsilon)    # Get next action from agent
            next_state, reward, done, _, _ = env.step(action)   # Act in environment
            
            # Save experience to replay buffer
            memory.save2memoryVectEnvs(state, action, reward, next_state, done)

            # Learn according to learning frequency
            if memory.counter % agent.learn_step == 0 and len(memory) >= agent.batch_size:
                experiences = memory.sample(agent.batch_size) # Sample replay buffer
                agent.learn(experiences)    # Learn according to agent's RL algorithm
            
            state = next_state
            score += reward

    epsilon = max(eps_end, epsilon*eps_decay) # Update epsilon for exploration

    # Now evolve population if necessary
    if (idx_epi+1) % evo_epochs == 0:
        
        # Evaluate population
        fitnesses = [agent.test(env, swap_channels=False, max_steps=max_steps, loop=evo_loop) for agent in pop]

        print(f'Episode {idx_epi+1}/{max_episodes}')
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}')

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)
```

View <a href="https://agilerl.readthedocs.io/en/latest/">documentation</a>.
