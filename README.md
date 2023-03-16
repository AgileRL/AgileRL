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
  
This library is initially focused on reducing the time taken for training models and hyperparameter optimisation (HPO) by pioneering evolutionary HPO techniques for reinforcement learning.<br>
Evolutionary HPO has been shown to drastically reduce overall training times by automatically converging on optimal hyperparameters, without requiring numerous training runs.<br>
We are constantly adding more algorithms, with a view to add hierarchical and multi-agent algorithms soon.

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

## Algorithms implemented (more coming soon!)
  * DQN
  * DDPG

## Train an agent
Before starting training, there are some meta-hyperparameters and settings that must be set. These are defined in <code>INIT_HP</code>, for general parameters, and <code>MUTATION_PARAMS</code>, which define the evolutionary probabilities. For example:
```python
INIT_HP = {
    'ENV_NAME': 'LunarLander-v2',   # Gym environment name
    'ALGO': 'DQN',                  # Algorithm
    'HIDDEN_SIZE': [64,64],         # Actor network hidden size
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
First, use <code>utils.initialPopulation</code> to create a list of agents - our population that will evolve and mutate to the optimal hyperparameters.
```python
from agilerl.utils import makeVectEnvs, initialPopulation
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = makeVectEnvs(env_name=INIT_HP['ENV_NAME'], num_envs=16)
try:
    num_states = env.single_observation_space.n
    one_hot = True
except:
    num_states = env.single_observation_space.shape[0]
    one_hot = True
try:
    num_actions = env.single_action_space.n
except:
    num_actions = env.single_action_space.shape[0]

agent_pop = initialPopulation(INIT_HP['ALGO'],
  num_states,
  num_actions,
  one_hot,
  INIT_HP,
  INIT_HP['POP_SIZE'],
  device=device)
```
Next, create the tournament, mutations and experience replay buffer objects that allow agents to share memory and efficiently perform evolutionary HPO.
```python
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
import torch

field_names = ["state", "action", "reward", "next_state", "done"]
memory = ReplayBuffer(num_actions, INIT_HP['MEMORY_SIZE'], field_names=field_names, device=device)

tournament = TournamentSelection(INIT_HP['TOURN_SIZE'],
    INIT_HP['ELITISM'],
    INIT_HP['POP_SIZE'],
    INIT_HP['EVO_EPOCHS'])
    
mutations = Mutations(algo=INIT_HP['ALGO'],
    no_mutation=MUTATION_PARAMS['NO_MUT'], 
    architecture=MUTATION_PARAMS['ARCH_MUT'], 
    new_layer_prob=MUTATION_PARAMS['NEW_LAYER'], 
    parameters=MUTATION_PARAMS['PARAMS_MUT'], 
    activation=MUTATION_PARAMS['ACT_MUT'], 
    rl_hp=MUTATION_PARAMS['RL_HP_MUT'], 
    rl_hp_selection=MUTATION_PARAMS['RL_HP_SELECTION'], 
    mutation_sd=MUTATION_PARAMS['MUT_SD'], 
    rand_seed=MUTATION_PARAMS['RAND_SEED'],
    device=device)
```
The easiest training loop implementation is to use our <code>training.train()</code> function. It requires the <code>agent</code> have functions <code>getAction()</code> and <code>learn().</code>
```python
from agilerl.training.train import train

trained_pop, pop_fitnesses = train(env,
    INIT_HP['ENV_NAME'],
    INIT_HP['ALGO'],
    agent_pop,
    memory=memory,
    n_episodes=INIT_HP['EPISODES'],
    evo_epochs=INIT_HP['EVO_EPOCHS'],
    evo_loop=1,
    target=INIT_HP['TARGET_SCORE'],
    tournament=tournament,
    mutation=mutations,
    wb=INIT_HP['WANDB'],
    device=device)
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

INIT_HP = {
            'HIDDEN_SIZE': [64,64], # Actor network hidden size
            'BATCH_SIZE': 128,      # Batch size
            'LR': 1e-3,             # Learning rate
            'GAMMA': 0.99,          # Discount factor
            'LEARN_STEP': 1,        # Learning frequency
            'TAU': 1e-3             # For soft update of target network parameters
            }

env = makeVectEnvs('LunarLander-v2', num_envs=16)   # Create environment
try:
    num_states = env.single_observation_space.n         # Discrete observation space
    one_hot = True                                      # Requires one-hot encoding
except:
    num_states = env.single_observation_space.shape[0]  # Continuous observation space
    one_hot = False                                     # Does not require one-hot encoding
try:
    num_actions = env.single_action_space.n             # Discrete action space
except:
    num_actions = env.single_action_space.shape[0]      # Continuous action space

pop = initialPopulation(algo='DQN',                 # Algorithm
                        num_states=num_states,      # State dimension
                        num_actions=num_actions,    # Action dimension
                        one_hot=one_hot,            # One-hot encoding
                        INIT_HP=INIT_HP,            # Initial hyperparameters
                        population_size=6,          # Population size
                        device=torch.device("cuda"))

field_names = ["state", "action", "reward", "next_state", "done"]
memory = ReplayBuffer(n_actions=num_actions,    # Number of agent actions
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
        fitnesses = [agent.test(env, max_steps=max_steps, loop=evo_loop) for agent in pop]

        print(f'Episode {idx_epi+1}/{max_episodes}')
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(f'100 fitness avgs: {["%.2f"%np.mean(agent.fitness[-100:]) for agent in pop]}')

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)
```

View <a href="https://agilerl.readthedocs.io/en/latest/">documentation</a>.
