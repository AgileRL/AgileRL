# AgileRL
<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/222710068-e09a4e3c-368c-458a-9e01-b68674806887.png height="120">
</p>
<p align="center"><b>Reinforcement learning streamlined.</b> Easier and faster reinforcement learning with RLOps.</p>

This is a Deep Reinforcement Learning library focused on improving development by introducing RLOps - MLOps for reinforcement learning.
  
This library is initially focused on reducing the time taken for training models and hyperparameter optimisation (HPO) by pioneering evolutionary HPO techniques for RL.<br>
Evolutionary HPO has been shown to drastically reduce overall training times by automatically converging on optimal hyperparameters, without requiring numerous training runs.<br>
We are constantly adding more algorithms, with a view to add hierarchical and multi-agent algorithms soon.

## Get Started
```
    git clone https://github.com/AgileRL/AgileRL
    pip install -r requirements.txt
```
```    
    python benchmarking.py
```

## Use in your training loop
Before starting training, there are some meta-hyperparameters and settings that must be set. These are defined in <code>INIT_HP</code>, for general parameters, and <code>MUTATION_PARAMS</code>, which define the evolutionary probabilities. For example:
```
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
```
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
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.make(INIT_HP['ENV_NAME'], render_mode='rgb_array')
num_states = env.observation_space.shape[0]
try:
    num_actions = env.action_space.n
except:
    num_actions = env.action_space.shape[0]

agent_pop = initialPopulation(INIT_HP['ALGO'], num_states, num_actions, INIT_HP, INIT_HP['POP_SIZE'], device=device)
```
Next, create the tournament, mutations and experience replay buffer objects that allow agents to share memory and efficiently perform evolutionary HPO.
```
field_names = ["state", "action", "reward", "next_state", "done"]
memory = ReplayBuffer(num_actions, INIT_HP['MEMORY_SIZE'], field_names=field_names, device=device)
tournament = TournamentSelection(INIT_HP['TOURN_SIZE'], INIT_HP['ELITISM'], INIT_HP['POP_SIZE'], INIT_HP['EVO_EPOCHS'])
mutations = Mutations(no_mutation=MUTATION_PARAMS['NO_MUT'], 
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
```
trained_pop, pop_fitnesses = train(env,
    INIT_HP['ENV_NAME'],
    INIT_HP['ALGO'],
    agent_pop,
    memory=memory,
    n_episodes=INIT_HP['EPISODES'],
    evo_epochs=INIT_HP['EVO_EPOCHS'],
    evo_loop=1,
    target=INIT_HP['TARGET_SCORE'],
    chkpt=INIT_HP['SAVE_CHKPT'],
    tournament=tournament,
    mutation=mutations,
    wb=INIT_HP['WANDB'],
    device=device)
```

## Algorithms implemented
  * DQN
  * DDPG


