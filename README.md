# AgileRL
<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/222710068-e09a4e3c-368c-458a-9e01-b68674806887.png height="120">
</p>
<p align="center"><b>Reinforcement learning streamlined.</b><br>Easier and faster reinforcement learning with RLOps. Visit our <a href="https://agilerl.com">website</a>. View <a href="https://docs.agilerl.com">documentation</a>.<br>Join the <a href="https://discord.gg/eB8HyTA2ux">Discord Server</a> for questions, help and collaboration.</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/agilerl/badge/?version=latest)](https://docs.agilerl.com/en/latest/?badge=latest)
[![Downloads](https://static.pepy.tech/badge/agilerl)](https://pypi.python.org/pypi/agilerl/)
[![Discord](https://dcbadge.vercel.app/api/server/eB8HyTA2ux?style=flat)](https://discord.gg/eB8HyTA2ux)

</div>

AgileRL is a Deep Reinforcement Learning library focused on improving development by introducing RLOps - MLOps for reinforcement learning.

This library is initially focused on reducing the time taken for training models and hyperparameter optimization (HPO) by pioneering [evolutionary HPO techniques](https://docs.agilerl.com/en/latest/evo_hyperparam_opt/index.html) for reinforcement learning.<br>
Evolutionary HPO has been shown to drastically reduce overall training times by automatically converging on optimal hyperparameters, without requiring numerous training runs.<br>
We are constantly adding more algorithms and features. AgileRL already includes state-of-the-art evolvable [on-policy](https://docs.agilerl.com/en/latest/on_policy/index.html), [off-policy](https://docs.agilerl.com/en/latest/off_policy/index.html), [offline](https://docs.agilerl.com/en/latest/offline_training/index.html), [multi-agent](https://docs.agilerl.com/en/latest/multi_agent_training/index.html) and [contextual multi-armed bandit](https://docs.agilerl.com/en/latest/bandits/index.html) reinforcement learning algorithms with [distributed training](https://docs.agilerl.com/en/latest/distributed_training/index.html).

<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/236407686-21363eb3-ffcf-419f-b019-0be4ddf1ed4a.gif width="100%" max-width="900">
</p>
<p align="center">AgileRL offers 10x faster hyperparameter optimization than SOTA.</p>

## Table of Contents
  * [Get Started](#get-started)
  * [Benchmarks](#benchmarks)
  * [Tutorials](#tutorials)
  * [Algorithms implemented](#evolvable-algorithms-implemented-more-coming-soon)
  * [Train an agent](#train-an-agent-to-beat-a-gym-environment)
  * [Citing AgileRL](#citing-agilerl)

## Get Started

To see the full AgileRL documentation, including tutorials, visit our [documentation site](https://docs.agilerl.com/). To ask questions and get help, collaborate, or discuss anything related to reinforcement learning, join the [AgileRL Discord Server](https://discord.gg/eB8HyTA2ux).

Install as a package with pip:
```bash
pip install agilerl
```
Or install in development mode:
```bash
git clone https://github.com/AgileRL/AgileRL.git && cd AgileRL
pip install -e .
```

Demo:
```bash
cd demos
python demo_off_policy.py
```

## Benchmarks

Reinforcement learning algorithms and libraries are usually benchmarked once the optimal hyperparameters for training are known, but it often takes hundreds or thousands of experiments to discover these. This is unrealistic and does not reflect the true, total time taken for training. What if we could remove the need to conduct all these prior experiments?

In the charts below, a single AgileRL run, which automatically tunes hyperparameters, is benchmarked against Optuna's multiple training runs traditionally required for hyperparameter optimization, demonstrating the real time savings possible. Global steps is the sum of every step taken by any agent in the environment, including across an entire population.

<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/227481592-27a9688f-7c0a-4655-ab32-90d659a71c69.png min-width="100%" width="600">
</p>
<p align="center">AgileRL offers an order of magnitude speed up in hyperparameter optimization vs popular reinforcement learning training frameworks combined with Optuna. Remove the need for multiple training runs and save yourself hours.</p>

AgileRL also supports multi-agent reinforcement learning using the Petting Zoo-style (parallel API). The charts below highlight the performance of our MADDPG and MATD3 algorithms with evolutionary hyper-parameter optimisation (HPO), benchmarked against epymarl's MADDPG algorithm with grid-search HPO for the simple speaker listener and simple spread environments.

<p align="center">
  <img src=https://github-production-user-asset-6210df.s3.amazonaws.com/118982716/264712154-4965ea5f-b777-423c-989b-e4db86eda3bd.png  min-width="100%" width="700">
</p>

## Tutorials
We are in the process of creating tutorials on how to use AgileRL and train agents on a variety of tasks.

Currently, we have [tutorials for single-agent tasks](https://docs.agilerl.com/en/latest/tutorials/gymnasium/index.html)
that will guide you through the process of training both on and off-policy agents to beat a variety of Gymnasium environments. Additionally, we have [multi-agent tutorials](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/index.html) that make use of PettingZoo environments such as training DQN to play Connect Four with curriculum learning and self-play, and also for multi-agent tasks in MPE environments. The [tutorial on using hierarchical curriculum learning](https://docs.agilerl.com/en/latest/tutorials/skills/index.html) shows how to teach agents Skills and combine them to achieve an end goal. There are also [tutorials for contextual multi-arm bandits](https://docs.agilerl.com/en/latest/tutorials/bandits/index.html), which learn to make the correct decision in environments that only have one timestep.

The demo files in ``demos`` also provide examples on how to train agents using AgileRL, and more information can be found in our <a href="https://docs.agilerl.com">documentation</a>.

## Evolvable algorithms (more coming soon!)

  ### Single-agent algorithms

  | RL         | Algorithm |
  | ---------- | --------- |
  | [On-Policy](https://docs.agilerl.com/en/latest/on_policy/index.html)  | [Proximal Policy Optimization (PPO)](https://docs.agilerl.com/en/latest/api/algorithms/ppo.html) |
  | [Off-Policy](https://docs.agilerl.com/en/latest/off_policy/index.html) | [Deep Q Learning (DQN)](https://docs.agilerl.com/en/latest/api/algorithms/dqn.html) <br>  [Rainbow DQN](https://docs.agilerl.com/en/latest/api/algorithms/dqn_rainbow.html) <br> [Deep Deterministic Policy Gradient (DDPG)](https://docs.agilerl.com/en/latest/api/algorithms/ddpg.html) <br> [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://docs.agilerl.com/en/latest/api/algorithms/td3.html) |
  | [Offline](https://docs.agilerl.com/en/latest/offline_training/index.html)    | [Conservative Q-Learning (CQL)](https://docs.agilerl.com/en/latest/api/algorithms/cql.html) <br>  [Implicit Language Q-Learning (ILQL)](https://docs.agilerl.com/en/latest/api/algorithms/ilql.html) |

  ### Multi-agent algorithms

  | RL         | Algorithm |
  | ---------- | --------- |
  | [Multi-agent](https://docs.agilerl.com/en/latest/multi_agent_training/index.html) | [Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](https://docs.agilerl.com/en/latest/api/algorithms/maddpg.html) <br> [Multi-Agent Twin-Delayed Deep Deterministic Policy Gradient (MATD3)](https://docs.agilerl.com/en/latest/api/algorithms/matd3.html) |

  ### Contextual multi-armed bandit algorithms

  | RL         | Algorithm |
  | ---------- | --------- |
  | [Bandits](https://docs.agilerl.com/en/latest/bandits/index.html) | [Neural Contextual Bandits with UCB-based Exploration (NeuralUCB)](https://docs.agilerl.com/en/latest/api/algorithms/neural_ucb.html) <br> [Neural Contextual Bandits with Thompson Sampling (NeuralTS)](https://docs.agilerl.com/en/latest/api/algorithms/neural_ts.html) |

## Train an agent to beat a Gym environment

Before starting training, there are some meta-hyperparameters and settings that must be set. These are defined in <code>INIT_HP</code>, for general parameters, and <code>MUTATION_PARAMS</code>, which define the evolutionary probabilities, and <code>NET_CONFIG</code>, which defines the network architecture. For example:
```python
INIT_HP = {
    'ENV_NAME': 'LunarLander-v2',   # Gym environment name
    'ALGO': 'DQN',                  # Algorithm
    'DOUBLE': True,                 # Use double Q-learning
    'CHANNELS_LAST': False,         # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
    'BATCH_SIZE': 256,              # Batch size
    'LR': 1e-3,                     # Learning rate
    'MAX_STEPS': 1_000_000,         # Max no. steps
    'TARGET_SCORE': 200.,           # Early training stop at avg score of last 100 episodes
    'GAMMA': 0.99,                  # Discount factor
    'MEMORY_SIZE': 10000,           # Max memory buffer size
    'LEARN_STEP': 1,                # Learning frequency
    'TAU': 1e-3,                    # For soft update of target parameters
    'TOURN_SIZE': 2,                # Tournament size
    'ELITISM': True,                # Elitism in tournament selection
    'POP_SIZE': 6,                  # Population size
    'EVO_STEPS': 10_000,            # Evolution frequency
    'EVAL_STEPS': None,             # Evaluation steps
    'EVAL_LOOP': 1,                 # Evaluation episodes
    'LEARNING_DELAY': 1000,         # Steps before starting learning
    'WANDB': True,                  # Log with Weights and Biases
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
    'hidden_size': [32, 32], # Actor hidden size
}
```
First, use <code>utils.utils.create_population</code> to create a list of agents - our population that will evolve and mutate to the optimal hyperparameters.
```python
from agilerl.utils.utils import make_vect_envs, create_population
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_envs = 16
env = make_vect_envs(env_name=INIT_HP['ENV_NAME'], num_envs=num_envs)
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

agent_pop = create_population(
    algo=INIT_HP['ALGO'],                 # Algorithm
    state_dim=state_dim,                  # State dimension
    action_dim=action_dim,                # Action dimension
    one_hot=one_hot,                      # One-hot encoding
    net_config=NET_CONFIG,                # Network configuration
    INIT_HP=INIT_HP,                      # Initial hyperparameters
    population_size=INIT_HP['POP_SIZE'],  # Population size
    num_envs=num_envs,                    # Number of vectorized environments
    device=device,
)
```
Next, create the tournament, mutations and experience replay buffer objects that allow agents to share memory and efficiently perform evolutionary HPO.
```python
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations

field_names = ["state", "action", "reward", "next_state", "done"]
memory = ReplayBuffer(
    memory_size=INIT_HP['MEMORY_SIZE'],   # Max replay buffer size
    field_names=field_names,              # Field names to store in memory
    device=device,
)

tournament = TournamentSelection(
    tournament_size=INIT_HP['TOURN_SIZE'], # Tournament selection size
    elitism=INIT_HP['ELITISM'],            # Elitism in tournament selection
    population_size=INIT_HP['POP_SIZE'],   # Population size
    eval_loop=INIT_HP['EVAL_LOOP'],        # Evaluate using last N fitness scores
)

mutations = Mutations(
    algo=INIT_HP['ALGO'],                                 # Algorithm
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
    device=device,
)
```
The easiest training loop implementation is to use our <code>train_off_policy()</code> function. It requires the <code>agent</code> have methods <code>get_action()</code> and <code>learn().</code>
```python
from agilerl.training.train_off_policy import train_off_policy

trained_pop, pop_fitnesses = train_off_policy(
    env=env,                                   # Gym-style environment
    env_name=INIT_HP['ENV_NAME'],              # Environment name
    algo=INIT_HP['ALGO'],                      # Algorithm
    pop=agent_pop,                             # Population of agents
    memory=memory,                             # Replay buffer
    swap_channels=INIT_HP['CHANNELS_LAST'],    # Swap image channel from last to first
    max_steps=INIT_HP["MAX_STEPS"],            # Max number of training steps
    evo_steps=INIT_HP['EVO_STEPS'],            # Evolution frequency
    eval_steps=INIT_HP["EVAL_STEPS"],          # Number of steps in evaluation episode
    eval_loop=INIT_HP["EVAL_LOOP"],            # Number of evaluation episodes
    learning_delay=INIT_HP['LEARNING_DELAY'],  # Steps before starting learning
    target=INIT_HP['TARGET_SCORE'],            # Target score for early stopping
    tournament=tournament,                     # Tournament selection object
    mutation=mutations,                        # Mutations object
    wb=INIT_HP['WANDB'],                       # Weights and Biases tracking
)

```

## Citing AgileRL

If you use AgileRL in your work, please cite the repository:
```bibtex
@software{Ustaran-Anderegg_AgileRL,
author = {Ustaran-Anderegg, Nicholas and Pratt, Michael},
license = {Apache-2.0},
title = {{AgileRL}},
url = {https://github.com/AgileRL/AgileRL}
}
```
