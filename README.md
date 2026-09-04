<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/222710068-e09a4e3c-368c-458a-9e01-b68674806887.png height="120">
</p>
<p align="center"><b>Reinforcement learning streamlined.</b><br>Easier and faster reinforcement learning with RLOps. Visit our <a href="https://agilerl.com">website</a>. View <a href="https://docs.agilerl.com">documentation</a>.<br>Join the <a href="https://discord.gg/eB8HyTA2ux">Discord Server</a> for questions, help and collaboration.</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/agilerl/badge/?version=latest)](https://docs.agilerl.com/en/latest/?badge=latest)
[![Coverage](https://codecov.io/gh/AgileRL/AgileRL/graph/badge.svg)](https://codecov.io/gh/AgileRL/AgileRL)
[![CI](https://github.com/AgileRL/AgileRL/actions/workflows/ci.yml/badge.svg)](https://github.com/AgileRL/AgileRL/actions/workflows/ci.yml)
[![Downloads](https://static.pepy.tech/badge/agilerl)](https://pypi.python.org/pypi/agilerl/)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/eB8HyTA2ux?style=flat)](https://discord.gg/eB8HyTA2ux)
[![Arena](./.github/badges/arena-github-badge.svg)](https://arena.agilerl.com)
<br>
<h3><i>🚀 <b>Train super-fast for free on <a href="https://arena.agilerl.com">Arena</a>, the RLOps platform from AgileRL 🚀</b></i></h3>
</div>
<br>

AgileRL is a Deep Reinforcement Learning library focused on improving development by introducing RLOps - MLOps for reinforcement learning.

This library is initially focused on reducing the time taken for training models and hyperparameter optimization (HPO) by pioneering [evolutionary HPO techniques](https://docs.agilerl.com/en/latest/evo_hyperparam_opt/index.html) for reinforcement learning.<br>
Evolutionary HPO has been shown to drastically reduce overall training times by automatically converging on optimal hyperparameters, without requiring numerous training runs.<br>
We are constantly adding more algorithms and features. AgileRL already includes state-of-the-art evolvable [LLM fine-tuning](https://docs.agilerl.com/en/latest/llm_finetuning/index.html), [on-policy](https://docs.agilerl.com/en/latest/on_policy/index.html), [off-policy](https://docs.agilerl.com/en/latest/off_policy/index.html), [offline](https://docs.agilerl.com/en/latest/offline_training/index.html), [multi-agent](https://docs.agilerl.com/en/latest/multi_agent_training/index.html) and [contextual multi-armed bandit](https://docs.agilerl.com/en/latest/bandits/index.html) reinforcement learning algorithms with [distributed training](https://docs.agilerl.com/en/latest/distributed_training/index.html).

<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/236407686-21363eb3-ffcf-419f-b019-0be4ddf1ed4a.gif width="100%" max-width="900">
</p>
<p align="center">AgileRL offers 10x faster hyperparameter optimization than SOTA.</p>

## Table of Contents
  * [Benchmarks](#benchmarks)
  * [Get Started](#get-started)
  * [Training](#training)
  * [Arena](#arena)
  * [Tutorials](#tutorials)
  * [Algorithms](#evolvable-algorithms-more-coming-soon)
  * [Citing AgileRL](#citing-agilerl)

## Benchmarks

### LLM Fine-tuning

AgileRL's multi-turn LLM training enables state-of-the-art performance on long-horizon tasks with small models. In the following example, AgileRL's CISPO was benchmarked against ART and TRL on the <a href="https://github.com/axon-rl/gem">GEM</a> Sudoku Hard task. This is a difficult multi-turn problem, which requires a context length of 32k tokens and up to 50 turns per rollout. The sync AgileRL run is a single agent using the AgileRL framework. The async and HPO runs were performed on <a href="https://arena.agilerl.com">Arena</a>, AgileRL's RLOps platform. All runs used the same starting hyperparameters. AgileRL runs were run on A100 40GB nodes, whereas ART and TRL required A100 80GB nodes due to a lack of optimizations. AgileRL runs significantly outperformed those using the ART and TRL frameworks.

<p align="center">
  <img src="https://raw.githubusercontent.com/AgileRL/AgileRL/main/docs/_static/multi_turn_llm_benchmarks.png" min-width="100%" width="700">
</p>

### Classic RL

Reinforcement learning algorithms and libraries are usually benchmarked once the optimal hyperparameters for training are known, but it often takes hundreds or thousands of experiments to discover these. This is unrealistic and does not reflect the true, total time taken for training. What if we could remove the need to conduct all these prior experiments?

In the charts below, a single AgileRL run, which automatically tunes hyperparameters, is benchmarked against Optuna's multiple training runs traditionally required for hyperparameter optimization, demonstrating the real time savings possible. Global steps is the sum of every step taken by any agent in the environment, including across an entire population.

<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/227481592-27a9688f-7c0a-4655-ab32-90d659a71c69.png min-width="100%" width="600">
</p>
<p align="center"><small>AgileRL offers an order of magnitude speed up in hyperparameter optimization vs popular reinforcement learning training frameworks combined with Optuna. Remove the need for multiple training runs and save yourself hours.</small></p>

AgileRL also supports multi-agent reinforcement learning using the Petting Zoo-style (parallel API). The charts below highlight the performance of our MADDPG and MATD3 algorithms with evolutionary hyper-parameter optimisation (HPO), benchmarked against epymarl's MADDPG algorithm with grid-search HPO for the simple speaker listener and simple spread environments.

<p align="center">
  <img src=https://github-production-user-asset-6210df.s3.amazonaws.com/118982716/264712154-4965ea5f-b777-423c-989b-e4db86eda3bd.png  min-width="100%" width="700">
</p>

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

AgileRL ships optional dependency groups that you can install as needed:

| Installation | Description |
|-------|--------------|
| `agilerl[box2d]` | Box2D physics engine for Gymnasium environments |
| `agilerl-arena` | [Arena](https://arena.agilerl.com) SDK & CLI. Validate custom environments, and train & deploy agents on managed cloud infrastructure. |
| `agilerl[llm]` | LLM reinforcement fine-tuning. |
| `agilerl[all]` | Cover all functionalities of AgileRL. |

In development mode:
```bash
pip install -e .
```

To install AgileRL from the development tip of ``main`` (not a PyPI release), use:

```bash
pip install git+https://github.com/AgileRL/AgileRL.git@main
```

## Training Locally

AgileRL provides the tools to train RL algorithms in a variety of ways, focusing on flexibility and modularity as a stepping stone for efficiently training
arbitrarily large populations of agents in a distributed manner on Arena.

### Training a Single Agent without Evolutionary HPO

The simplest way to train an RL agent with AgileRL is through the
[`LocalTrainer`](https://docs.agilerl.com/en/latest/trainers/index.html). Here is an example of training a DQN agent on the `LunarLander-v3` environment:

```python
from agilerl import LocalTrainer

trainer = LocalTrainer(algorithm="DQN", environment="LunarLander-v3")
population, fitnesses = trainer.train()
```

> With no other arguments provided, `LocalTrainer` defaults to 1,000,000 steps with a
> single agent and the algorithm's default hyperparameters — no evolutionary
> HPO is applied.

### Training a Population with Evolutionary HPO

To unlock AgileRL's evolutionary hyperparameter optimization, train a population
of agents whose hyperparameters will evolve and mutate towards their optimal
values:

```python
from agilerl import LocalTrainer
from agilerl.models import TrainingSpec

trainer = LocalTrainer(
    algorithm="DQN",
    environment="LunarLander-v3",
    training=TrainingSpec(pop_size=4),  # Train four agents simultaneously
    hpo=True,  # Enable evolutionary HPO using default settings
)
population, fitnesses = trainer.train()
```

This trains a population of four DQN agents that share experiences but learn individually. Every 10,000 steps
(default value for `evo_steps` in `TrainingSpec`), tournament selection identifies the best
performers and mutations are applied to explore the hyperparameter space. See [Evolutionary Hyperparameter Optimization](https://docs.agilerl.com/en/latest/evo_hyperparam_opt/index.html) for details on how evolutionary HPO works in AgileRL.

Or via a YAML manifest:

<details>
<summary>DQN-LunarLander-v3 manifest (<code>configs/training/dqn/dqn.yaml</code>)</summary>

```yaml
---
algorithm:
    name: DQN
    batch_size: 128
    lr: 6.3e-4
    learn_step: 4
    gamma: 0.99
    tau: 0.001
    double: false
    cudagraphs: false

environment:
    name: LunarLander-v3
    num_envs: 16

mutation:
    probabilities:
        no_mut: 0.4
        arch_mut: 0.2
        new_layer: 0.2
        params_mut: 0.2
        act_mut: 0.2
        rl_hp_mut: 0.2
    rl_hp_selection:
        lr:
            min: 0.0000625
            max: 0.01
        batch_size:
            min: 8
            max: 512
        learn_step:
            min: 1
            max: 10
    mutation_sd: 0.1
    rand_seed: 42

network:
    latent_dim: 128
    arch: mlp
    encoder_config:
        hidden_size:
            - 128
    head_config:
        hidden_size:
            - 128

replay_buffer:
    max_size: 100_000

selection_strategy:
    tournament_size: 2
    elitism: true

training:
    max_steps: 1_000_000
    target_score: 200.0
    pop_size: 4
    evo_steps: 10_000
    eval_steps:
    eval_loop: 1
    learning_delay: 0
    eps_start: 1.0
    eps_end: 0.1
    eps_decay: 0.99
```

</details>

**Python**

```python
from agilerl import LocalTrainer

trainer = LocalTrainer.from_manifest("configs/training/dqn/dqn.yaml")
population, fitnesses = trainer.train()
```

**CLI**

```bash
python -m agilerl.train configs/training/dqn/dqn.yaml
```

Every aspect of the training pipeline is customisable — from modifying
hyperparameters and mutation strategies in our off-the-shelf tools, to
implementing your own [evolvable algorithms](https://docs.agilerl.com/en/latest/custom_algorithms/index.html),
[network architectures](https://docs.agilerl.com/en/latest/evolvable_networks/index.html), and
[training loops](https://docs.agilerl.com/en/latest/off_policy/index.html).

### Custom Training Pipelines

For full control over training, you can build each component individually:

<details>
<summary>Custom RL pipeline example</summary>

```python
import torch

from agilerl.algorithms import DQN
from agilerl.utils.utils import make_vect_envs
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_off_policy import train_off_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize environment
env = make_vect_envs(env_name="LunarLander-v3", num_envs=16)

# Network configuration
net_config = {
    "latent_dim": 64,
    "encoder_config": {"hidden_size": [64]},
    "head_config": {"hidden_size": [64]},
}

# Algorithm hyperparameters
init_hp = {
    "double": True,
    "batch_size": 256,
    "lr": 1e-3,
    "gamma": 0.99,
    "learn_step": 1,
    "tau": 1e-3,
}

# Create a population of DQN agents
population_size = 6
agent_pop = DQN.population(
    size=population_size,
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    net_config=net_config,
    device=device,
    **init_hp,
)

# Replay buffer
memory = ReplayBuffer(max_size=10_000, device=device)

# Evolutionary HPO
tournament = TournamentSelection(
    tournament_size=2, elitism=True, population_size=population_size
)
mutations = Mutations(
    no_mutation=0.4,
    architecture=0.2,
    new_layer_prob=0.2,
    parameters=0.2,
    activation=0.0,
    rl_hp=0.2,
    mutation_sd=0.1,
    rand_seed=42,
    device=device,
)

trained_pop, pop_fitnesses = train_off_policy(
    env=env,
    env_name="LunarLander-v3",
    algo="DQN",
    pop=agent_pop,
    memory=memory,
    max_steps=1_000_000,
    evo_steps=10_000,
    target=200.0,
    selection_strategy=tournament,
    mutation=mutations,
)
```

</details>

This approach gives you the flexibility to swap in your own Gymnasium or PettingZoo environments, custom evolvable networks, or entirely custom training loops while still leveraging AgileRL's evolutionary HPO.

## Training on Arena

[Arena](https://arena.agilerl.com) is the RLOps platform from AgileRL. We provide tools to create and validate custom reinforcement learning environments on the platform and train RL agents on managed cloud infrastructure specifically tailored to RL workloads.

AgileRL ships a **Python SDK** and a **CLI** for interacting with the platform through the [`agilerl-arena`](agilerl-arena/README.md) package. It is a **separate PyPI distribution** that contributes the `agilerl.arena` namespace. Install it directly, or via the AgileRL extra:

```bash
pip install agilerl-arena
# or
pip install agilerl
```

### Python

Use the `ArenaClient` to interact with Arena programmatically from scripts or notebooks:

```python
from agilerl.arena import ArenaClient

client = ArenaClient()
client.login()  # OAuth2 device-flow authentication

# Upload and validate a custom environment
client.validate_environment(name="my-custom-env", source="path/to/my_env.py")

# Train on validated custom environment
client.submit_experiment(
    manifest="path/to/manifest.yaml",
    project="my-project",
)
```

### Arena CLI

The same operations are available from the command line:

```bash
# Authenticate with Arena
arena login

# Upload and validate
arena env validate my-custom-env --source path/to/my_env.py

# Train on validated custom environment
arena experiments submit path/to/manifest.yaml --project my-project
```

For the full CLI and Python SDK reference, including authentication, environment validation, experiments, and deployment, see the [Arena Client](https://docs.agilerl.com/en/latest/arena/index.html) documentation.

## Tutorials

We are constantly updating our tutorials to showcase the latest features of AgileRL and how users can leverage our evolutionary HPO to achieve 10x faster hyperparameter optimization. Please see the available tutorials below.

| Tutorial Type | Description | Tutorials |
|---------------|-------------|-----------|
| [Single-agent tasks](https://docs.agilerl.com/en/latest/tutorials/gymnasium/index.html) | Guides for training both on and off-policy agents to beat a variety of Gymnasium environments. | [PPO - Acrobot](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_ppo_tutorial.html) <br> [TD3 - Lunar Lander](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_td3_tutorial.html) <br> [Rainbow DQN - CartPole](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_rainbow_dqn_tutorial.html) <br> [Recurrent PPO - Masked Pendulum](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_recurrent_ppo_tutorial.html)  |
| [Multi-agent tasks](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/index.html) | Use of PettingZoo environments such as training DQN to play Connect Four with curriculum learning and self-play, and for multi-agent tasks in MPE environments. | [DQN - Connect Four](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/dqn.html) <br> [MADDPG - Space Invaders](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/maddpg.html) <br> [MATD3 - Speaker Listener](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/matd3.html) |
| [Hierarchical curriculum learning](https://docs.agilerl.com/en/latest/tutorials/skills/index.html) | Shows how to teach agents Skills and combine them to achieve an end goal. | [PPO - Lunar Lander](https://docs.agilerl.com/en/latest/tutorials/skills/index.html) |
| [Contextual multi-arm bandits](https://docs.agilerl.com/en/latest/tutorials/bandits/index.html) | Learn to make the correct decision in environments that only have one timestep. | [NeuralUCB - Iris Dataset](https://docs.agilerl.com/en/latest/tutorials/bandits/agilerl_neural_ucb_tutorial.html) <br> [NeuralTS - PenDigits](https://docs.agilerl.com/en/latest/tutorials/bandits/agilerl_neural_ts_tutorial.html) |
| [Custom Modules & Networks](https://docs.agilerl.com/en/latest/tutorials/custom_networks/index.html) | Learn how to create custom evolvable modules and networks for RL algorithms. | [Dueling Distributional Q Network](https://docs.agilerl.com/en/latest/tutorials/custom_networks/agilerl_rainbow_tutorial.html) <br> [EvolvableSimBa](https://docs.agilerl.com/en/latest/tutorials/custom_networks/agilerl_simba_tutorial.html) |
| [Training on Arena](https://docs.agilerl.com/en/latest/tutorials/arena_training/index.html) | Upload and validate custom environments, submit training jobs on managed cloud infrastructure, and deploy trained agents for inference. | [PPO - Acrobot Custom Environment](https://docs.agilerl.com/en/latest/tutorials/arena_training/ppo_custom_env.html) |
| [LLM Finetuning](https://docs.agilerl.com/en/latest/tutorials/llm_finetuning/index.html) | Learn how to finetune an LLM using AgileRL. | [GRPO](https://docs.agilerl.com/en/latest/tutorials/llm_finetuning/index.html) |

## Evolvable Algorithms (more coming soon!)

  ### Single-agent

  | RL         | Algorithm |
  | ---------- | --------- |
  | [On-Policy](https://docs.agilerl.com/en/latest/on_policy/index.html)  | [Proximal Policy Optimization (PPO)](https://docs.agilerl.com/en/latest/api/algorithms/ppo.html) |
  | [Off-Policy](https://docs.agilerl.com/en/latest/off_policy/index.html) | [Deep Q Learning (DQN)](https://docs.agilerl.com/en/latest/api/algorithms/dqn.html) <br>  [Rainbow DQN](https://docs.agilerl.com/en/latest/api/algorithms/dqn_rainbow.html) <br> [Deep Deterministic Policy Gradient (DDPG)](https://docs.agilerl.com/en/latest/api/algorithms/ddpg.html) <br> [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://docs.agilerl.com/en/latest/api/algorithms/td3.html) |
  | [Offline](https://docs.agilerl.com/en/latest/offline_training/index.html)    | [Conservative Q-Learning (CQL)](https://docs.agilerl.com/en/latest/api/algorithms/cql.html) <br>  [Implicit Language Q-Learning (ILQL)](https://docs.agilerl.com/en/latest/api/algorithms/ilql.html) |

  ### Multi-agent

  | RL         | Algorithm |
  | ---------- | --------- |
  | [Multi-agent](https://docs.agilerl.com/en/latest/multi_agent_training/index.html) | [Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](https://docs.agilerl.com/en/latest/api/algorithms/maddpg.html) <br> [Multi-Agent Twin-Delayed Deep Deterministic Policy Gradient (MATD3)](https://docs.agilerl.com/en/latest/api/algorithms/matd3.html)  <br> [Independent Proximal Policy Optimization (IPPO)](https://docs.agilerl.com/en/latest/api/algorithms/ippo.html)|

  ### Contextual multi-armed bandit

  | RL         | Algorithm |
  | ---------- | --------- |
  | [Bandits](https://docs.agilerl.com/en/latest/bandits/index.html) | [Neural Contextual Bandits with UCB-based Exploration (NeuralUCB)](https://docs.agilerl.com/en/latest/api/algorithms/neural_ucb.html) <br> [Neural Contextual Bandits with Thompson Sampling (NeuralTS)](https://docs.agilerl.com/en/latest/api/algorithms/neural_ts.html) |

  ### LLM Fine-tuning

  | RL         | Algorithm |
  | ---------- | --------- |
  | [On-Policy](https://docs.agilerl.com/en/latest/llm_finetuning/index.html) | [Group Relative Policy Optimization (GRPO)](https://docs.agilerl.com/en/latest/api/algorithms/grpo.html) <br> [Clipped Importance Sampling Policy Optimization (CISPO)](https://docs.agilerl.com/en/latest/api/algorithms/cispo.html) <br> [Grouped Sequence Policy Optimization (GSPO)](https://docs.agilerl.com/en/latest/api/algorithms/gspo.html) <br> [LLM Proximal Policy Optimization (LLM PPO)](https://docs.agilerl.com/en/latest/api/algorithms/llmppo.html) <br> [LLM REINFORCE](https://docs.agilerl.com/en/latest/api/algorithms/llmreinforce.html) <br>
  | [Off-Policy](https://docs.agilerl.com/en/latest/llm_finetuning/index.html) | [Direct Preference Optimization (DPO)](https://docs.agilerl.com/en/latest/api/algorithms/dpo.html)

## Citing AgileRL

If you use AgileRL in your work, please cite the repository:
```bibtex
@software{Ustaran-Anderegg_AgileRL,
author = {Ustaran-Anderegg, Nicholas and Pratt, Michael and Sabal-Bermudez, Jaime},
license = {Apache-2.0},
title = {{AgileRL}},
url = {https://github.com/AgileRL/AgileRL}
}
```
