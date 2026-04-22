<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/222710068-e09a4e3c-368c-458a-9e01-b68674806887.png height="120">
</p>
<p align="center"><b>Reinforcement learning streamlined.</b><br>Easier and faster reinforcement learning with RLOps. Visit our <a href="https://agilerl.com">website</a>. View <a href="https://docs.agilerl.com">documentation</a>.<br>Join the <a href="https://discord.gg/eB8HyTA2ux">Discord Server</a> for questions, help and collaboration.</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/agilerl/badge/?version=latest)](https://docs.agilerl.com/en/latest/?badge=latest)
[![Coverage](https://codecov.io/gh/AgileRL/AgileRL/graph/badge.svg)](https://codecov.io/gh/AgileRL/AgileRL)
[![Linux](https://github.com/AgileRL/AgileRL/actions/workflows/linux-tests.yml/badge.svg)](https://github.com/AgileRL/AgileRL/actions/workflows/linux-tests.yml)
[![macOS](https://github.com/AgileRL/AgileRL/actions/workflows/macos-tests.yml/badge.svg)](https://github.com/AgileRL/AgileRL/actions/workflows/macos-tests.yml)
[![Windows](https://github.com/AgileRL/AgileRL/actions/workflows/windows-tests.yml/badge.svg)](https://github.com/AgileRL/AgileRL/actions/workflows/windows-tests.yml)
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
We are constantly adding more algorithms and features. AgileRL already includes state-of-the-art evolvable [on-policy](https://docs.agilerl.com/en/latest/on_policy/index.html), [off-policy](https://docs.agilerl.com/en/latest/off_policy/index.html), [offline](https://docs.agilerl.com/en/latest/offline_training/index.html), [multi-agent](https://docs.agilerl.com/en/latest/multi_agent_training/index.html) and [contextual multi-armed bandit](https://docs.agilerl.com/en/latest/bandits/index.html) reinforcement learning algorithms with [distributed training](https://docs.agilerl.com/en/latest/distributed_training/index.html).

<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/236407686-21363eb3-ffcf-419f-b019-0be4ddf1ed4a.gif width="100%" max-width="900">
</p>
<p align="center">AgileRL offers 10x faster hyperparameter optimization than SOTA.</p>

## Table of Contents
  * [Get Started](#get-started)
  * [Benchmarks](#benchmarks)
  * [Arena](#arena)
  * [Tutorials](#tutorials)
  * [Algorithms implemented](#evolvable-algorithms-more-coming-soon)
  * [Training](#training)
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

AgileRL ships optional dependency groups that you can install as needed:

| Extra | Install | Description |
|-------|---------|-------------|
| `arena` | `pip install agilerl[arena]` | [Arena](https://arena.agilerl.com) SDK & CLI — validate, profile, and train with custom environments on Arena. |
| `llm` | `pip install agilerl[llm]` | LLM reinforcement fine-tuning (GRPO, DPO) via DeepSpeed, vLLM, Transformers, and PEFT. |
| `all` | `pip install agilerl[all]` | Everything above. |

In development mode, quote the extras:
```bash
pip install -e ".[arena]"
```

To install the ``nightly`` version of AgileRL with the latest features, use:

```bash
pip install git+https://github.com/AgileRL/AgileRL.git@nightly
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

## Arena

[Arena](https://arena.agilerl.com) is the RLOps platform from AgileRL. We provide tools to create and validate custom reinforcement learning environments on the platform and train RL agents on managed cloud infrastructure — no cluster setup required. See the documentation

AgileRL ships an **Arena SDK** (Python client) and an **Arena CLI** for interacting with the platform. Install them with:

```bash
pip install agilerl[arena]
```

### Arena SDK

Use the `ArenaClient` to interact with Arena programmatically from scripts or notebooks:

```python
from agilerl.arena import ArenaClient

client = ArenaClient()
client.login()

# Register and validate a custom environment
client.validate_environment(
    name="my-env",
    source="path/to/my_env/",          # directory or .tar.gz

    entrypoint="my_env:MyCustomEnv",
)

# Train on validated custom environment
client.submit_training_job(
  manifest=
)
```

### Arena CLI

The same operations are available from the command line:

```bash
# Authenticate with Arena
arena login

# Upload and validate
arena env validate
    --name my-env \
    --source path/to/my_env/ \
    --entrypoint my_env:MyCustomEnv

# Train on validated custom environment
arena train --manifest path/to/manifest
```

## Tutorials

We are constantly updating our tutorials to showcase the latest features of AgileRL and how users can leverage our evolutionary HPO to achieve 10x faster hyperparameter optimization. Please see the available tutorials below.

| Tutorial Type | Description | Tutorials |
|---------------|-------------|-----------|
| [Single-agent tasks](https://docs.agilerl.com/en/latest/tutorials/gymnasium/index.html) | Guides for training both on and off-policy agents to beat a variety of Gymnasium environments. | [PPO - Acrobot](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_ppo_tutorial.html) <br> [TD3 - Lunar Lander](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_td3_tutorial.html) <br> [Rainbow DQN - CartPole](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_rainbow_dqn_tutorial.html) <br> [Recurrent PPO - Masked Pendulum](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_recurrent_ppo_tutorial.html)  |
| [Multi-agent tasks](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/index.html) | Use of PettingZoo environments such as training DQN to play Connect Four with curriculum learning and self-play, and for multi-agent tasks in MPE environments. | [DQN - Connect Four](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/dqn.html) <br> [MADDPG - Space Invaders](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/maddpg.html) <br> [MATD3 - Speaker Listener](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/matd3.html) |
| [Hierarchical curriculum learning](https://docs.agilerl.com/en/latest/tutorials/skills/index.html) | Shows how to teach agents Skills and combine them to achieve an end goal. | [PPO - Lunar Lander](https://docs.agilerl.com/en/latest/tutorials/skills/index.html) |
| [Contextual multi-arm bandits](https://docs.agilerl.com/en/latest/tutorials/bandits/index.html) | Learn to make the correct decision in environments that only have one timestep. | [NeuralUCB - Iris Dataset](https://docs.agilerl.com/en/latest/tutorials/bandits/agilerl_neural_ucb_tutorial.html) <br> [NeuralTS - PenDigits](https://docs.agilerl.com/en/latest/tutorials/bandits/agilerl_neural_ts_tutorial.html) |
| [Custom Modules & Networks](https://docs.agilerl.com/en/latest/tutorials/custom_networks/index.html) | Learn how to create custom evolvable modules and networks for RL algorithms. | [Dueling Distributional Q Network](https://docs.agilerl.com/en/latest/tutorials/custom_networks/agilerl_rainbow_tutorial.html) <br> [EvolvableSimBa](https://docs.agilerl.com/en/latest/tutorials/custom_networks/agilerl_simba_tutorial.html) |
| [LLM Finetuning](https://docs.agilerl.com/en/latest/tutorials/llm_finetuning/index.html) | Learn how to finetune an LLM using AgileRL. | [GRPO](https://docs.agilerl.com/en/latest/tutorials/llm_finetuning/index.html) |

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
  | [Multi-agent](https://docs.agilerl.com/en/latest/multi_agent_training/index.html) | [Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](https://docs.agilerl.com/en/latest/api/algorithms/maddpg.html) <br> [Multi-Agent Twin-Delayed Deep Deterministic Policy Gradient (MATD3)](https://docs.agilerl.com/en/latest/api/algorithms/matd3.html)  <br> [Independent Proximal Policy Optimization (IPPO)](https://docs.agilerl.com/en/latest/api/algorithms/ippo.html)|

  ### Contextual multi-armed bandit algorithms

  | RL         | Algorithm |
  | ---------- | --------- |
  | [Bandits](https://docs.agilerl.com/en/latest/bandits/index.html) | [Neural Contextual Bandits with UCB-based Exploration (NeuralUCB)](https://docs.agilerl.com/en/latest/api/algorithms/neural_ucb.html) <br> [Neural Contextual Bandits with Thompson Sampling (NeuralTS)](https://docs.agilerl.com/en/latest/api/algorithms/neural_ts.html) |

  ### LLM Fine-tuning Algorithms

  | RL         | Algorithm |
  | ---------- | --------- |
  | [On-Policy](https://docs.agilerl.com/en/latest/llm_finetuning/index.html) | [Group Relative Policy Optimization (GRPO)](https://docs.agilerl.com/en/latest/api/algorithms/grpo.html)
  | [Off-Policy](https://docs.agilerl.com/en/latest/llm_finetuning/index.html) | [Direct Preference Optimization (DPO)](https://docs.agilerl.com/en/latest/api/algorithms/dpo.html)


## Training

AgileRL provides manifest-driven trainers that handle environment creation, population management, evolutionary HPO, and training loop dispatch from a single YAML file.

### Local Training with a Manifest

Define your entire experiment — algorithm, environment, network architecture, mutation probabilities, and training loop — in one YAML manifest:

<details>
<summary>Example manifest (<code>configs/training/ppo/ppo.yaml</code>)</summary>

```yaml
algorithm:
    name: PPO
    batch_size: 128
    lr: 0.001
    learn_step: 2048
    gamma: 0.99
    gae_lambda: 0.95
    action_std_init: 0.6
    clip_coef: 0.2
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5
    update_epochs: 4

environment:
    name: LunarLander-v3
    num_envs: 16

network:
    latent_dim: 64
    arch: mlp
    encoder_config:
        hidden_size: [64]
        activation: ReLU
    head_config:
        hidden_size: [64]
        activation: ReLU

mutation:
    probabilities:
        no_mut: 0.4
        arch_mut: 0.2
        new_layer: 0.2
        params_mut: 0.2
        act_mut: 0.0
        rl_hp_mut: 0.2
    mutation_sd: 0.1
    rand_seed: 42

tournament_selection:
    tournament_size: 2
    elitism: true

training:
    max_steps: 6_000_000
    target_score: 250.0
    pop_size: 4
    evo_steps: 10_240
    eval_loop: 1
```

</details>

Then train with two lines of Python:

```python
from agilerl.training.trainer import LocalTrainer

trainer = LocalTrainer.from_manifest("configs/training/ppo/ppo.yaml", device="cuda")
trainer.train()
```

`LocalTrainer.from_manifest` parses the YAML, builds the environment, creates the population, and configures evolutionary HPO before training — all validated through Pydantic models.

### Training on Arena

To run the same experiment on [Arena](https://arena.agilerl.com)'s managed infrastructure, use the `ArenaTrainer`. The manifest is identical except the `environment` section references an environment that has already been [validated on Arena](#arena):

```python
from agilerl.training.trainer import ArenaTrainer

trainer = ArenaTrainer.from_manifest("configs/training/ppo/ppo.yaml")
trainer.train()
```

The `ArenaTrainer` builds a training manifest and submits it as a job to Arena. It handles authentication, manifest validation, and job submission automatically.

### Custom Training Pipelines

For full control — custom environments, network architectures, or training loops — you can build each component individually:

<details>
<summary>Custom pipeline example</summary>

```python
import torch
from agilerl.utils.utils import make_vect_envs, create_population
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.hpo.tournament import TournamentSelection
from agilerl.hpo.mutation import Mutations
from agilerl.training.train_off_policy import train_off_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = make_vect_envs(env_name="LunarLander-v3", num_envs=16)

agent_pop = create_population(
    algo="DQN",
    observation_space=env.single_observation_space,
    action_space=env.single_action_space,
    net_config={"latent_dim": 16, "encoder_config": {"hidden_size": [32]}, "head_config": {"hidden_size": [32]}},
    INIT_HP={"DOUBLE": True, "BATCH_SIZE": 256, "LR": 1e-3, "GAMMA": 0.99, "LEARN_STEP": 1, "TAU": 1e-3},
    population_size=6,
    num_envs=16,
    device=device,
)

memory = ReplayBuffer(max_size=10_000, device=device)
tournament = TournamentSelection(tournament_size=2, elitism=True, population_size=6)
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
    tournament=tournament,
    mutation=mutations,
)
```

</details>

This approach gives you the flexibility to swap in your own Gymnasium or PettingZoo environments, custom evolvable networks, or entirely custom training loops while still leveraging AgileRL's evolutionary HPO.

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
