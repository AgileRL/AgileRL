<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/222710068-e09a4e3c-368c-458a-9e01-b68674806887.png height="120">
</p>
<p align="center"><b>The complete post-training stack for LLMs + RL for any problem</b><br>Visit our <a href="https://agilerl.com">website</a>. View <a href="https://docs.agilerl.com">documentation</a>.<br>Join the <a href="https://discord.gg/eB8HyTA2ux">Discord Server</a> for questions, help and collaboration.</p>

<div align="center">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/agilerl/badge/?version=latest)](https://docs.agilerl.com/en/latest/?badge=latest)
[![Coverage](https://codecov.io/gh/AgileRL/AgileRL/graph/badge.svg)](https://codecov.io/gh/AgileRL/AgileRL)
[![CI](https://github.com/AgileRL/AgileRL/actions/workflows/ci-status.yml/badge.svg?branch=main)](https://github.com/AgileRL/AgileRL/actions/workflows/ci-status.yml?query=branch%3Amain)
[![Downloads](https://static.pepy.tech/badge/agilerl)](https://pypi.python.org/pypi/agilerl/)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/eB8HyTA2ux?style=flat)](https://discord.gg/eB8HyTA2ux)
[![Arena](./.github/badges/arena-github-badge.svg)](https://arena.agilerl.com)
<br>
<h3><i>🚀 <b>Train and deploy at scale with <a href="https://arena.agilerl.com">Arena</a> from AgileRL 🚀</b></i></h3>
</div>
<br>

AgileRL takes an open-weights LLM from supervised fine-tuning through preference tuning to multi-turn agentic RL, in one library. The same library covers classic deep RL (single-agent, multi-agent, offline and bandits) with evolutionary hyperparameter optimization built in.

## Table of Contents
  * [LLM Post-Training](#llm-post-training)
  * [Get Started](#get-started)
  * [Train at Scale on Arena](#train-at-scale-on-arena)
  * [Train an LLM Locally](#train-an-llm-locally)
  * [Classic RL](#classic-rl)
  * [Tutorials](#tutorials)
  * [Algorithms](#algorithms)
  * [Citing AgileRL](#citing-agilerl)

## LLM Post-Training

- **The whole pipeline.** [SFT](https://docs.agilerl.com/en/latest/api/algorithms/sft.html) and [DPO](https://docs.agilerl.com/en/latest/api/algorithms/dpo.html), then RL with [GRPO](https://docs.agilerl.com/en/latest/api/algorithms/grpo.html), [CISPO](https://docs.agilerl.com/en/latest/api/algorithms/cispo.html), [GSPO](https://docs.agilerl.com/en/latest/api/algorithms/gspo.html), [REINFORCE](https://docs.agilerl.com/en/latest/api/algorithms/llmreinforce.html) or [PPO](https://docs.agilerl.com/en/latest/api/algorithms/llmppo.html).
- **Multi-turn agentic RL.** Train on any [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment: a Python class, a library entrypoint such as [GEM](https://github.com/axon-rl/gem), or a remote env server. Or use a dataset plus a reward function. See [Environments](https://docs.agilerl.com/en/latest/llm_finetuning/environments.html).
- **LoRA.** Train small adapters on a frozen base. For RL post-training, [LoRA can match full fine-tuning](https://thinkingmachines.ai/blog/lora/) at a fraction of the memory. [QLoRA](https://docs.agilerl.com/en/latest/llm_finetuning/quantization.html) squeezes in bigger models for colocated local runs.
- **Torch-native parallelism.** Data parallel or FSDP2 sharding, launched with `torchrun`. One `FSDPConfig` works for every LLM algorithm. See [Multi-GPU LLM training](https://docs.agilerl.com/en/latest/llm_finetuning/distributed.html).
- **Memory optimizations.** Chunked fused log-probs, activation checkpointing and offload, CPU offload for optimizer state or parameters, and trainer offload during rollout.
- **Model-specific optimizations.** Per-family kernels and settings for high throughput at the lowest memory, on dense, MoE and hybrid Mamba models. MoE LoRA runs per expert without materializing full tensors, and hybrid models like Nemotron-H get fused kernels and fixes for Mamba2.
- **Fast rollouts.** vLLM generation, with the option to colocate the trainer and vLLM on a single GPU.
- **Evolutionary HPO for LLMs.** Train a population and let learning rate, KL penalty and more tune themselves during the run.
- **Agent-friendly CLI.** A training run is one YAML manifest and one command. Your coding agent can read the manifest schema, validate its own config and launch runs on Arena without a human in the loop. See [Built for coding agents](#built-for-coding-agents).
- **Async training at scale.** Move to [Arena](#train-at-scale-on-arena) for async training on managed GPU clusters, with models up to 120B such as Nemotron 3.5 Super VL. We keep adding models.
- **Deploy and chat.** Deploy the best checkpoint on Arena and chat with it from the CLI with `arena agent generate`.

### Benchmark

**AgileRL trains on over 4x more tokens/s than ART and TRL, and reaches higher reward on half the GPU memory.**

AgileRL's CISPO against ART and TRL on the [GEM](https://github.com/axon-rl/gem) Sudoku Hard task: 32k-token context, up to 50 turns per rollout. The async and HPO runs were on [Arena](https://arena.agilerl.com). AgileRL ran on A100 40GB nodes. ART and TRL needed A100 80GB. All runs used the same starting hyperparameters.

<p align="center">
  <img src="https://raw.githubusercontent.com/AgileRL/AgileRL/main/docs/_static/multi_turn_llm_benchmarks.png" min-width="100%" width="700">
</p>

## Get Started

```bash
pip install "agilerl[llm]"        # LLM post-training
pip install agilerl               # classic RL + the Arena CLI
```

| Installation | Description |
|-------|--------------|
| `agilerl[llm]` | Hugging Face transformers, PEFT, datasets, Liger, bitsandbytes, and vLLM (Linux). |
| `agilerl[cpu-llm]` | Same Hugging Face stack as `[llm]`, without vLLM. Use this when you do not need the vLLM engine (for example with `[cpu]`). |
| `agilerl[box2d]` | Box2D physics engine for Gymnasium environments. |
| `agilerl[cpu]` | CPU-only PyTorch wheels (no NVIDIA stack). |
| `agilerl[all]` | Box2D and LLM extras. |
| `agilerl-arena` | [Arena](https://arena.agilerl.com) SDK and CLI only, no torch. Included with `agilerl`. |

For the development tip of `main`: `pip install git+https://github.com/AgileRL/AgileRL.git@main`. For development mode: clone the repo and run `pip install -e .`.

## Train at Scale on Arena

Every training run is a YAML manifest. [Arena](https://arena.agilerl.com) runs it on managed GPU clusters, with async rollouts and population-based HPO across nodes. Supported base models range from Qwen, Granite and Gemma up to 100B+ models such as Nemotron 3.5 Super VL 120B-A12B, and the list keeps growing. The `arena` CLI ships with `agilerl`:

```bash
arena login
arena models list                                   # supported base models
arena experiments submit my_manifest.yaml --project my-project
arena agent deploy my-experiment                    # deploy the best checkpoint
arena agent run my-deployment
arena agent generate --prompt "What is 17 * 23?"
```

### Built for coding agents

Point Cursor, Claude Code or Codex at the `arena` CLI and it can run the whole loop itself: write a manifest, check it, launch it, pull metrics and deploy.

```bash
export ARENA_API_KEY="arena_pat_..."                # no interactive login
arena manifest schema                               # JSON Schema the agent writes against
arena manifest validate my_manifest.yaml --json     # one JSON verdict, stable exit codes
arena experiments submit my_manifest.yaml --project my-project
arena experiments metrics my-experiment             # download metrics to judge the run
```

Upload your own data with `arena datasets create`, or your own environment with `arena env validate`. The same operations are on `ArenaClient` in Python. See the [Arena docs](https://docs.agilerl.com/en/latest/arena/index.html) and the [`agilerl-arena` README](agilerl-arena/README.md).

## Train an LLM Locally

The same manifests run on your own GPU. This one trains Qwen2.5-0.5B with GRPO on GEM's multi-turn Guess the Number game:

```bash
pip install gem-llm
python -m agilerl.train configs/training/llm_finetuning/grpo_env.yaml
```

Or from Python:

```python
from agilerl import LocalTrainer

trainer = LocalTrainer.from_manifest("configs/training/llm_finetuning/grpo_env.yaml")
population, fitnesses = trainer.train()
```

More manifests for SFT, DPO, CISPO, GSPO, LLM PPO and LLM REINFORCE are in [`configs/training/llm_finetuning`](configs/training/llm_finetuning).

## Classic RL

<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/236407686-21363eb3-ffcf-419f-b019-0be4ddf1ed4a.gif width="100%" max-width="900">
</p>

On-policy, off-policy, offline, multi-agent and contextual bandit algorithms, all with [evolutionary HPO](https://docs.agilerl.com/en/latest/evo_hyperparam_opt/index.html). Train a population of agents and the hyperparameters evolve during a single run, instead of running hundreds of trials first.

```python
from agilerl import LocalTrainer
from agilerl.models import TrainingSpec

trainer = LocalTrainer(
    algorithm="DQN",
    environment="LunarLander-v3",
    training=TrainingSpec(pop_size=4),
    hpo=True,
)
population, fitnesses = trainer.train()
```

Or from a manifest: `python -m agilerl.train configs/training/dqn/dqn.yaml`. You can swap in your own Gymnasium or PettingZoo environments, [evolvable networks](https://docs.agilerl.com/en/latest/evolvable_networks/index.html), [algorithms](https://docs.agilerl.com/en/latest/custom_algorithms/index.html) or [training loops](https://docs.agilerl.com/en/latest/off_policy/index.html). See [Trainers](https://docs.agilerl.com/en/latest/trainers/index.html).

A single AgileRL run with evolutionary HPO against the multiple Optuna runs needed to tune other frameworks. Global steps counts every step taken by every agent in the population:

<p align="center">
  <img src=https://user-images.githubusercontent.com/47857277/227481592-27a9688f-7c0a-4655-ab32-90d659a71c69.png min-width="100%" width="600">
</p>

## Tutorials

| Tutorial Type | Description | Tutorials |
|---------------|-------------|-----------|
| [LLM Fine-tuning](https://docs.agilerl.com/en/latest/tutorials/llm_finetuning/index.html) | SFT, DPO, single- and multi-turn RL, HPO and remote environments. | [GRPO reasoning](https://docs.agilerl.com/en/latest/tutorials/llm_finetuning/grpo_finetuning.html) <br> [SFT and DPO](https://docs.agilerl.com/en/latest/tutorials/llm_finetuning/sft_dpo_finetuning.html) <br> [GRPO with HPO](https://docs.agilerl.com/en/latest/tutorials/llm_finetuning/grpo_hpo.html) <br> [Multi-turn GRPO and PPO](https://docs.agilerl.com/en/latest/tutorials/llm_finetuning/env_grpo_ppo.html) <br> [Remote env server](https://docs.agilerl.com/en/latest/tutorials/llm_finetuning/remote_env_server.html) |
| [Training on Arena](https://docs.agilerl.com/en/latest/tutorials/arena_training/index.html) | Upload and validate custom environments, submit training jobs on managed cloud infrastructure, and deploy trained agents for inference. | [PPO - Acrobot Custom Environment](https://docs.agilerl.com/en/latest/tutorials/arena_training/ppo_custom_env.html) |
| [Single-agent tasks](https://docs.agilerl.com/en/latest/tutorials/gymnasium/index.html) | Train on- and off-policy agents on Gymnasium environments. | [PPO - Acrobot](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_ppo_tutorial.html) <br> [TD3 - Lunar Lander](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_td3_tutorial.html) <br> [Rainbow DQN - CartPole](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_rainbow_dqn_tutorial.html) <br> [Recurrent PPO - Masked Pendulum](https://docs.agilerl.com/en/latest/tutorials/gymnasium/agilerl_recurrent_ppo_tutorial.html)  |
| [Multi-agent tasks](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/index.html) | PettingZoo environments, including Connect Four with curriculum learning and self-play. | [DQN - Connect Four](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/dqn.html) <br> [MADDPG - Space Invaders](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/maddpg.html) <br> [MATD3 - Speaker Listener](https://docs.agilerl.com/en/latest/tutorials/pettingzoo/matd3.html) |
| [Hierarchical curriculum learning](https://docs.agilerl.com/en/latest/tutorials/skills/index.html) | Teach agents skills and combine them to reach an end goal. | [PPO - Lunar Lander](https://docs.agilerl.com/en/latest/tutorials/skills/index.html) |
| [Contextual multi-arm bandits](https://docs.agilerl.com/en/latest/tutorials/bandits/index.html) | Make the right decision in single-timestep environments. | [NeuralUCB - Iris Dataset](https://docs.agilerl.com/en/latest/tutorials/bandits/agilerl_neural_ucb_tutorial.html) <br> [NeuralTS - PenDigits](https://docs.agilerl.com/en/latest/tutorials/bandits/agilerl_neural_ts_tutorial.html) |
| [Custom Modules & Networks](https://docs.agilerl.com/en/latest/tutorials/custom_networks/index.html) | Build custom evolvable modules and networks. | [Dueling Distributional Q Network](https://docs.agilerl.com/en/latest/tutorials/custom_networks/agilerl_rainbow_tutorial.html) <br> [EvolvableSimBa](https://docs.agilerl.com/en/latest/tutorials/custom_networks/agilerl_simba_tutorial.html) |

## Algorithms

### LLM Post-Training

| Type | Algorithm |
| ---------- | --------- |
| [RL](https://docs.agilerl.com/en/latest/llm_finetuning/index.html) | [Group Relative Policy Optimization (GRPO)](https://docs.agilerl.com/en/latest/api/algorithms/grpo.html) <br> [Clipped Importance Sampling Policy Optimization (CISPO)](https://docs.agilerl.com/en/latest/api/algorithms/cispo.html) <br> [Group Sequence Policy Optimization (GSPO)](https://docs.agilerl.com/en/latest/api/algorithms/gspo.html) <br> [LLM Proximal Policy Optimization (LLM PPO)](https://docs.agilerl.com/en/latest/api/algorithms/llmppo.html) <br> [LLM REINFORCE](https://docs.agilerl.com/en/latest/api/algorithms/llmreinforce.html) |
| [Preference](https://docs.agilerl.com/en/latest/llm_finetuning/index.html) | [Direct Preference Optimization (DPO)](https://docs.agilerl.com/en/latest/api/algorithms/dpo.html) |
| [Supervised](https://docs.agilerl.com/en/latest/llm_finetuning/index.html) | [Supervised Fine-Tuning (SFT)](https://docs.agilerl.com/en/latest/api/algorithms/sft.html) |

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

## Citing AgileRL

If you use AgileRL in your work, please cite the repository:
```bibtex
@software{Ustaran-Anderegg_AgileRL,
author = {Ustaran-Anderegg, Nicholas and Pratt, Michael and Sabal-Bermudez, Jaime and Doherty, Michael},
license = {Apache-2.0},
title = {{AgileRL}},
url = {https://github.com/AgileRL/AgileRL}
}
```
