# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

import torch
from accelerate import Accelerator
from tensordict import TensorDict
from torch.utils.data import DataLoader

from agilerl.algorithms import NeuralTS, NeuralUCB
from agilerl.components.data import ReplayDataset
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.sampler import Sampler
from agilerl.protocols import BanditEnvProtocol
from agilerl.training.configs import BanditTrainRunConfig
from agilerl.training.loop import (
    TrainSession,
    close_training,
    maybe_checkpoint,
    maybe_evolve,
    report_after_eval,
    resolve_train_evolution,
    start_train_session,
    stop_if_target,
    validate_train_run,
)
from agilerl.typing import InitHyperparams
from agilerl.utils.constructor_kwargs import accept_flat_kwargs

PopulationType = list[NeuralTS | NeuralUCB]

logger = logging.getLogger(__name__)


@dataclass
class BanditSession:
    """Replay sampler and bandit env for one training run."""

    train: TrainSession
    env: BanditEnvProtocol
    memory: ReplayBuffer
    sampler: Sampler
    evo_count: int = 0


def _bandit_sampler(
    pop: PopulationType,
    memory: ReplayBuffer,
    accelerator: Accelerator | None,
) -> Sampler:
    """Build a replay sampler, wrapping the buffer in a dataloader when distributed."""
    if accelerator is None:
        return Sampler(memory=memory)
    replay_dataset = ReplayDataset(memory, pop[0].batch_size)
    replay_dataloader = accelerator.prepare(DataLoader(replay_dataset, batch_size=None))
    return Sampler(dataset=replay_dataset, dataloader=replay_dataloader)


def _run_bandit_episode(
    session: BanditSession,
    agent: NeuralTS | NeuralUCB,
    context: object,
) -> object:
    """Play one bandit episode, storing transitions and learning from replay."""
    episode_steps = session.train.run.loop.episode_steps
    score = 0.0
    sample = session.sampler.sample
    for _ in range(episode_steps):
        action = agent.get_action(context)
        next_context, reward = session.env.step(action)
        transition = TensorDict(
            {
                "obs": torch.as_tensor(context[action]),
                "reward": torch.as_tensor(reward),
            },
        )
        transition = transition.unsqueeze(0)
        transition.batch_size = torch.Size([1])
        session.memory.add(transition)
        if len(session.memory) >= agent.batch_size:
            for _ in range(agent.learn_step):
                agent.learn(sample(agent.batch_size))
        score += reward
        agent.regret.append(agent.regret[-1] + 1 - reward)
        context = next_context
    agent.add_scores([score])
    agent.finalize_training_step(episode_steps)
    session.train.pbar.update(episode_steps // session.train.population.size)
    return context


def _run_bandit_generation(session: BanditSession) -> None:
    """Play one episode per agent in the population."""
    if session.train.accelerator is not None:
        session.train.accelerator.wait_for_everyone()
    for agent in session.train.population.agents:
        agent.set_training_mode(True)
        agent.init_training_step(session.train.capture_grama)
        _run_bandit_episode(session, agent, session.env.reset())


def _evaluate_bandits(session: BanditSession) -> None:
    """Run test episodes on the bandit environment."""
    loop = session.train.run.loop
    for agent in session.train.population.agents:
        agent.test(session.env, max_steps=loop.eval_steps, loop=loop.eval_loop)


def _maybe_evolve_bandits(session: BanditSession) -> None:
    """Evolve only when the evo_steps cadence elapses."""
    evo_steps = session.train.run.loop.evo_steps
    if session.train.population.local_step // evo_steps <= session.evo_count:
        return
    maybe_evolve(session.train)
    session.evo_count += 1


@accept_flat_kwargs
def train_bandits(
    env: BanditEnvProtocol,
    env_name: str,
    algo: str,
    pop: PopulationType,
    memory: ReplayBuffer,
    run: BanditTrainRunConfig | None = None,
    init_hp: InitHyperparams = None,
    accelerator: Accelerator | None = None,
) -> tuple[PopulationType, list[float]]:
    """Train a bandit population; returns agents and fitnesses.

    :param env: Bandit environment to train in.
    :type env: BanditEnvProtocol
    :param env_name: Environment name used in logs and checkpoint paths.
    :type env_name: str
    :param algo: Algorithm name.
    :type algo: str
    :param pop: Population of neural bandit agents.
    :type pop: list
    :param memory: Replay buffer.
    :type memory: ReplayBuffer
    :param run: Loop, evolution, checkpoint, and logging settings.
    :type run: BanditTrainRunConfig, optional
    :param init_hp: Initial hyperparameters logged to W&B.
    :type init_hp: dict, optional
    :param accelerator: Accelerator for distributed training.
    :type accelerator: Accelerator, optional
    :return: Trained population and last fitnesses.
    :rtype: tuple[list, list[float]]
    """
    run = resolve_train_evolution(run or BanditTrainRunConfig())
    validate_train_run(run, algo=algo)
    train = start_train_session(
        pop,
        run,
        algo=algo,
        env_name=env_name,
        accelerator=accelerator,
        init_hp=init_hp,
    )
    session = BanditSession(
        train=train,
        env=env,
        memory=memory,
        sampler=_bandit_sampler(pop, memory, accelerator),
    )
    while train.population.all_below(run.loop.max_steps):
        _run_bandit_generation(session)
        _evaluate_bandits(session)
        report_after_eval(train)
        if stop_if_target(train):
            return train.population.agents, train.population.last_scalar_fitnesses
        _maybe_evolve_bandits(session)
        maybe_checkpoint(train)
    close_training(train)
    return train.population.agents, train.population.last_scalar_fitnesses
