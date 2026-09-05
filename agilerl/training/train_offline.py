# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

import gymnasium as gym
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from agilerl.algorithms import CQN
from agilerl.components.data import (
    ReplayDataset,
    Transition,
    transition_to_tensordict,
)
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.sampler import Sampler
from agilerl.training.configs import OfflineDatasetConfig, TrainRunConfig
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
from agilerl.utils.minari_utils import minari_to_agile_buffer

PopulationType = list[CQN]

logger = logging.getLogger(__name__)


@dataclass
class OfflineSession:
    """Replay sampler and eval env for one offline training run."""

    train: TrainSession
    env: gym.vector.VectorEnv
    sampler: Sampler


def _fill_offline_memory(
    memory: ReplayBuffer,
    data: OfflineDatasetConfig,
    accelerator: Accelerator | None,
) -> ReplayBuffer:
    """Load a Minari id or a transition dict into *memory*."""
    if accelerator is not None:
        if accelerator.is_main_process:
            logger.info("Filling replay buffer with dataset...")
        accelerator.wait_for_everyone()
    else:
        logger.info("Filling replay buffer with dataset...")
    if data.minari_dataset_id:
        return minari_to_agile_buffer(
            data.minari_dataset_id,
            memory,
            accelerator,
            data.remote,
        )
    if data.dataset is None:
        msg = "Either 'minari_dataset_id' or 'dataset' must be provided for offline training."
        raise ValueError(msg)
    dataset = data.dataset
    dataset_length = dataset["rewards"].shape[0]
    for i in range(dataset_length - 1):
        transition = transition_to_tensordict(
            Transition(
                obs=dataset["observations"][i],
                action=dataset["actions"][i],
                reward=dataset["rewards"][i],
                next_obs=dataset["observations"][i + 1],
                done=bool(dataset["terminals"][i]),
            )
        ).unsqueeze(0)
        transition.batch_size = torch.Size([1])
        memory.add(transition)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    return memory


def _offline_sampler(
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


def _learn_offline_generation(session: OfflineSession) -> None:
    """Sample replay and learn for one evolution window per agent."""
    if session.train.accelerator is not None:
        session.train.accelerator.wait_for_everyone()
    evo_steps = session.train.run.loop.evo_steps
    for agent in session.train.population.agents:
        agent.set_training_mode(True)
        agent.init_training_step(session.train.capture_grama)
        for _ in range(evo_steps):
            agent.learn(session.sampler.sample(agent.batch_size))
        agent.finalize_training_step(evo_steps)
        session.train.pbar.update(evo_steps // session.train.population.size)


def _evaluate_offline(session: OfflineSession) -> None:
    """Run test episodes in the evaluation environment."""
    loop = session.train.run.loop
    for agent in session.train.population.agents:
        agent.test(session.env, max_steps=loop.eval_steps, loop=loop.eval_loop)


@accept_flat_kwargs
def train_offline(
    env: gym.vector.VectorEnv,
    env_name: str,
    algo: str,
    pop: PopulationType,
    memory: ReplayBuffer,
    run: TrainRunConfig | None = None,
    data: OfflineDatasetConfig | None = None,
    init_hp: InitHyperparams = None,
    accelerator: Accelerator | None = None,
) -> tuple[PopulationType, list[float]]:
    """Train an offline population; returns agents and fitnesses.

    :param env: Vectorized environment used to evaluate the population.
    :type env: gym.vector.VectorEnv
    :param env_name: Environment name used in logs and checkpoint paths.
    :type env_name: str
    :param algo: Algorithm name.
    :type algo: str
    :param pop: Population of offline agents.
    :type pop: list
    :param memory: Replay buffer filled from *data*.
    :type memory: ReplayBuffer
    :param run: Loop, evolution, checkpoint, and logging settings.
    :type run: TrainRunConfig, optional
    :param data: Minari id or transition dict used to fill the buffer.
    :type data: OfflineDatasetConfig, optional
    :param init_hp: Initial hyperparameters logged to W&B.
    :type init_hp: dict, optional
    :param accelerator: Accelerator for distributed training.
    :type accelerator: Accelerator, optional
    :return: Trained population and last fitnesses.
    :rtype: tuple[list, list[float]]
    """
    run = resolve_train_evolution(run or TrainRunConfig())
    data = data or OfflineDatasetConfig()
    validate_train_run(run, algo=algo)
    memory = _fill_offline_memory(memory, data, accelerator)
    train = start_train_session(
        pop,
        run,
        algo=algo,
        env_name=env_name,
        accelerator=accelerator,
        init_hp=init_hp,
    )
    session = OfflineSession(
        train=train,
        env=env,
        sampler=_offline_sampler(pop, memory, accelerator),
    )
    while train.population.all_below(run.loop.max_steps):
        _learn_offline_generation(session)
        _evaluate_offline(session)
        report_after_eval(train)
        if stop_if_target(train):
            return train.population.agents, train.population.last_scalar_fitnesses
        maybe_evolve(train)
        maybe_checkpoint(train, steps=train.population.agents[0].metrics.steps)
    close_training(train)
    return train.population.agents, train.population.last_scalar_fitnesses
