# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from accelerate import Accelerator
from pettingzoo import ParallelEnv
from torch.utils.data import DataLoader

from agilerl.algorithms import MADDPG, MATD3
from agilerl.components.data import (
    MultiAgentTransition,
    ReplayDataset,
    transition_to_tensordict,
)
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.components.sampler import Sampler
from agilerl.training.configs import (
    MultiAgentOffPolicyExploreConfig,
    MultiAgentTrainRunConfig,
)
from agilerl.training.loop import (
    TrainSession,
    close_training,
    evolve_and_checkpoint,
    report_after_eval,
    resolve_train_evolution,
    start_train_session,
    stop_if_target,
    validate_train_run,
)
from agilerl.typing import InitHyperparams
from agilerl.utils.algo_utils import get_num_envs
from agilerl.utils.constructor_kwargs import accept_flat_kwargs
from agilerl.vector import PzDummyVecEnv
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.vector.pz_vec_env import PettingZooVecEnv

PopulationType = list[MADDPG | MATD3]

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentOffPolicySession:
    """Vectorized env, replay, and score settings for one MA off-policy run."""

    train: TrainSession
    vec_env: PettingZooVecEnv
    memory: ReplayBuffer
    sampler: Sampler
    num_envs: int
    explore: MultiAgentOffPolicyExploreConfig


def _ma_off_policy_sampler(
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


def _ma_score_increment(
    reward: dict[str, npt.NDArray],
    sum_scores: bool,
) -> npt.NDArray:
    """Stack per-agent rewards and optionally sum them into one score column."""
    agent_rewards = np.column_stack(
        [np.asarray(v).ravel() for v in reward.values()]
    )
    agent_rewards = np.where(np.isnan(agent_rewards), 0, agent_rewards)
    if sum_scores:
        return np.sum(agent_rewards, axis=-1)[:, np.newaxis]
    return agent_rewards


def _ma_env_dones(
    agent: MADDPG | MATD3,
    termination: dict[str, npt.NDArray],
    truncation: dict[str, npt.NDArray],
) -> dict[str, npt.NDArray]:
    """Per-agent done flags, treating NaN terminated as finished."""
    dones = {}
    for agent_id in agent.agent_ids:
        terminated = np.where(
            np.isnan(termination.get(agent_id, True)),
            True,
            termination.get(agent_id, True),
        ).astype(bool)
        truncated = np.where(
            np.isnan(truncation.get(agent_id, False)),
            False,
            truncation.get(agent_id, False),
        ).astype(bool)
        dones[agent_id] = terminated | truncated
    return dones


def _record_ma_finished_envs(
    dones: dict[str, npt.NDArray],
    scores: npt.NDArray,
    completed_episode_scores: list[float | list[float]],
    sum_scores: bool,
) -> list[int]:
    """Zero finished env rows and return their indices for noise reset."""
    reset_noise_indices = []
    for idx, agent_dones in enumerate(zip(*dones.values(), strict=False)):
        if all(agent_dones):
            completed_score = (
                np.asarray(scores[idx]).item() if sum_scores else list(scores[idx])
            )
            completed_episode_scores.append(completed_score)
            scores[idx].fill(0)
            reset_noise_indices.append(idx)
    return reset_noise_indices


def _maybe_learn_ma_off_policy(
    agent: MADDPG | MATD3,
    sampler: Sampler,
    memory: ReplayBuffer,
    idx_step: int,
    num_envs: int,
    learning_delay: int,
) -> None:
    """Learn from replay when the buffer and learn_step cadence allow it."""
    ready = len(memory) >= agent.batch_size and memory.counter > learning_delay
    sample = sampler.sample
    if agent.learn_step > num_envs:
        if idx_step % (agent.learn_step // num_envs) == 0 and ready:
            agent.learn(sample(agent.batch_size))
        return
    if not ready:
        return
    for _ in range(num_envs // agent.learn_step):
        agent.learn(sample(agent.batch_size))


def _collect_ma_off_policy_agent(
    session: MultiAgentOffPolicySession,
    agent: MADDPG | MATD3,
) -> None:
    """Collect one evolution window of multi-agent transitions and learn."""
    num_envs = session.num_envs
    sum_scores = session.explore.sum_scores
    agent.set_training_mode(True)
    agent.init_training_step(session.train.capture_grama)
    obs, info = session.vec_env.reset()
    scores = (
        np.zeros((num_envs, 1))
        if sum_scores
        else np.zeros((num_envs, len(agent.agent_ids)))
    )
    completed_episode_scores: list[float | list[float]] = []
    steps = 0
    for idx_step in range(session.train.run.loop.evo_steps // num_envs):
        action, raw_action = agent.get_action(obs=obs, infos=info)
        next_obs, reward, termination, truncation, info = session.vec_env.step(action)
        scores += _ma_score_increment(reward, sum_scores)
        steps += num_envs
        transition = transition_to_tensordict(
            MultiAgentTransition(
                obs=obs,
                action=raw_action,
                reward=reward,
                next_obs=next_obs,
                done=termination,
            )
        )
        transition.batch_size = torch.Size([num_envs])
        session.memory.add(transition)
        _maybe_learn_ma_off_policy(
            agent,
            session.sampler,
            session.memory,
            idx_step,
            num_envs,
            session.explore.learning_delay,
        )
        obs = next_obs
        reset_noise_indices = _record_ma_finished_envs(
            _ma_env_dones(agent, termination, truncation),
            scores,
            completed_episode_scores,
            sum_scores,
        )
        agent.reset_action_noise(reset_noise_indices)
    agent.add_scores(completed_episode_scores)
    agent.finalize_training_step(steps)
    session.train.pbar.update(
        session.train.run.loop.evo_steps // session.train.population.size
    )


def _run_ma_off_policy_generation(session: MultiAgentOffPolicySession) -> None:
    """Collect rollouts for every agent in the population."""
    if session.train.accelerator is not None:
        session.train.accelerator.wait_for_everyone()
    for agent in session.train.population.agents:
        _collect_ma_off_policy_agent(session, agent)


def _evaluate_ma_off_policy(session: MultiAgentOffPolicySession) -> None:
    """Run test episodes for the current population."""
    loop = session.train.run.loop
    sum_scores = session.explore.sum_scores
    for agent in session.train.population.agents:
        agent.test(
            session.vec_env,
            max_steps=loop.eval_steps,
            loop=loop.eval_loop,
            sum_scores=sum_scores,
        )


@accept_flat_kwargs
def train_multi_agent_off_policy(
    env: ParallelEnv | AsyncPettingZooVecEnv,
    env_name: str,
    algo: str,
    pop: PopulationType,
    memory: ReplayBuffer,
    run: MultiAgentTrainRunConfig | None = None,
    explore: MultiAgentOffPolicyExploreConfig | None = None,
    init_hp: InitHyperparams = None,
    accelerator: Accelerator | None = None,
) -> tuple[PopulationType, list[float] | list[dict[str, float]]]:
    """Train a multi-agent off-policy population; returns agents and fitnesses.

    :param env: PettingZoo env or vectorized wrapper.
    :type env: ParallelEnv | AsyncPettingZooVecEnv
    :param env_name: Environment name used in logs and checkpoint paths.
    :type env_name: str
    :param algo: Algorithm name.
    :type algo: str
    :param pop: Population of MADDPG / MATD3 agents.
    :type pop: list
    :param memory: Replay buffer.
    :type memory: ReplayBuffer
    :param run: Loop, evolution, checkpoint, and logging settings.
    :type run: MultiAgentTrainRunConfig, optional
    :param explore: Learning delay and whether to sum sub-agent scores.
    :type explore: MultiAgentOffPolicyExploreConfig, optional
    :param init_hp: Initial hyperparameters logged to W&B.
    :type init_hp: dict, optional
    :param accelerator: Accelerator for distributed training.
    :type accelerator: Accelerator, optional
    :return: Trained population and last fitnesses.
    :rtype: tuple[list, list]
    """
    run = resolve_train_evolution(run or MultiAgentTrainRunConfig())
    explore = explore or MultiAgentOffPolicyExploreConfig()
    validate_train_run(run, algo=algo)
    vec_env: PettingZooVecEnv = (
        env if isinstance(env, PettingZooVecEnv) else PzDummyVecEnv(env)
    )
    train = start_train_session(
        pop,
        run,
        algo=algo,
        env_name=env_name,
        accelerator=accelerator,
        init_hp=init_hp,
        wandb_kwargs_overlay={"project": "AgileRLMultiAgent"},
    )
    session = MultiAgentOffPolicySession(
        train=train,
        vec_env=vec_env,
        memory=memory,
        sampler=_ma_off_policy_sampler(pop, memory, accelerator),
        num_envs=get_num_envs(vec_env),
        explore=explore,
    )
    while train.population.all_below(run.loop.max_steps):
        _run_ma_off_policy_generation(session)
        _evaluate_ma_off_policy(session)
        report_after_eval(train)
        if stop_if_target(train):
            return train.population.agents, train.population.last_fitnesses
        evolve_and_checkpoint(train)
    close_training(train)
    return train.population.agents, train.population.last_fitnesses
