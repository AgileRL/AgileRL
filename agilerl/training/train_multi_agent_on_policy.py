# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from accelerate import Accelerator
from gymnasium import spaces
from pettingzoo import ParallelEnv

from agilerl.algorithms import IPPO
from agilerl.networks import StochasticActor
from agilerl.training.configs import MultiAgentScoreConfig, MultiAgentTrainRunConfig
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
from agilerl.vector.pz_vec_env import PettingZooVecEnv

if TYPE_CHECKING:
    from agilerl.typing import SingleAgentModule

MultiAgentOnPolicyAlgorithms = IPPO
PopulationType = list[MultiAgentOnPolicyAlgorithms]


@dataclass
class MultiAgentOnPolicySession:
    """Vectorized env and score settings for one MA on-policy run."""

    train: TrainSession
    vec_env: PettingZooVecEnv
    num_envs: int
    scores: MultiAgentScoreConfig


def _clip_box_policy_action(
    agent_policy: object,
    agent_space: spaces.Space,
    agent_action: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Clip or scale a continuous action to the env Box."""
    if not (
        isinstance(agent_policy, StochasticActor)
        and isinstance(agent_space, spaces.Box)
    ):
        return agent_action
    if agent_policy.squash_output:
        return agent_policy.scale_action(agent_action)
    return np.clip(agent_action, agent_space.low, agent_space.high)


def _unwrap_compiled_policy(
    agent: IPPO,
    network_id: str,
    compiled_agent: bool,
) -> object:
    """Return the policy module used to clip actions, unwrapping torch.compile."""
    policy_name = agent.registry.policy()
    assert policy_name is not None, "Agent registry does not define a policy network."
    policy = getattr(agent, policy_name)
    agent_policy: SingleAgentModule = policy[network_id]
    if compiled_agent:
        return agent_policy._orig_mod
    return agent_policy


def _clip_ippo_actions(
    agent: IPPO,
    action: dict[str, np.ndarray | torch.Tensor],
    compiled_agent: bool,
) -> dict[str, np.ndarray | torch.Tensor]:
    """Clip each agent's action to its Box when the policy is continuous."""
    clipped_action = {}
    for agent_id, agent_action in action.items():
        network_id = (
            agent_id if agent_id in agent.actors else agent.get_group_id(agent_id)
        )
        clipped_action[agent_id] = _clip_box_policy_action(
            _unwrap_compiled_policy(agent, network_id, compiled_agent),
            agent.possible_action_spaces[agent_id],
            agent_action,
        )
    return clipped_action


def _ma_on_policy_score_increment(
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


def _next_ma_dones(
    termination: dict[str, npt.NDArray],
    truncation: dict[str, npt.NDArray],
) -> dict[str, npt.NDArray]:
    """Combine terminated/truncated, leaving NaN for inactive agents."""
    next_done = {}
    for agent_id in termination:
        terminated = termination[agent_id]
        truncated = truncation[agent_id]
        mask = ~(np.isnan(terminated) | np.isnan(truncated))
        result = np.full_like(mask, np.nan, dtype=float)
        result[mask] = np.logical_or(terminated[mask], truncated[mask])
        next_done[agent_id] = result
    return next_done


def _record_ippo_finished_envs(
    next_done: dict[str, npt.NDArray],
    scores: npt.NDArray,
    completed_episode_scores: list[float | list[float]],
    sum_scores: bool,
    agent_ids: list[str],
    num_envs: int,
) -> dict[str, npt.NDArray]:
    """Zero finished env rows and reset per-agent done flags when all agents finish."""
    done = next_done
    for idx, agent_dones in enumerate(zip(*next_done.values(), strict=False)):
        if all(agent_dones):
            completed_score = (
                float(scores[idx].item()) if sum_scores else list(scores[idx])
            )
            completed_episode_scores.append(completed_score)
            scores[idx].fill(0)
            done = {agent_id: np.zeros(num_envs) for agent_id in agent_ids}
    return done


def _append_ippo_transition(
    obs: dict[str, npt.NDArray],
    action: dict[str, object],
    log_prob: dict[str, object],
    reward: dict[str, object],
    value: dict[str, object],
    done: dict[str, npt.NDArray],
    buffers: tuple[
        dict[str, list],
        dict[str, list],
        dict[str, list],
        dict[str, list],
        dict[str, list],
        dict[str, list],
    ],
) -> None:
    """Append one env step onto the per-agent rollout lists."""
    states, actions, log_probs, rewards, dones, values = buffers
    for agent_id in obs:
        states[agent_id].append(obs[agent_id])
        rewards[agent_id].append(reward[agent_id])
        actions[agent_id].append(action[agent_id])
        log_probs[agent_id].append(log_prob[agent_id])
        values[agent_id].append(value[agent_id])
        dones[agent_id].append(done[agent_id])


def _collect_ippo_learn_step(
    session: MultiAgentOnPolicySession,
    agent: IPPO,
    compiled_agent: bool,
    obs: dict[str, npt.NDArray],
    info: dict[str, object],
    scores: npt.NDArray,
    completed_episode_scores: list[float | list[float]],
) -> tuple[object, dict[str, npt.NDArray], dict[str, npt.NDArray], int]:
    """Collect ``learn_step`` env steps and return the IPPO experience tuple."""
    num_envs = session.num_envs
    sum_scores = session.scores.sum_scores
    agent_ids = agent.agent_ids
    states = {agent_id: [] for agent_id in agent_ids}
    actions = {agent_id: [] for agent_id in agent_ids}
    log_probs = {agent_id: [] for agent_id in agent_ids}
    rewards = {agent_id: [] for agent_id in agent_ids}
    dones = {agent_id: [] for agent_id in agent_ids}
    values = {agent_id: [] for agent_id in agent_ids}
    done = {agent_id: np.zeros(num_envs) for agent_id in agent_ids}
    steps = 0
    next_obs, next_done = obs, done
    buffers = (states, actions, log_probs, rewards, dones, values)
    for _ in range(-(agent.learn_step // -num_envs)):
        action, log_prob, _entropy, value = agent.get_action(obs=obs, infos=info)
        clipped_action = _clip_ippo_actions(agent, action, compiled_agent)
        next_obs, reward, termination, truncation, info = session.vec_env.step(
            clipped_action,
        )
        scores += _ma_on_policy_score_increment(reward, sum_scores)
        steps += num_envs
        _append_ippo_transition(
            obs,
            action,
            log_prob,
            reward,
            value,
            done,
            buffers,
        )
        next_done = _next_ma_dones(termination, truncation)
        obs = next_obs
        done = _record_ippo_finished_envs(
            next_done,
            scores,
            completed_episode_scores,
            sum_scores,
            agent_ids,
            num_envs,
        )
    experiences = (
        states,
        actions,
        log_probs,
        rewards,
        dones,
        values,
        next_obs,
        next_done,
    )
    return experiences, obs, info, steps


def _collect_ippo_agent(
    session: MultiAgentOnPolicySession,
    agent: IPPO,
) -> None:
    """Collect rollouts and learn until evo_steps for one IPPO agent."""
    num_envs = session.num_envs
    sum_scores = session.scores.sum_scores
    compiled_agent = agent.torch_compiler is not None
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
    for _ in range(-(session.train.run.loop.evo_steps // -agent.learn_step)):
        experiences, obs, info, chunk_steps = _collect_ippo_learn_step(
            session,
            agent,
            compiled_agent,
            obs,
            info,
            scores,
            completed_episode_scores,
        )
        steps += chunk_steps
        agent.learn(experiences)
    agent.add_scores(completed_episode_scores)
    agent.finalize_training_step(steps)
    session.train.pbar.update(steps // session.train.population.size)


def _run_ma_on_policy_generation(session: MultiAgentOnPolicySession) -> None:
    """Collect rollouts for every agent in the population."""
    if session.train.accelerator is not None:
        session.train.accelerator.wait_for_everyone()
    for agent in session.train.population.agents:
        _collect_ippo_agent(session, agent)


def _evaluate_ma_on_policy(session: MultiAgentOnPolicySession) -> None:
    """Run test episodes for the current population."""
    loop = session.train.run.loop
    sum_scores = session.scores.sum_scores
    for agent in session.train.population.agents:
        agent.test(
            session.vec_env,
            max_steps=loop.eval_steps,
            loop=loop.eval_loop,
            sum_scores=sum_scores,
        )


@accept_flat_kwargs
def train_multi_agent_on_policy(
    env: ParallelEnv | PettingZooVecEnv,
    env_name: str,
    algo: str,
    pop: PopulationType,
    run: MultiAgentTrainRunConfig | None = None,
    scores: MultiAgentScoreConfig | None = None,
    init_hp: InitHyperparams = None,
    accelerator: Accelerator | None = None,
) -> tuple[PopulationType, list[float] | list[dict[str, float]]]:
    """Train a multi-agent on-policy population; returns agents and fitnesses.

    :param env: PettingZoo env or vectorized wrapper.
    :type env: ParallelEnv | PettingZooVecEnv
    :param env_name: Environment name used in logs and checkpoint paths.
    :type env_name: str
    :param algo: Algorithm name.
    :type algo: str
    :param pop: Population of IPPO agents.
    :type pop: list
    :param run: Loop, evolution, checkpoint, and logging settings.
    :type run: MultiAgentTrainRunConfig, optional
    :param scores: Whether to sum sub-agent rewards into one episode score.
    :type scores: MultiAgentScoreConfig, optional
    :param init_hp: Initial hyperparameters logged to W&B.
    :type init_hp: dict, optional
    :param accelerator: Accelerator for distributed training.
    :type accelerator: Accelerator, optional
    :return: Trained population and last fitnesses.
    :rtype: tuple[list, list]
    """
    run = resolve_train_evolution(run or MultiAgentTrainRunConfig())
    scores = scores or MultiAgentScoreConfig()
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
    session = MultiAgentOnPolicySession(
        train=train,
        vec_env=vec_env,
        num_envs=get_num_envs(vec_env),
        scores=scores,
    )
    while train.population.all_below(run.loop.max_steps):
        _run_ma_on_policy_generation(session)
        _evaluate_ma_on_policy(session)
        report_after_eval(train)
        if stop_if_target(train):
            return train.population.agents, train.population.last_fitnesses
        evolve_and_checkpoint(train)
    close_training(train)
    return train.population.agents, train.population.last_fitnesses
