# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Callable
from dataclasses import dataclass

import gymnasium as gym
from accelerate import Accelerator

from agilerl.algorithms import PPO
from agilerl.rollouts import collect_rollouts, collect_rollouts_recurrent
from agilerl.training.configs import TrainRunConfig
from agilerl.training.loop import (
    TrainSession,
    close_training,
    evolve_and_checkpoint,
    report_before_evo,
    resolve_train_evolution,
    start_train_session,
    stop_if_target,
    validate_train_run,
)
from agilerl.typing import InitHyperparams, RolloutReturn
from agilerl.utils.algo_utils import get_num_envs
from agilerl.utils.constructor_kwargs import accept_flat_kwargs
from agilerl.vector import DummyVecEnv

OnPolicyAlgorithms = PPO
PopulationType = list[OnPolicyAlgorithms]
CollectRolloutsFn = Callable[..., RolloutReturn]

logger = logging.getLogger(__name__)


@dataclass
class OnPolicySession:
    """Vectorized env and rollout collector for one on-policy run."""

    train: TrainSession
    vec_env: gym.vector.VectorEnv
    num_envs: int
    collect_rollouts_fn: CollectRolloutsFn | None


def _collect_on_policy_agent(
    session: OnPolicySession,
    agent: OnPolicyAlgorithms,
    collect_fn: CollectRolloutsFn,
) -> CollectRolloutsFn:
    """Collect rollouts and learn for one evolution window."""
    num_envs = session.num_envs
    evo_steps = session.train.run.loop.evo_steps
    agent.set_training_mode(True)
    agent.init_training_step(session.train.capture_grama)
    steps = 0
    completed_episode_scores: list[float] = []
    n_steps = -(agent.learn_step // -num_envs)
    last_obs, last_done, last_scores, last_info = None, None, None, None
    for _ in range(-(evo_steps // -agent.learn_step)):
        episode_scores, last_obs, last_done, last_scores, last_info = collect_fn(
            agent,
            session.vec_env,
            n_steps=n_steps,
            last_obs=last_obs,
            last_done=last_done,
            last_scores=last_scores,
            last_info=last_info,
        )
        agent.learn()
        steps += n_steps * num_envs
        completed_episode_scores += episode_scores
    agent.add_scores(completed_episode_scores)
    agent.finalize_training_step(steps)
    session.train.pbar.update(steps // session.train.population.size)
    return collect_fn


def _run_on_policy_generation(session: OnPolicySession) -> None:
    """Collect rollouts for every agent in the population."""
    if session.train.accelerator is not None:
        session.train.accelerator.wait_for_everyone()
    collect_fn = session.collect_rollouts_fn
    for agent in session.train.population.agents:
        if collect_fn is None:
            collect_fn = (
                collect_rollouts_recurrent
                if getattr(agent, "recurrent", False)
                else collect_rollouts
            )
        collect_fn = _collect_on_policy_agent(session, agent, collect_fn)
    session.collect_rollouts_fn = collect_fn


def _evaluate_on_policy(session: OnPolicySession) -> None:
    """Run test episodes for the current population."""
    loop = session.train.run.loop
    for agent in session.train.population.agents:
        agent.test(session.vec_env, max_steps=loop.eval_steps, loop=loop.eval_loop)


@accept_flat_kwargs
def train_on_policy(
    env: gym.Env | gym.vector.VectorEnv,
    env_name: str,
    algo: str,
    pop: PopulationType,
    run: TrainRunConfig | None = None,
    init_hp: InitHyperparams = None,
    accelerator: Accelerator | None = None,
    collect_rollouts_fn: CollectRolloutsFn | None = None,
) -> tuple[PopulationType, list[float]]:
    """Train an on-policy population; returns agents and fitnesses.

    :param env: Environment to train in. Can be vectorized.
    :type env: gym.Env | gym.vector.VectorEnv
    :param env_name: Environment name used in logs and checkpoint paths.
    :type env_name: str
    :param algo: Algorithm name.
    :type algo: str
    :param pop: Population of on-policy agents.
    :type pop: list
    :param run: Loop, evolution, checkpoint, and logging settings.
    :type run: TrainRunConfig, optional
    :param init_hp: Initial hyperparameters logged to W&B.
    :type init_hp: dict, optional
    :param accelerator: Accelerator for distributed training.
    :type accelerator: Accelerator, optional
    :param collect_rollouts_fn: Optional rollout collector; chosen from the
        agent when omitted.
    :type collect_rollouts_fn: Callable, optional
    :return: Trained population and last fitnesses.
    :rtype: tuple[list, list[float]]
    """
    run = resolve_train_evolution(run or TrainRunConfig())
    validate_train_run(run, algo=algo)
    vec_env: gym.vector.VectorEnv = (
        env if isinstance(env, gym.vector.VectorEnv) else DummyVecEnv(env)
    )
    train = start_train_session(
        pop,
        run,
        algo=algo,
        env_name=env_name,
        accelerator=accelerator,
        init_hp=init_hp,
    )
    session = OnPolicySession(
        train=train,
        vec_env=vec_env,
        num_envs=get_num_envs(vec_env),
        collect_rollouts_fn=collect_rollouts_fn,
    )
    while train.population.all_below(run.loop.max_steps):
        _run_on_policy_generation(session)
        _evaluate_on_policy(session)
        report_before_evo(train)
        if stop_if_target(train):
            return train.population.agents, train.population.last_scalar_fitnesses
        evolve_and_checkpoint(train)
    close_training(train)
    return train.population.agents, train.population.last_scalar_fitnesses
