# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from typing import Protocol, TypeGuard

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from accelerate import Accelerator
from tensordict import TensorDict
from torch.utils.data import DataLoader

from agilerl.algorithms import DDPG, DQN, TD3, RainbowDQN
from agilerl.components import (
    MultiStepReplayBuffer,
    PrioritizedReplayBuffer,
)
from agilerl.components.data import (
    ReplayDataset,
    Transition,
    transition_to_tensordict,
)
from agilerl.components.replay_buffer import BufferType
from agilerl.components.sampler import Sampler
from agilerl.networks.actors import DeterministicActor
from agilerl.training.configs import OffPolicyExploreConfig, TrainRunConfig
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
from agilerl.vector import DummyVecEnv

SupportedOffPolicy = DQN | RainbowDQN | DDPG | TD3
PopulationType = list[SupportedOffPolicy]

logger = logging.getLogger(__name__)


class NStepAgent(Protocol):
    """A RainbowDQN-style agent that consumes n-step / prioritized batches."""

    batch_size: int
    beta: float

    def learn(
        self,
        experiences: TensorDict,
        n_experiences: TensorDict | None = None,
        per: bool = False,
    ) -> tuple[float, torch.Tensor | None, npt.NDArray | None]: ...


def _is_n_step_agent(agent: SupportedOffPolicy) -> TypeGuard[NStepAgent]:
    """Whether *agent* exposes the n-step / prioritized interface."""
    return hasattr(agent, "beta")


def _as_n_step_agent(agent: SupportedOffPolicy) -> NStepAgent:
    """Read ``agent`` through the n-step / prioritized interface."""
    assert _is_n_step_agent(agent), (
        f"Prioritized replay needs an n-step agent, got {type(agent).__name__}."
    )
    return agent


def _learn_from_buffer(
    agent: SupportedOffPolicy,
    sampler: Sampler,
    memory: BufferType,
    n_step_memory: MultiStepReplayBuffer | None,
    n_step_sampler: Sampler | None,
    per: bool,
) -> None:
    """Execute a single learning step for the agent."""
    sample = sampler.sample

    # Prioritized and n-step replay are the preserve of RainbowDQN: only it
    # anneals `beta`, accepts `n_experiences`/`per` in `learn`, returns indices
    # and priorities to write back, and pairs with a PrioritizedReplayBuffer.
    if per:
        n_step_agent = _as_n_step_agent(agent)
        assert isinstance(memory, PrioritizedReplayBuffer)
        experiences = sample(n_step_agent.batch_size, n_step_agent.beta)
        n_step_experiences = (
            n_step_sampler.sample(experiences["idxs"])
            if n_step_sampler is not None
            else None
        )
        _loss, idxs, priorities = n_step_agent.learn(
            experiences,
            n_experiences=n_step_experiences,
            per=per,
        )
        assert idxs is not None
        assert priorities is not None
        memory.update_priorities(idxs, priorities)
    else:
        experiences = sample(
            agent.batch_size,
            return_idx=n_step_memory is not None,
        )
        if n_step_sampler is not None:
            n_step_experiences = n_step_sampler.sample(experiences["idxs"])
            _as_n_step_agent(agent).learn(experiences, n_experiences=n_step_experiences)
        else:
            agent.learn(experiences)


def _off_policy_action(
    agent: SupportedOffPolicy,
    obs: npt.NDArray,
    info: dict[str, object],
    epsilon: float,
    eps_end: float,
    eps_decay: float,
) -> tuple[npt.NDArray, npt.NDArray, float]:
    """Choose an env action and the action stored in the replay buffer."""
    if isinstance(agent, DQN):
        action_mask = info.get("action_mask", None)
        action = agent.get_action(obs, epsilon, action_mask=action_mask)
        return action, action, max(eps_end, epsilon * eps_decay)
    if isinstance(agent, RainbowDQN):
        action_mask = info.get("action_mask", None)
        action = agent.get_action(obs, action_mask=action_mask)
        return action, action, epsilon
    raw_action = agent.get_action(obs)
    action = DeterministicActor.rescale_action(
        action=torch.from_numpy(raw_action),
        low=agent.action_low,
        high=agent.action_high,
        output_activation=agent.actor.output_activation,
    )
    return action.cpu().numpy(), raw_action, epsilon


def _record_finished_envs(
    done: npt.NDArray,
    trunc: npt.NDArray,
    scores: npt.NDArray,
    completed_episode_scores: list[float],
) -> list[int]:
    """Zero finished env scores and return their indices for noise reset."""
    reset_noise_indices = []
    for idx, (d, t) in enumerate(zip(done, trunc, strict=False)):
        if d or t:
            completed_episode_scores.append(scores[idx])
            scores[idx] = 0
            reset_noise_indices.append(idx)
    return reset_noise_indices


def _add_transition_to_memory(
    memory: BufferType,
    n_step_memory: MultiStepReplayBuffer | None,
    transition: TensorDict,
) -> None:
    """Write one vectorized transition into the replay buffer."""
    if n_step_memory is None:
        memory.add(transition)
        return
    one_step_transition = n_step_memory.add(transition)
    if one_step_transition is not None:
        memory.add(one_step_transition)


def _maybe_learn_off_policy(
    agent: SupportedOffPolicy,
    sampler: Sampler,
    memory: BufferType,
    n_step_memory: MultiStepReplayBuffer | None,
    n_step_sampler: Sampler | None,
    per: bool,
    idx_step: int,
    num_envs: int,
    learning_delay: int,
) -> None:
    """Learn from replay when the buffer and learn_step cadence allow it."""
    ready = len(memory) >= agent.batch_size and memory.size > learning_delay
    if agent.learn_step > num_envs:
        learn_step = agent.learn_step // num_envs
        if idx_step % learn_step == 0 and ready:
            _learn_from_buffer(
                agent,
                sampler,
                memory,
                n_step_memory,
                n_step_sampler,
                per,
            )
        return
    if not ready:
        return
    for _ in range(num_envs // agent.learn_step):
        _learn_from_buffer(
            agent,
            sampler,
            memory,
            n_step_memory,
            n_step_sampler,
            per,
        )


@dataclass
class OffPolicySession:
    """Replay and exploration state for one off-policy training run."""

    train: TrainSession
    vec_env: gym.vector.VectorEnv
    memory: BufferType
    n_step_memory: MultiStepReplayBuffer | None
    sampler: Sampler
    n_step_sampler: Sampler | None
    per: bool
    num_envs: int
    explore: OffPolicyExploreConfig
    eps_start: float


def _off_policy_samplers(
    pop: PopulationType,
    memory: BufferType,
    n_step_memory: MultiStepReplayBuffer | None,
    accelerator: Accelerator | None,
) -> tuple[Sampler, Sampler | None]:
    """Build the replay sampler and optional n-step sampler."""
    if accelerator is None:
        n_step_sampler = (
            Sampler(memory=n_step_memory) if n_step_memory is not None else None
        )
        return Sampler(memory=memory), n_step_sampler
    replay_dataset = ReplayDataset(memory, pop[0].batch_size)
    replay_dataloader = accelerator.prepare(DataLoader(replay_dataset, batch_size=None))
    # n-step sampling needs index lookups the distributed sampler does not support.
    return Sampler(dataset=replay_dataset, dataloader=replay_dataloader), None


def _collect_off_policy_agent(
    session: OffPolicySession,
    agent: SupportedOffPolicy,
    epsilon: float,
) -> float:
    """Collect one evolution window of transitions and learn from replay."""
    explore = session.explore
    num_envs = session.num_envs
    agent.set_training_mode(True)
    agent.init_training_step(session.train.capture_grama)
    obs, info = session.vec_env.reset()
    scores = np.zeros(num_envs)
    completed_episode_scores: list[float] = []
    steps = 0
    for idx_step in range(session.train.run.loop.evo_steps // num_envs):
        action, buffer_action, epsilon = _off_policy_action(
            agent,
            obs,
            info,
            epsilon,
            explore.eps_end,
            explore.eps_decay,
        )
        next_obs, reward, done, trunc, info = session.vec_env.step(action)
        scores += np.array(reward)
        reset_noise_indices = _record_finished_envs(
            done,
            trunc,
            scores,
            completed_episode_scores,
        )
        if isinstance(agent, (DDPG, TD3)):
            agent.reset_action_noise(reset_noise_indices)
        steps += num_envs
        transition = transition_to_tensordict(
            Transition(
                obs=obs,
                action=buffer_action,
                reward=reward,
                next_obs=next_obs,
                done=done,
            )
        )
        transition.batch_size = torch.Size([num_envs])
        _add_transition_to_memory(session.memory, session.n_step_memory, transition)
        if session.per:
            fraction = min(
                ((agent.metrics.steps + idx_step + 1) * num_envs / session.train.run.loop.max_steps),
                1.0,
            )
            n_step_agent = _as_n_step_agent(agent)
            n_step_agent.beta += fraction * (1.0 - n_step_agent.beta)
        _maybe_learn_off_policy(
            agent,
            session.sampler,
            session.memory,
            session.n_step_memory,
            session.n_step_sampler,
            session.per,
            idx_step,
            num_envs,
            explore.learning_delay,
        )
        obs = next_obs
    agent.add_scores(completed_episode_scores)
    agent.finalize_training_step(steps)
    session.train.pbar.update(session.train.run.loop.evo_steps // session.train.population.size)
    return epsilon


def _run_off_policy_generation(session: OffPolicySession) -> None:
    """Collect rollouts for every agent in the population."""
    if session.train.accelerator is not None:
        session.train.accelerator.wait_for_everyone()
    epsilon = session.eps_start
    agent: SupportedOffPolicy | None = None
    for agent in session.train.population.agents:
        epsilon = _collect_off_policy_agent(session, agent, epsilon)
    if agent is not None and isinstance(agent, DQN):
        session.eps_start = epsilon


def _evaluate_off_policy(session: OffPolicySession) -> None:
    """Run test episodes for the current population."""
    loop = session.train.run.loop
    for agent in session.train.population.agents:
        agent.test(session.vec_env, max_steps=loop.eval_steps, loop=loop.eval_loop)


@accept_flat_kwargs
def train_off_policy(
    env: gym.Env | gym.vector.VectorEnv,
    env_name: str,
    algo: str,
    pop: PopulationType,
    memory: BufferType,
    run: TrainRunConfig | None = None,
    explore: OffPolicyExploreConfig | None = None,
    init_hp: InitHyperparams = None,
    accelerator: Accelerator | None = None,
    n_step_memory: MultiStepReplayBuffer | None = None,
) -> tuple[PopulationType, list[float]]:
    """Train an off-policy population; returns agents and fitnesses.

    :param env: Environment to train in. Can be vectorized.
    :type env: gym.Env | gym.vector.VectorEnv
    :param env_name: Environment name used in logs and checkpoint paths.
    :type env_name: str
    :param algo: Algorithm name.
    :type algo: str
    :param pop: Population of off-policy agents.
    :type pop: list
    :param memory: Replay buffer.
    :type memory: BufferType
    :param run: Loop, evolution, checkpoint, and logging settings.
    :type run: TrainRunConfig, optional
    :param explore: Epsilon-greedy schedule and learning delay.
    :type explore: OffPolicyExploreConfig, optional
    :param init_hp: Initial hyperparameters logged to W&B.
    :type init_hp: dict, optional
    :param accelerator: Accelerator for distributed training.
    :type accelerator: Accelerator, optional
    :param n_step_memory: Optional n-step buffer used with prioritized replay.
    :type n_step_memory: MultiStepReplayBuffer, optional
    :return: Trained population and last fitnesses.
    :rtype: tuple[list, list[float]]
    """
    run = resolve_train_evolution(run or TrainRunConfig())
    explore = explore or OffPolicyExploreConfig()
    validate_train_run(run, algo=algo)
    assert isinstance(explore.eps_start, float), "Starting epsilon must be a float."
    assert isinstance(explore.eps_end, float), "Final value of epsilon must be a float."
    assert isinstance(explore.eps_decay, float), "Epsilon decay rate must be a float."
    vec_env: gym.vector.VectorEnv = (
        env if isinstance(env, gym.vector.VectorEnv) else DummyVecEnv(env)
    )
    sampler, n_step_sampler = _off_policy_samplers(
        pop, memory, n_step_memory, accelerator
    )
    train = start_train_session(
        pop,
        run,
        algo=algo,
        env_name=env_name,
        accelerator=accelerator,
        init_hp=init_hp,
    )
    session = OffPolicySession(
        train=train,
        vec_env=vec_env,
        memory=memory,
        n_step_memory=n_step_memory,
        sampler=sampler,
        n_step_sampler=n_step_sampler,
        per=isinstance(memory, PrioritizedReplayBuffer),
        num_envs=get_num_envs(vec_env),
        explore=explore,
        eps_start=explore.eps_start,
    )
    while train.population.all_below(run.loop.max_steps):
        _run_off_policy_generation(session)
        _evaluate_off_policy(session)
        report_after_eval(train)
        if stop_if_target(train):
            return train.population.agents, train.population.last_scalar_fitnesses
        evolve_and_checkpoint(train)
    close_training(train)
    return train.population.agents, train.population.last_scalar_fitnesses


