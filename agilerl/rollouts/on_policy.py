# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

"""Functions for collecting rollouts for on-policy algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import PPO
from agilerl.components.llm_rollout_data import (
    LLMExperienceBatch,
    RolloutGroup,
    Trajectory,
    collate_rollout_groups,
)
from agilerl.networks import StochasticActor
from agilerl.typing import RolloutReturn

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from agilerl.algorithms import GRPO, LLMPPO, LLMREINFORCE
    from agilerl.llm_envs import RolloutCollector

SupportedOnPolicy = PPO
RolloutHarness = gym.Env | gym.vector.VectorEnv


def _collect_rollouts(
    agent: SupportedOnPolicy,
    env: RolloutHarness,
    n_steps: int | None = None,
    last_obs: npt.NDArray | None = None,
    last_done: npt.NDArray | None = None,
    last_scores: npt.NDArray | None = None,
    last_info: dict[str, Any] | None = None,
    *,
    recurrent: bool,
) -> RolloutReturn:
    """Collect rollouts for on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: SupportedOnPolicy
    :param env: The environment to collect rollouts from.
    :type env: gym.Env | gym.vector.VectorEnv
    :param n_steps: The number of steps to collect rollouts for. Defaults to agent.learn_step if not provided.
    :type n_steps: int | None
    :param last_obs: The observation to use for the first step. Defaults to None, where the environment is reset.
    :type last_obs: npt.NDArray | None
    :param last_done: The done flag to use for the first step. Defaults to None, where the environment is reset.
    :type last_done: npt.NDArray | None
    :param last_scores: The scores to use for the first step. Defaults to None, where the environment is reset.
    :type last_scores: npt.NDArray | None
    :param last_info: The info for the current step. Defaults to None, where the environment is reset.
    :type last_info: dict[str, Any] | None
    :param recurrent: Whether the agent is recurrent.
    :type recurrent: bool

    :return: The scores for the episodes completed in the rollouts, followed by
        the observation, done flag, scores, and info for the current step.
    :rtype: tuple[list[float], npt.NDArray, npt.NDArray, npt.NDArray, dict[str, Any]]
    """
    if (
        last_obs is None
        or last_done is None
        or last_scores is None
        or last_info is None
    ):
        obs, info = env.reset()
        scores = np.zeros(agent.num_envs)
        done = np.zeros(agent.num_envs)
        agent.hidden_state = (
            agent.get_initial_hidden_state(agent.num_envs) if recurrent else None
        )
    else:
        obs = last_obs
        done = last_done
        scores = last_scores
        info = last_info

    n_steps = n_steps or -(agent.learn_step // -agent.num_envs)
    agent.rollout_buffer.reset()

    current_hidden_state_for_actor = agent.hidden_state

    completed_episode_scores = []
    for _ in range(n_steps):
        current_hidden_state_for_buffer = current_hidden_state_for_actor

        # Get action, statistics and (maybe) recurrent hidden state from agent.
        # ``get_action`` returns a 4-tuple, or a 5-tuple with the next hidden
        # state for recurrent policies; the length check narrows the union.
        if recurrent:
            action_result = agent.get_action(
                obs,
                action_mask=info.get("action_mask", None),
                hidden_state=current_hidden_state_for_actor,
            )
            assert len(action_result) == 5, (
                "Recurrent get_action must return a 5-tuple "
                "(action, log_prob, entropy, value, next_hidden_state)."
            )
            action, log_prob, _, value, next_hidden_for_actor = action_result
            agent.hidden_state = next_hidden_for_actor
        else:
            action_result = agent.get_action(
                obs,
                action_mask=info.get("action_mask", None),
            )
            assert len(action_result) == 4, (
                "Non-recurrent get_action must return a 4-tuple "
                "(action, log_prob, entropy, value)."
            )
            action, log_prob, _, value = action_result

        # Clip action to action space
        policy_attr = agent.registry.policy()
        assert isinstance(policy_attr, str)  # PPO always registers a policy network
        policy = getattr(agent, policy_attr)
        if isinstance(policy, StochasticActor) and isinstance(
            agent.action_space,
            spaces.Box,
        ):
            clipped_action = np.clip(
                action,
                agent.action_space.low,
                agent.action_space.high,
            )
        else:
            clipped_action = action

        next_obs, reward, term, trunc, next_info = env.step(clipped_action)

        # Check if termination condition is met
        if isinstance(term, (list, np.ndarray)):
            is_terminal = (
                np.logical_or(term, trunc)
                if isinstance(trunc, (list, np.ndarray))
                else term
            )
        else:
            is_terminal = term or trunc

        # ``env.step`` types the reward as ``SupportsFloat | np.ndarray``, which
        # numpy's ``atleast_1d`` stub rejects; ``asarray`` accepts both arms.
        reward_np = np.atleast_1d(np.asarray(reward))
        is_terminal_np = np.atleast_1d(is_terminal)
        value_np = np.atleast_1d(value)
        log_prob_np = np.atleast_1d(log_prob)

        current_action_mask = info.get("action_mask", None)
        agent.rollout_buffer.add(
            obs=obs,
            action=action,
            reward=reward_np,
            done=is_terminal_np,
            value=value_np,
            log_prob=log_prob_np,
            next_obs=next_obs,
            hidden_state=current_hidden_state_for_buffer,
            action_mask=current_action_mask,
        )

        scores += reward_np
        done = is_terminal_np

        if recurrent and np.any(is_terminal_np):
            finished_mask = is_terminal_np.astype(bool)
            initial_hidden_states_for_reset = agent.get_initial_hidden_state(
                agent.num_envs,
            )
            if isinstance(agent.hidden_state, dict):
                for key in agent.hidden_state:
                    reset_states_for_key = initial_hidden_states_for_reset[key][
                        :,
                        finished_mask,
                        :,
                    ]
                    if reset_states_for_key.shape[1] > 0:
                        agent.hidden_state[key][
                            :,
                            finished_mask,
                            :,
                        ] = torch.as_tensor(reset_states_for_key)

        if recurrent:
            current_hidden_state_for_actor = agent.hidden_state

        obs = next_obs
        info = next_info

        for idx, env_done in enumerate(is_terminal_np):
            if env_done:
                completed_episode_scores.append(scores[idx])
                agent.scores.append(scores[idx])
                scores[idx] = 0

    # Calculate last value to compute returns and advantages properly
    with torch.no_grad():
        if recurrent:
            _, _, _, last_value, _ = agent._get_action_and_values(
                agent.preprocess_observation(obs),
                hidden_state=agent.hidden_state,
            )
        else:
            _, _, _, last_value, _ = agent._get_action_and_values(
                agent.preprocess_observation(obs),
            )

        last_value = last_value.cpu().numpy()
        last_done = np.atleast_1d(term)

    agent.rollout_buffer.compute_returns_and_advantages(
        last_value=last_value,
        last_done=last_done,
    )

    return completed_episode_scores, obs, done, scores, info


def collect_rollouts(
    agent: SupportedOnPolicy,
    env: RolloutHarness,
    n_steps: int | None = None,
    **kwargs: Any,
) -> RolloutReturn:
    """Collect rollouts for non-recurrent on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: SingleAgentAlgorithm
    :param env: The environment to collect rollouts from.
    :type env: gym.Env | gym.vector.VectorEnv
    :param n_steps: The number of steps to collect rollouts for.
    :type n_steps: int | None

    :return: The scores for the episodes completed in the rollouts, followed by
        the observation, done flag, scores, and info for the current step.
    :rtype: tuple[list[float], npt.NDArray, npt.NDArray, npt.NDArray, dict[str, Any]]
    """
    return _collect_rollouts(agent, env, n_steps, recurrent=False, **kwargs)


def collect_rollouts_recurrent(
    agent: SupportedOnPolicy,
    env: RolloutHarness,
    n_steps: int | None = None,
    **kwargs: Any,
) -> RolloutReturn:
    """Collect rollouts for recurrent on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: SingleAgentAlgorithm
    :param env: The environment to collect rollouts from.
    :type env: gym.Env | gym.vector.VectorEnv
    :param n_steps: The number of steps to collect rollouts for.
    :type n_steps: int | None

    :return: The scores for the episodes completed in the rollouts, followed by
        the observation, done flag, scores, and info for the current step.
    :rtype: tuple[list[float], npt.NDArray, npt.NDArray, npt.NDArray, dict[str, Any]]
    """
    return _collect_rollouts(agent, env, n_steps, recurrent=True, **kwargs)


def collect_rollouts_llm(
    agent: LLMPPO | LLMREINFORCE | GRPO,
    env: RolloutCollector,
    n_steps: int,
    batch_size: int,
    group_seed: int,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    int,
    int,
    list[torch.Tensor | None] | None,
]:
    """Collect multi-turn rollouts for LLM on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: SupportedOnPolicyLLM
    :param env: Batched rollout environment.
    :type env: RolloutCollector
    :param n_steps: Number of steps (max turns) for the agent to take.
    :type n_steps: int
    :param batch_size: Number of environments to collect rollouts from.
    :type batch_size: int
    :param group_seed: Seed for the group of environments.
    :type group_seed: int
    :return: Episode tensors, masks, turn ids, rewards, counted batch steps,
        updated group seed, and per-trajectory vLLM sampling logprobs (one 1-D
        tensor of generated-token logprobs each, individual entries possibly
        ``None``; ``None`` overall when none were captured this rollout).
    :rtype: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], int, int, list[torch.Tensor | None] | None]
    """
    prompts = env.reset(
        seed=group_seed,
    )

    # Colocated vLLM requires tensor_parallel_size==1, and there is no per-generate
    # DP barrier, so ranks may early-exit independently. Rejoin even when the
    # turn loop raises so a sibling is not left blocked in this wait.
    accelerator = getattr(agent, "accelerator", None)
    try:
        for _turn_idx in range(n_steps):
            if prompts is None:
                break
            if isinstance(agent, GRPO):
                action_result = agent.get_action(
                    prompts,
                    training=True,
                    repeat_prompts=False,
                )
            else:
                action_result = agent.get_action(prompts, training=True)
            prompts = env.step(action_result.token_ids, action_result.sampling_logps)
    finally:
        if accelerator is not None:
            accelerator.wait_for_everyone()

    (
        token_ids_list,
        action_masks_list,
        all_turn_ids,
        all_rewards,
        batch_steps,
        all_sampling_logps,
    ) = env.get_trajectories()
    group_seed = group_seed + batch_size

    return (
        token_ids_list,
        action_masks_list,
        all_turn_ids,
        all_rewards,
        batch_steps,
        group_seed,
        all_sampling_logps,
    )


def collate_llm_rollouts(
    token_ids_list: list[torch.Tensor],
    action_masks_list: list[torch.Tensor],
    all_turn_ids: list[torch.Tensor],
    all_rewards: list[torch.Tensor],
    all_sampling_logps: list[torch.Tensor | None] | None = None,
    *,
    group_size: int,
) -> LLMExperienceBatch:
    """Collate a collected LLM rollout into one grouped experience batch.

    Trajectories from :func:`collect_rollouts_llm` are group-contiguous, so each
    ``group_size`` consecutive trajectories form one prompt group. ``group_size``
    is the number of completions sampled per prompt (e.g. for GRPO);
    ``group_size=1`` (PPO/REINFORCE) makes each group a single trajectory, i.e. a
    plain batch.

    :param token_ids_list: Per-trajectory ``(1, T)`` token-id tensors.
    :type token_ids_list: list[torch.Tensor]
    :param action_masks_list: Per-trajectory ``(1, T - 1)`` boolean masks.
    :type action_masks_list: list[torch.Tensor]
    :param all_turn_ids: Per-trajectory ``(1, T - 1)`` turn-index tensors.
    :type all_turn_ids: list[torch.Tensor]
    :param all_rewards: Per-trajectory ``(max_turns,)`` reward tensors.
    :type all_rewards: list[torch.Tensor]
    :param all_sampling_logps: Per-trajectory vLLM sampling logprobs parallel to
        ``token_ids_list``, or ``None`` when none were captured.
    :type all_sampling_logps: list[torch.Tensor | None] | None
    :param group_size: Number of trajectories per group.
    :type group_size: int
    :return: The collated batch.
    :rtype: LLMExperienceBatch
    """
    n = len(token_ids_list)
    if n % group_size != 0:
        msg = f"Number of trajectories ({n}) must be divisible by group_size ({group_size})."
        raise ValueError(msg)
    logps = all_sampling_logps if all_sampling_logps is not None else [None] * n
    groups = [
        RolloutGroup(
            group_size=group_size,
            trajectories=[
                Trajectory(
                    token_ids=token_ids,
                    action_masks=action_masks,
                    turn_ids=turn_ids,
                    rewards=rewards,
                    sampling_logps=sampling_logps,
                )
                for token_ids, action_masks, turn_ids, rewards, sampling_logps in zip(
                    token_ids_list[sl],
                    action_masks_list[sl],
                    all_turn_ids[sl],
                    all_rewards[sl],
                    logps[sl],
                    strict=True,
                )
            ],
        )
        for sl in (
            slice(g * group_size, (g + 1) * group_size) for g in range(n // group_size)
        )
    ]
    return collate_rollout_groups(groups)
