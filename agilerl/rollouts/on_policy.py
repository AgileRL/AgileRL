"""Functions for collecting rollouts for on-policy algorithms."""

from typing import Any, Callable

import numpy as np
import torch
from gymnasium import spaces

from agilerl.algorithms import PPO, LLMPPO, LLMReinforce, GRPO
from agilerl.networks import StochasticActor
from agilerl.typing import GymEnvType

SupportedOnPolicy = PPO
SupportedOnPolicyLLM = LLMPPO | LLMReinforce | GRPO

def _collect_rollouts(
    agent: SupportedOnPolicy,
    env: GymEnvType,
    n_steps: int | None = None,
    last_obs: np.ndarray | None = None,
    last_done: np.ndarray | None = None,
    last_scores: np.ndarray | None = None,
    last_info: dict[str, Any] | None = None,
    *,
    recurrent: bool,
) -> tuple[list[float], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Collect rollouts for on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: SupportedOnPolicy
    :param env: The environment to collect rollouts from.
    :type env: GymEnvType
    :param n_steps: The number of steps to collect rollouts for. Defaults to agent.learn_step if not provided.
    :type n_steps: int | None
    :param last_obs: The observation to use for the first step. Defaults to None, where the environment is reset.
    :type last_obs: np.ndarray | None
    :param last_done: The done flag to use for the first step. Defaults to None, where the environment is reset.
    :type last_done: np.ndarray | None
    :param last_scores: The scores to use for the first step. Defaults to None, where the environment is reset.
    :type last_scores: np.ndarray | None
    :param last_info: The info for the current step. Defaults to None, where the environment is reset.
    :type last_info: dict[str, Any] | None
    :param recurrent: Whether the agent is recurrent.
    :type recurrent: bool

    :return: The list of scores for the episodes completed in the rollouts
    :rtype: list[float]
    :return: The observation, done flag, scores, and info for the current step.
    :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
    """
    if not agent.use_rollout_buffer:
        msg = "collect_rollouts can only be used when use_rollout_buffer=True"
        raise RuntimeError(
            msg,
        )

    if (
        last_obs is None
        and last_done is None
        and last_scores is None
        and last_info is None
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

        # Get action, statistics and (maybe) recurrent hidden state from agent
        if recurrent:
            action, log_prob, _, value, next_hidden_for_actor = agent.get_action(
                obs,
                action_mask=info.get("action_mask", None),
                hidden_state=current_hidden_state_for_actor,
            )
            agent.hidden_state = next_hidden_for_actor
        else:
            action, log_prob, _, value = agent.get_action(
                obs,
                action_mask=info.get("action_mask", None),
            )

        # Clip action to action space
        policy = getattr(agent, agent.registry.policy())
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

        reward_np = np.atleast_1d(reward)
        is_terminal_np = np.atleast_1d(is_terminal)
        value_np = np.atleast_1d(value)
        log_prob_np = np.atleast_1d(log_prob)

        agent.rollout_buffer.add(
            obs=obs,
            action=action,
            reward=reward_np,
            done=is_terminal_np,
            value=value_np,
            log_prob=log_prob_np,
            next_obs=next_obs,
            hidden_state=current_hidden_state_for_buffer,
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
                        ] = reset_states_for_key

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
    env: GymEnvType,
    n_steps: int | None = None,
    **kwargs,
) -> list[float]:
    """Collect rollouts for non-recurrent on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: RLAlgorithm
    :param env: The environment to collect rollouts from.
    :type env: GymEnvType
    :param n_steps: The number of steps to collect rollouts for.
    :type n_steps: int | None

    :return: The list of scores for the episodes completed in the rollouts
    :rtype: list[float]
    """
    return _collect_rollouts(agent, env, n_steps, recurrent=False, **kwargs)


def collect_rollouts_recurrent(
    agent: SupportedOnPolicy,
    env: GymEnvType,
    n_steps: int | None = None,
    **kwargs,
) -> list[float]:
    """Collect rollouts for recurrent on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: RLAlgorithm
    :param env: The environment to collect rollouts from.
    :type env: GymEnvType
    :param n_steps: The number of steps to collect rollouts for.
    :type n_steps: int | None

    :return: The list of scores for the episodes completed in the rollouts
    :rtype: list[float]
    """
    return _collect_rollouts(agent, env, n_steps, recurrent=True, **kwargs)

def collect_rollouts_llm(
    agent: SupportedOnPolicyLLM,
    env: GymEnvType,
    n_steps: int | None = None,
    **kwargs,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    int,
    int,
]:
    """Collect multi-turn rollouts for LLM on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: SupportedOnPolicyLLM
    :param env: The environment to collect rollouts from.
    :type env: GymEnvType
    :param n_steps: Number of outer batch episodes to collect.
    :type n_steps: int | None
    :param kwargs: Extra rollout controls (``max_turns``, ``group_size``,
        ``group_seed``, ``total_steps``, ``env_factory``).
    :type kwargs: dict[str, Any]

    :return: Episode tensors, masks, turn ids, rewards, counted batch steps,
        and updated group seed.
    :rtype: tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], int, int]
    """
    batch_size = n_steps
    if batch_size is None:
        msg = "collect_rollouts_llm requires n_steps (batch size) to be provided."
        raise ValueError(msg)

    max_turns = kwargs.get("max_turns")
    group_size = kwargs.get("group_size", getattr(agent, "group_size", 1))
    group_seed = kwargs.get("group_seed")
    env_factory: Callable[[], GymEnvType] | None = kwargs.get("env_factory")

    if max_turns is None:
        msg = "collect_rollouts_llm requires max_turns in kwargs."
        raise ValueError(msg)
    if group_seed is None:
        msg = "collect_rollouts_llm requires group_seed in kwargs."
        raise ValueError(msg)

    if group_size > 1 and env_factory is None:
        msg = (
            "collect_rollouts_llm requires env_factory when group_size > 1 so each "
            "trajectory can have an independent environment state."
        )
        raise ValueError(msg)

    def _new_env() -> GymEnvType:
        return env if env_factory is None else env_factory()

    trajectories: list[dict[str, Any]] = []
    seed_base = group_seed
    for group_idx in range(batch_size):
        seed = seed_base + group_idx
        for member_idx in range(group_size):
            env_i = _new_env()
            prompt_dict, _info = env_i.reset(seed=seed)
            sw_ml = getattr(env_i, "_sw_max_model_len", None)
            if sw_ml is not None:
                assert sw_ml == agent.max_model_len, (
                    f"env max_model_len ({sw_ml}) != agent.max_model_len "
                    f"({agent.max_model_len})"
                )
            trajectories.append(
                {
                    "group_idx": group_idx,
                    "member_idx": member_idx,
                    "env": env_i,
                    "prompt": prompt_dict,
                    "done": False,
                },
            )

    for _turn_idx in range(max_turns):
        active = [traj for traj in trajectories if not traj["done"]]
        if not active:
            break
        active.sort(key=lambda t: (t["group_idx"], t["member_idx"]))
        prompts = [traj["prompt"] for traj in active]
        if isinstance(agent, GRPO):
            completion_ids, _ = agent.get_action(
                prompts,
                training=True,
                repeat_prompts=False,
            )
        else:
            completion_ids, _ = agent.get_action(prompts, training=True)

        for traj, completion in zip(active, completion_ids, strict=False):
            full_completion = completion
            if full_completion.dim() == 1:
                full_completion = full_completion.unsqueeze(0)
            next_prompt, _reward, terminated, truncated, _info = traj["env"].step(
                full_completion,
            )
            traj["done"] = bool(terminated or truncated)
            if not traj["done"]:
                traj["prompt"] = next_prompt

    completion_ids_list: list[torch.Tensor] = []
    action_masks_list: list[torch.Tensor] = []
    all_turn_ids: list[torch.Tensor] = []
    all_rewards: list[torch.Tensor] = []
    batch_steps = 0
    trajectories.sort(key=lambda t: (t["group_idx"], t["member_idx"]))
    for traj in trajectories:
        ep_ids, action_mask, turn_ids, turn_rewards_t = traj["env"].get_episode_data()
        completion_ids_list.append(ep_ids)
        action_masks_list.append(action_mask)
        all_turn_ids.append(turn_ids)
        all_rewards.append(turn_rewards_t)
        batch_steps += len(getattr(traj["env"], "turn_boundaries", []))

    group_seed = group_seed + batch_size

    return (
        completion_ids_list,
        action_masks_list,
        all_turn_ids,
        all_rewards,
        batch_steps,
        group_seed,
    )
