"""Functions for collecting rollouts for on-policy algorithms."""

from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from jaxtyping import Bool, Float, Int, Shaped

from agilerl import HAS_LLM_DEPENDENCIES
from agilerl.algorithms import PPO
from agilerl.algorithms.ppo import PolicyActionArray
from agilerl.networks import StochasticActor
from agilerl.typing import EnvDoneArray, EnvScoreArray, LogProbArray, ValueArray

if TYPE_CHECKING or HAS_LLM_DEPENDENCIES:
    from agilerl.algorithms import GRPO, LLMPPO, LLMREINFORCE
    from agilerl.llm_envs import SyncMultiTurnVecEnv

SupportedOnPolicy = PPO
RolloutEnv = gym.Env | gym.vector.VectorEnv

# Environment observation. Dict and Tuple observation spaces yield a dict or tuple
# of these leaves rather than a single array. The leading axis counts envs only for
# a vectorized env; a plain ``gym.Env`` hands back one unbatched observation, and a
# plain ``gym.Env`` over a ``Discrete`` space hands back a bare ``int``.
BatchedObsLeaf = Shaped[npt.NDArray, "_num_envs ..."]
BatchedObsArray = (
    BatchedObsLeaf | dict[str, BatchedObsLeaf] | tuple[BatchedObsLeaf, ...] | Number
)

# Per-env done flag carried between rollouts. Its dtype is bool once a step has
# been taken and float64 on the freshly reset flags a caller may start from.
LastDoneArray = Shaped[npt.NDArray, " num_envs"]

# Episode scores completed this rollout, then the observation, done flag, scores
# and info to resume from. The done flag is sized by the environment and the score
# row by ``agent.num_envs``, which agree only for a vectorized env.
OnPolicyRolloutReturn = tuple[
    list[float],
    BatchedObsArray,
    Shaped[npt.NDArray, " _num_envs"],
    Float[npt.NDArray[np.float64], " _num_envs"],
    dict[str, Any],
]


def _collect_rollouts(
    agent: SupportedOnPolicy,
    env: RolloutEnv,
    n_steps: int | None = None,
    last_obs: BatchedObsArray | None = None,
    last_done: LastDoneArray | None = None,
    last_scores: EnvScoreArray | None = None,
    last_info: dict[str, Any] | None = None,
    *,
    recurrent: bool,
) -> OnPolicyRolloutReturn:
    """Collect rollouts for on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: SupportedOnPolicy
    :param env: The environment to collect rollouts from.
    :type env: gym.Env | gym.vector.VectorEnv
    :param n_steps: The number of steps to collect rollouts for. Defaults to agent.learn_step if not provided.
    :type n_steps: int | None
    :param last_obs: The observation to use for the first step. Defaults to None, where the environment is reset.
    :type last_obs: BatchedObsArray | None
    :param last_done: The done flag to use for the first step. Defaults to None, where the environment is reset.
    :type last_done: LastDoneArray | None
    :param last_scores: The scores to use for the first step. Defaults to None, where the environment is reset.
    :type last_scores: EnvScoreArray | None
    :param last_info: The info for the current step. Defaults to None, where the environment is reset.
    :type last_info: dict[str, Any] | None
    :param recurrent: Whether the agent is recurrent.
    :type recurrent: bool

    :return: The scores for the episodes completed in the rollouts, followed by
        the observation, done flag, scores, and info for the current step.
    :rtype: OnPolicyRolloutReturn
    """
    if (
        last_obs is None
        or last_done is None
        or last_scores is None
        or last_info is None
    ):
        obs, info = env.reset()
        scores: EnvScoreArray = np.zeros(agent.num_envs)
        done: Shaped[npt.NDArray, " num_envs"] = np.zeros(agent.num_envs)
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

    completed_episode_scores: list[float] = []
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
            clipped_action: PolicyActionArray = np.clip(
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
        reward_np: Float[npt.NDArray[np.float64], " num_envs"] = np.atleast_1d(
            np.asarray(reward),
        )
        is_terminal_np: EnvDoneArray = np.atleast_1d(is_terminal)
        value_np: ValueArray = np.atleast_1d(value)
        log_prob_np: LogProbArray = np.atleast_1d(log_prob)

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
            finished_mask: Shaped[npt.NDArray, " num_envs"] = is_terminal_np.astype(
                bool,
            )
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
    env: RolloutEnv,
    n_steps: int | None = None,
    **kwargs: Any,
) -> OnPolicyRolloutReturn:
    """Collect rollouts for non-recurrent on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: RLAlgorithm
    :param env: The environment to collect rollouts from.
    :type env: gym.Env | gym.vector.VectorEnv
    :param n_steps: The number of steps to collect rollouts for.
    :type n_steps: int | None

    :return: The scores for the episodes completed in the rollouts, followed by
        the observation, done flag, scores, and info for the current step.
    :rtype: OnPolicyRolloutReturn
    """
    return _collect_rollouts(agent, env, n_steps, recurrent=False, **kwargs)


def collect_rollouts_recurrent(
    agent: SupportedOnPolicy,
    env: RolloutEnv,
    n_steps: int | None = None,
    **kwargs: Any,
) -> OnPolicyRolloutReturn:
    """Collect rollouts for recurrent on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: RLAlgorithm
    :param env: The environment to collect rollouts from.
    :type env: gym.Env | gym.vector.VectorEnv
    :param n_steps: The number of steps to collect rollouts for.
    :type n_steps: int | None

    :return: The scores for the episodes completed in the rollouts, followed by
        the observation, done flag, scores, and info for the current step.
    :rtype: OnPolicyRolloutReturn
    """
    return _collect_rollouts(agent, env, n_steps, recurrent=True, **kwargs)


def collect_rollouts_llm(
    agent: LLMPPO | LLMREINFORCE | GRPO,
    env: SyncMultiTurnVecEnv,
    n_steps: int,
    batch_size: int,
    group_seed: int,
    **kwargs: Any,
) -> tuple[
    list[Int[torch.Tensor, "1 seq_len"]],
    list[Bool[torch.Tensor, "1 mask_len"]],
    list[Int[torch.Tensor, "1 mask_len"]],
    list[Float[torch.Tensor, " num_turns"]],
    int,
    int,
    list[Float[torch.Tensor, " gen_len"] | None] | None,
]:
    """Collect multi-turn rollouts for LLM on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: SupportedOnPolicyLLM
    :param env: Synchronous vectorized multi-turn environment.
    :type env: SyncGemVecEnv
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
    :rtype: tuple[list[Int[torch.Tensor, "1 seq_len"]], list[Bool[torch.Tensor, "1 mask_len"]], list[Int[torch.Tensor, "1 mask_len"]], list[Float[torch.Tensor, " num_turns"]], int, int, list[Float[torch.Tensor, " gen_len"] | None] | None]
    """
    prompts = env.reset(
        seed=group_seed,
    )

    # Colocated vLLM requires tensor_parallel_size==1, and the per-generate DP
    # barrier was removed, so ranks may early-exit independently. Rejoin once
    # at the end before train_llm's cross-rank T align / learn.
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
        prompts = env.step(action_result.completion_ids, action_result.sampling_logps)

    accelerator = getattr(agent, "accelerator", None)
    if accelerator is not None:
        accelerator.wait_for_everyone()

    (
        completion_ids_list,
        action_masks_list,
        all_turn_ids,
        all_rewards,
        batch_steps,
        all_sampling_logps,
    ) = env.get_trajectories()
    group_seed = group_seed + batch_size

    return (
        completion_ids_list,
        action_masks_list,
        all_turn_ids,
        all_rewards,
        batch_steps,
        group_seed,
        all_sampling_logps,
    )
