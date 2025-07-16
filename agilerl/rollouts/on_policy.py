"""Functions for collecting rollouts for on-policy algorithms."""

from typing import List, Optional

import numpy as np
import torch
from gymnasium import spaces

from agilerl.algorithms import PPO
from agilerl.networks import StochasticActor
from agilerl.typing import GymEnvType

SupportedOnPolicy = PPO


def _collect_rollouts(
    agent: SupportedOnPolicy,
    env: GymEnvType,
    n_steps: Optional[int] = None,
    *,
    recurrent: bool,
) -> List[float]:
    if not getattr(agent, "use_rollout_buffer", False):
        raise RuntimeError(
            "collect_rollouts can only be used when use_rollout_buffer=True"
        )

    n_steps = n_steps or agent.learn_step
    agent.rollout_buffer.reset()

    obs, info = env.reset()

    agent.hidden_state = (
        agent.get_initial_hidden_state(agent.num_envs) if recurrent else None
    )
    current_hidden_state_for_actor = agent.hidden_state

    completed_episode_scores = []
    scores = np.zeros(agent.num_envs)
    for _ in range(n_steps):
        current_hidden_state_for_buffer = current_hidden_state_for_actor

        if recurrent:
            action, log_prob, _, value, next_hidden_for_actor = agent.get_action(
                obs,
                action_mask=info.get("action_mask", None),
                hidden_state=current_hidden_state_for_actor,
            )
            agent.hidden_state = next_hidden_for_actor
        else:
            action, log_prob, _, value = agent.get_action(
                obs, action_mask=info.get("action_mask", None)
            )

        # Clip action to action space
        policy = getattr(agent, agent.registry.policy())
        if isinstance(policy, StochasticActor) and isinstance(
            agent.action_space, spaces.Box
        ):
            if policy.squash_output:
                clipped_action = policy.scale_action(action)
            else:
                clipped_action = np.clip(
                    action,
                    agent.action_space.low,
                    agent.action_space.high,
                )
        else:
            clipped_action = action

        next_obs, reward, done, truncated, next_info = env.step(clipped_action)

        if isinstance(done, (list, np.ndarray)):
            is_terminal = (
                np.logical_or(done, truncated)
                if isinstance(truncated, (list, np.ndarray))
                else done
            )
        else:
            is_terminal = done or truncated

        reward_np = np.atleast_1d(reward)
        is_terminal_np = np.atleast_1d(is_terminal)
        value_np = np.atleast_1d(value)
        log_prob_np = np.atleast_1d(log_prob)

        scores += reward_np

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

        if recurrent and np.any(is_terminal_np):
            finished_mask = is_terminal_np.astype(bool)
            initial_hidden_states_for_reset = agent.get_initial_hidden_state(
                agent.num_envs
            )
            if isinstance(agent.hidden_state, dict):
                for key in agent.hidden_state:
                    reset_states_for_key = initial_hidden_states_for_reset[key][
                        :, finished_mask, :
                    ]
                    if reset_states_for_key.shape[1] > 0:
                        agent.hidden_state[key][
                            :, finished_mask, :
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
                agent.preprocess_observation(obs)
            )

        last_value = last_value.cpu().numpy()
        last_done = np.atleast_1d(done)

    agent.rollout_buffer.compute_returns_and_advantages(
        last_value=last_value, last_done=last_done
    )

    return completed_episode_scores


def collect_rollouts(
    agent: SupportedOnPolicy, env: GymEnvType, n_steps: Optional[int] = None
) -> List[float]:
    """Collect rollouts for non-recurrent on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: RLAlgorithm
    :param env: The environment to collect rollouts from.
    :type env: GymEnvType
    :param n_steps: The number of steps to collect rollouts for.
    :type n_steps: Optional[int]

    :return: The list of scores for the episodes completed in the rollouts
    :rtype: List[float]
    """

    return _collect_rollouts(agent, env, n_steps, recurrent=False)


def collect_rollouts_recurrent(
    agent: SupportedOnPolicy, env: GymEnvType, n_steps: Optional[int] = None
) -> List[float]:
    """Collect rollouts for recurrent on-policy algorithms.

    :param agent: The agent to collect rollouts for.
    :type agent: RLAlgorithm
    :param env: The environment to collect rollouts from.
    :type env: GymEnvType
    :param n_steps: The number of steps to collect rollouts for.
    :type n_steps: Optional[int]

    :return: The list of scores for the episodes completed in the rollouts
    :rtype: List[float]
    """

    return _collect_rollouts(agent, env, n_steps, recurrent=True)
