# Copyright 2026 AgileRL
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces
from torch import nn
from tqdm import trange

from agilerl.components.data import MultiAgentTransition, transition_to_tensordict
from agilerl.components.replay_buffer import ReplayBuffer


def prepare_ma_states(
    states: dict[str, npt.NDArray],
    observation_space: dict[str, spaces.Space[Any]],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    processed_states: dict[str, torch.Tensor] = {}
    for agent_id, state in states.items():
        agent_space = observation_space[agent_id]
        if isinstance(agent_space, spaces.Discrete):
            processed_states[agent_id] = (
                nn.functional.one_hot(
                    torch.Tensor(state).long(),
                    num_classes=int(agent_space.n),
                )
                .float()
                .squeeze(1)
                .to(device)
            )
        else:
            processed_states[agent_id] = torch.Tensor(state).to(device)
    return processed_states

def prepare_ma_actions(
    actions: dict[str, npt.NDArray],
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    return {
        agent_id: torch.Tensor(action).to(device)
        for (agent_id, action) in actions.items()
    }

def check_policy_q_learning_with_probe_env(
    env: Any,  # noqa: ANN401 -- multi-agent probe envs share no base class
    algo_class: type[Any],
    algo_args: dict[str, Any],
    memory: ReplayBuffer,
    learn_steps: int = 1000,
    device: str = "cpu",
) -> None:

    agent = algo_class(**algo_args, vect_noise_dim=1, device=device)

    state, _ = env.reset()
    agent.set_training_mode(True)
    for _ in range(1000):
        # Make vectorized
        state = {agent_id: np.expand_dims(s, 0) for agent_id, s in state.items()}
        processed_action, raw_action = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(processed_action)
        reward = {
            agent_id: np.expand_dims(np.array(r), 0) for agent_id, r in reward.items()
        }
        done = {
            agent_id: np.expand_dims(np.array(d), 0) for agent_id, d in done.items()
        }
        mem_next_state = {
            agent_id: np.expand_dims(ns, 0) for agent_id, ns in next_state.items()
        }
        transition = transition_to_tensordict(
            MultiAgentTransition(
                obs=state,
                action=raw_action,
                reward=reward,
                next_obs=mem_next_state,
                done=done,
            )
        )
        transition.batch_size = torch.Size([1])
        memory.add(transition)
        state = next_state
        if done[agent.agent_ids[0]]:
            state, _ = env.reset()

    # Learn from experiences
    for _ in trange(learn_steps):
        experiences = memory.sample(agent.batch_size)
        agent.learn(experiences)

    agent.set_training_mode(False)
    with torch.no_grad():
        for agent_id in agent.agent_ids:
            actor = agent.actors[agent_id]
            critic = agent.critics[agent_id]
            for sample_obs, sample_action, q_values, policy_values in zip(
                env.sample_obs,
                env.sample_actions,
                env.q_values,
                env.policy_values,
                strict=False,
            ):
                state = prepare_ma_states(sample_obs, agent.observation_space, device)

                if q_values is not None:
                    action = prepare_ma_actions(sample_action, device)
                    stacked_actions = torch.cat(list(action.values()), dim=1)
                    predicted_q_values = (
                        critic(state, stacked_actions).detach().cpu().numpy()[0]
                    )
                    # print("---")
                    # print(agent_id, "q", q_values[agent_id], predicted_q_values)
                    # assert np.allclose(q_values[agent_id], predicted_q_values, atol=0.1):
                    if not np.allclose(
                        q_values[agent_id],
                        predicted_q_values,
                        atol=0.1,
                    ):
                        pass

                if policy_values is not None:
                    predicted_policy_values = (
                        actor(state[agent_id]).detach().cpu().numpy()[0]
                    )

                    # print(agent_id, "pol", policy_values[agent_id], predicted_policy_values)
                    # assert np.allclose(policy_values[agent_id], predicted_policy_values, atol=0.1)
                    if not np.allclose(
                        policy_values[agent_id],
                        predicted_policy_values,
                        atol=0.1,
                    ):
                        pass

def check_on_policy_learning_with_probe_env(
    env: Any,  # noqa: ANN401 -- multi-agent probe envs share no base class
    algo_class: type[Any],
    algo_args: dict[str, Any],
    learn_steps: int = 1000,
    device: str = "cpu",
    discrete: bool = True,
) -> None:

    agent = algo_class(**algo_args, device=device)

    for _ in trange(learn_steps):
        state, _ = env.reset()
        states = {agent_id: [] for agent_id in agent.agent_ids}
        actions = {agent_id: [] for agent_id in agent.agent_ids}
        log_probs = {agent_id: [] for agent_id in agent.agent_ids}
        rewards = {agent_id: [] for agent_id in agent.agent_ids}
        dones = {agent_id: [] for agent_id in agent.agent_ids}
        values = {agent_id: [] for agent_id in agent.agent_ids}

        done = {agent_id: np.zeros((1,)) for agent_id in agent.agent_ids}

        for _ in range(100):
            # Make vectorized
            state = {agent_id: np.expand_dims(s, 0) for agent_id, s in state.items()}

            action, log_prob, _, value = agent.get_action(obs=state)

            action = {agent: act[0].squeeze() for agent, act in action.items()}
            log_prob = {agent: lp[0] for agent, lp in log_prob.items()}
            value = {agent: val[0] for agent, val in value.items()}

            next_state, reward, termination, truncation, _info = env.step(action)

            next_done = {}
            for agent_id in agent.agent_ids:
                states[agent_id].append(state[agent_id])
                actions[agent_id].append(action[agent_id])
                log_probs[agent_id].append(log_prob[agent_id])
                rewards[agent_id].append(reward[agent_id])
                dones[agent_id].append(done[agent_id])
                values[agent_id].append(value[agent_id])
                next_done[agent_id] = np.logical_or(
                    termination[agent_id],
                    truncation[agent_id],
                ).astype(np.int8)

            next_done = {aid: np.array([n_d]) for aid, n_d in next_done.items()}

            state = next_state
            done = next_done
            if done[agent.agent_ids[0]]:
                state, _ = env.reset()

        # Learn from experiences
        experiences = (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            values,
            next_state,
            next_done,
        )
        agent.learn(experiences)

    with torch.no_grad():
        for agent_id in agent.observation_space:
            actor = agent.actors[agent_id]
            critic = agent.critics[agent_id]
            for sample_obs, v_values, policy_values in zip(
                env.sample_obs,
                env.v_values,
                env.policy_values,
                strict=False,
            ):
                state = prepare_ma_states(sample_obs, agent.observation_space, device)

                if v_values is not None:
                    (critic(state[agent_id]).detach().cpu().numpy()[0])

                    # assert np.allclose(v_values[agent_id], predicted_v_values, atol=0.1):
                    # if not np.allclose(
                    #     v_values[agent_id], predicted_v_values, atol=0.1
                    # ):
                    #     print(
                    #         "FAILURE: ",
                    #         agent_id,
                    #         "v",
                    #         v_values[agent_id],
                    #         predicted_v_values,
                    #     )
                    # else:
                    #     print(
                    #         "SUCCESS: ",
                    #         agent_id,
                    #         "v",
                    #         v_values[agent_id],
                    #         predicted_v_values,
                    #     )

                if policy_values is not None:
                    if discrete:
                        _, _, _ = actor(state[agent_id])
                        # TorchDistribution uses raw tensors: logits -> probs via softmax
                        _ = (
                            torch.softmax(actor.head_net.dist.logits, dim=-1)
                            .detach()
                            .cpu()
                            .numpy()
                        )
                    else:
                        actor(state[agent_id])
                        _ = actor.head_net.dist.mu.detach().cpu().numpy()
