import numpy as np
import pytest
import torch

from agilerl.algorithms.maddpg import MADDPG
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.utils.probe_envs_ma import (
    ConstantRewardContActionsEnv,
    ConstantRewardContActionsImageEnv,
    ConstantRewardEnv,
    ConstantRewardImageEnv,
    DiscountedRewardContActionsEnv,
    DiscountedRewardContActionsImageEnv,
    DiscountedRewardEnv,
    DiscountedRewardImageEnv,
    FixedObsPolicyContActionsEnv,
    FixedObsPolicyContActionsImageEnv,
    FixedObsPolicyEnv,
    FixedObsPolicyImageEnv,
    MultiPolicyEnv,
    MultiPolicyImageEnv,
    ObsDependentRewardContActionsEnv,
    ObsDependentRewardContActionsImageEnv,
    ObsDependentRewardEnv,
    ObsDependentRewardImageEnv,
    PolicyContActionsEnv,
    PolicyContActionsImageEnv,
    PolicyEnv,
    PolicyImageEnv,
    check_policy_q_learning_with_probe_env,
)


@pytest.mark.parametrize(
    "env_class, exp_states, exp_rewards, exp_terminateds, exp_truncateds, exp_infos",
    [
        (ConstantRewardEnv, [0, 0], [1, 0], [True, True], [False, False], {}),
        (ConstantRewardImageEnv, [0, 0], [1, 0], [True, True], [False, False], {}),
        (
            ConstantRewardContActionsEnv,
            [0, 0],
            [1, 0],
            [True, True],
            [False, False],
            {},
        ),
        (
            ConstantRewardContActionsImageEnv,
            [0, 0],
            [1, 0],
            [True, True],
            [False, False],
            {},
        ),
    ],
)
def test_constant_reward_envs(
    env_class, exp_states, exp_rewards, exp_terminateds, exp_truncateds, exp_infos
):
    env = env_class()
    states, infos = env.reset()
    for _ in range(20):
        actions = {
            env.agents[0]: env.action_space[env.agents[0]].sample(),
            env.agents[1]: env.action_space[env.agents[1]].sample(),
        }
        states, rewards, terminateds, truncateds, infos = env.step(actions)

        for (
            state,
            reward,
            terminated,
            truncated,
            info,
            exp_state,
            exp_reward,
            exp_terminated,
            exp_truncated,
            exp_info,
        ) in zip(
            list(states.values()),
            list(rewards.values()),
            list(terminateds.values()),
            list(truncateds.values()),
            list(infos.values()),
            exp_states,
            exp_rewards,
            exp_terminateds,
            exp_truncateds,
            exp_infos,
        ):

            assert int(np.mean(np.array(state))) == exp_state
            assert reward == exp_reward
            assert terminated == exp_terminated
            assert truncated == exp_truncated
            assert info == exp_info

        if terminateds[env.agents[0]] or truncateds[env.agents[0]]:
            states, infos = env.reset()


@pytest.mark.parametrize(
    "env_class, exp_states, exp_rewards, exp_terminateds, exp_truncateds, exp_infos",
    [
        (ObsDependentRewardEnv, [0, 0], [1, 0], [True, True], [False, False], {}),
        (ObsDependentRewardEnv, [1, 1], [0, 1], [True, True], [False, False], {}),
        (ObsDependentRewardImageEnv, [0, 0], [1, 0], [True, True], [False, False], {}),
        (ObsDependentRewardImageEnv, [1, 1], [0, 1], [True, True], [False, False], {}),
        (
            ObsDependentRewardContActionsEnv,
            [0, 0],
            [1, 0],
            [True, True],
            [False, False],
            {},
        ),
        (
            ObsDependentRewardContActionsEnv,
            [1, 1],
            [0, 1],
            [True, True],
            [False, False],
            {},
        ),
        (
            ObsDependentRewardContActionsImageEnv,
            [0, 0],
            [1, 0],
            [True, True],
            [False, False],
            {},
        ),
        (
            ObsDependentRewardContActionsImageEnv,
            [1, 1],
            [0, 1],
            [True, True],
            [False, False],
            {},
        ),
    ],
)
def test_observation_dependent_reward_envs(
    env_class, exp_states, exp_rewards, exp_terminateds, exp_truncateds, exp_infos
):
    env = env_class()
    states, info = env.reset()
    for _ in range(20):
        actions = {
            env.agents[0]: env.action_space[env.agents[0]].sample(),
            env.agents[1]: env.action_space[env.agents[1]].sample(),
        }
        states, rewards, terminateds, truncateds, infos = env.step(actions)

        for (
            state,
            reward,
            terminated,
            truncated,
            info,
            exp_state,
            exp_reward,
            exp_terminated,
            exp_truncated,
            exp_info,
        ) in zip(
            list(states.values()),
            list(rewards.values()),
            list(terminateds.values()),
            list(truncateds.values()),
            list(infos.values()),
            exp_states,
            exp_rewards,
            exp_terminateds,
            exp_truncateds,
            exp_infos,
        ):
            if int(np.mean(np.array(state))) == exp_state:
                assert reward == exp_reward
            assert terminated == exp_terminated
            assert truncated == exp_truncated
            assert info == exp_info

        if terminateds[env.agents[0]] or truncateds[env.agents[0]]:
            states, infos = env.reset()


@pytest.mark.parametrize(
    "env_class, exp_states, exp_rewards, exp_truncateds, exp_infos",
    [
        (DiscountedRewardEnv, [0, 0], [0, 0], [False, False], {}),
        (DiscountedRewardEnv, [1, 1], [1, 0.5], [False, False], {}),
        (DiscountedRewardImageEnv, [0, 0], [0, 0], [False, False], {}),
        (DiscountedRewardImageEnv, [1, 1], [1, 0.5], [False, False], {}),
        (DiscountedRewardContActionsEnv, [0, 0], [0, 0], [False, False], {}),
        (DiscountedRewardContActionsEnv, [1, 1], [1, 0.5], [False, False], {}),
        (DiscountedRewardContActionsImageEnv, [0, 0], [0, 0], [False, False], {}),
        (DiscountedRewardContActionsImageEnv, [1, 1], [1, 0.5], [False, False], {}),
    ],
)
def test_discounted_reward_envs(
    env_class, exp_states, exp_rewards, exp_truncateds, exp_infos
):
    env = env_class()
    states, info = env.reset()
    for _ in range(20):
        actions = {
            env.agents[0]: env.action_space[env.agents[0]].sample(),
            env.agents[1]: env.action_space[env.agents[1]].sample(),
        }
        next_states, rewards, terminateds, truncateds, infos = env.step(actions)

        for (
            state,
            reward,
            terminated,
            truncated,
            info,
            exp_state,
            exp_reward,
            exp_truncated,
            exp_info,
        ) in zip(
            list(states.values()),
            list(rewards.values()),
            list(terminateds.values()),
            list(truncateds.values()),
            list(infos.values()),
            exp_states,
            exp_rewards,
            exp_truncateds,
            exp_infos,
        ):
            if int(np.mean(np.array(state))) == exp_state:
                assert reward == exp_reward
            assert terminated == int(np.mean(np.array(state)))
            assert truncated == exp_truncated
            assert info == exp_info

        states = next_states

        if terminateds[env.agents[0]] or truncateds[env.agents[0]]:
            states, infos = env.reset()


@pytest.mark.parametrize(
    "env_class, exp_states, exp_actions, exp_rewards, exp_terminateds, exp_truncateds, exp_infos",
    [
        (FixedObsPolicyEnv, [0, 0], [0, 0], [1, -1], [True, True], [False, False], {}),
        (FixedObsPolicyEnv, [0, 0], [1, 1], [-1, 1], [True, True], [False, False], {}),
        (
            FixedObsPolicyImageEnv,
            [0, 0],
            [0, 0],
            [1, -1],
            [True, True],
            [False, False],
            {},
        ),
        (
            FixedObsPolicyImageEnv,
            [0, 0],
            [1, 1],
            [-1, 1],
            [True, True],
            [False, False],
            {},
        ),
    ],
)
def test_discrete_actions_fixed_observation_policy_reward_envs(
    env_class,
    exp_states,
    exp_actions,
    exp_rewards,
    exp_terminateds,
    exp_truncateds,
    exp_infos,
):
    env = env_class()
    states, info = env.reset()
    for _ in range(20):
        actions = {
            env.agents[0]: env.action_space[env.agents[0]].sample(),
            env.agents[1]: env.action_space[env.agents[1]].sample(),
        }
        states, rewards, terminateds, truncateds, infos = env.step(actions)

        for (
            state,
            action,
            reward,
            terminated,
            truncated,
            info,
            exp_state,
            exp_action,
            exp_reward,
            exp_terminated,
            exp_truncated,
            exp_info,
        ) in zip(
            list(states.values()),
            list(actions.values()),
            list(rewards.values()),
            list(terminateds.values()),
            list(truncateds.values()),
            list(infos.values()),
            exp_states,
            exp_actions,
            exp_rewards,
            exp_terminateds,
            exp_truncateds,
            exp_infos,
        ):
            assert int(np.mean(np.array(state))) == exp_state
            if action == exp_action:
                assert reward == exp_reward
            assert terminated == exp_terminated
            assert truncated == exp_truncated
            assert info == exp_info

        if terminateds[env.agents[0]] or truncateds[env.agents[0]]:
            states, infos = env.reset()


@pytest.mark.parametrize(
    "env_class, exp_states, exp_actions, exp_terminateds, exp_truncateds, exp_infos",
    [
        (
            FixedObsPolicyContActionsEnv,
            [0, 0],
            [1, 0],
            [True, True],
            [False, False],
            {},
        ),
        (
            FixedObsPolicyContActionsImageEnv,
            [0, 0],
            [1, 0],
            [True, True],
            [False, False],
            {},
        ),
    ],
)
def test_continuous_actions_fixed_observation_policy_reward_envs(
    env_class, exp_states, exp_actions, exp_terminateds, exp_truncateds, exp_infos
):
    env = env_class()
    states, info = env.reset()
    for _ in range(20):
        actions = {
            env.agents[0]: env.action_space[env.agents[0]].sample(),
            env.agents[1]: env.action_space[env.agents[1]].sample(),
        }
        states, rewards, terminateds, truncateds, infos = env.step(actions)

        for (
            state,
            action,
            reward,
            terminated,
            truncated,
            info,
            exp_state,
            exp_action,
            exp_terminated,
            exp_truncated,
            exp_info,
        ) in zip(
            list(states.values()),
            list(actions.values()),
            list(rewards.values()),
            list(terminateds.values()),
            list(truncateds.values()),
            list(infos.values()),
            exp_states,
            exp_actions,
            exp_terminateds,
            exp_truncateds,
            exp_infos,
        ):
            assert int(np.mean(np.array(state))) == exp_state
            assert reward == -((exp_action - action[0]) ** 2)
            assert terminated == exp_terminated
            assert truncated == exp_truncated
            assert info == exp_info

        if terminateds[env.agents[0]] or truncateds[env.agents[0]]:
            states, infos = env.reset()


@pytest.mark.parametrize(
    "env_class, same_rewards, diff_rewards, exp_terminateds, exp_truncateds, exp_infos",
    [
        (PolicyEnv, [1, 0], [0, 1], [True, True], [False, False], {}),
        (PolicyImageEnv, [1, 0], [0, 1], [True, True], [False, False], {}),
    ],
)
def test_discrete_actions_policy_envs(
    env_class, same_rewards, diff_rewards, exp_terminateds, exp_truncateds, exp_infos
):
    env = env_class()
    states, info = env.reset()
    for _ in range(20):
        actions = {
            env.agents[0]: env.action_space[env.agents[0]].sample(),
            env.agents[1]: env.action_space[env.agents[1]].sample(),
        }
        states, rewards, terminateds, truncateds, infos = env.step(actions)

        for (
            state,
            action,
            reward,
            terminated,
            truncated,
            info,
            same_reward,
            diff_reward,
            exp_terminated,
            exp_truncated,
            exp_info,
        ) in zip(
            list(states.values()),
            list(actions.values()),
            list(rewards.values()),
            list(terminateds.values()),
            list(truncateds.values()),
            list(infos.values()),
            same_rewards,
            diff_rewards,
            exp_terminateds,
            exp_truncateds,
            exp_infos,
        ):
            if int(np.mean(np.array(state))) == action:
                assert reward == same_reward
            else:
                assert reward == diff_reward
            assert terminated == exp_terminated
            assert truncated == exp_truncated
            assert info == exp_info

        if terminateds[env.agents[0]] or truncateds[env.agents[0]]:
            states, infos = env.reset()


@pytest.mark.parametrize(
    "env_class, reward_goal_0s, reward_goal_1s, exp_terminateds, exp_truncateds, exp_infos",
    [
        (PolicyContActionsEnv, [0, 1], [1, 0], [True, True], [False, False], {}),
        (PolicyContActionsImageEnv, [0, 1], [1, 0], [True, True], [False, False], {}),
    ],
)
def test_continuous_actions_policy_envs(
    env_class,
    reward_goal_0s,
    reward_goal_1s,
    exp_terminateds,
    exp_truncateds,
    exp_infos,
):
    env = env_class()
    states, info = env.reset()
    for _ in range(60):
        actions = {
            env.agents[0]: env.action_space[env.agents[0]].sample(),
            env.agents[1]: env.action_space[env.agents[1]].sample(),
        }
        states, rewards, terminateds, truncateds, infos = env.step(actions)

        for (
            state,
            action,
            reward,
            terminated,
            truncated,
            info,
            reward_goal_0,
            reward_goal_1,
            exp_terminated,
            exp_truncated,
            exp_info,
        ) in zip(
            list(states.values()),
            list(actions.values()),
            list(rewards.values()),
            list(terminateds.values()),
            list(truncateds.values()),
            list(infos.values()),
            reward_goal_0s,
            reward_goal_1s,
            exp_terminateds,
            exp_truncateds,
            exp_infos,
        ):
            if int(np.mean(np.array(state))):
                assert (
                    reward
                    == -((reward_goal_0 - action[0]) ** 2)
                    - (reward_goal_1 - action[1]) ** 2
                )
            else:
                assert (
                    reward
                    == -((reward_goal_1 - action[0]) ** 2)
                    - (reward_goal_0 - action[1]) ** 2
                )
            assert terminated == exp_terminated
            assert truncated == exp_truncated
            assert info == exp_info

        if terminateds[env.agents[0]] or truncateds[env.agents[0]]:
            states, infos = env.reset()


@pytest.mark.parametrize(
    "env_class, same_rewards, diff_rewards, exp_terminateds, exp_truncateds, exp_infos",
    [
        (MultiPolicyEnv, [1, 0], [0, 1], [True, True], [False, False], {}),
        (MultiPolicyImageEnv, [1, 0], [0, 1], [True, True], [False, False], {}),
    ],
)
def test_discrete_actions_multi_policy_envs(
    env_class, same_rewards, diff_rewards, exp_terminateds, exp_truncateds, exp_infos
):
    env = env_class()
    states, info = env.reset()
    for _ in range(20):
        actions = {
            env.agents[0]: env.action_space[env.agents[0]].sample(),
            env.agents[1]: env.action_space[env.agents[1]].sample(),
        }
        states, rewards, terminateds, truncateds, infos = env.step(actions)

        for (
            state,
            action,
            reward,
            terminated,
            truncated,
            info,
            same_reward,
            diff_reward,
            exp_terminated,
            exp_truncated,
            exp_info,
        ) in zip(
            list(states.values()),
            list(actions.values()),
            list(rewards.values()),
            list(terminateds.values()),
            list(truncateds.values()),
            list(infos.values()),
            same_rewards,
            diff_rewards,
            exp_terminateds,
            exp_truncateds,
            exp_infos,
        ):
            if int(np.mean(np.array(state))) == action:
                assert reward == same_reward
            else:
                assert reward == diff_reward
            assert terminated == exp_terminated
            assert truncated == exp_truncated
            assert info == exp_info

        if terminateds[env.agents[0]] or truncateds[env.agents[0]]:
            states, infos = env.reset()


def test_policy_q_learning_with_probe_env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConstantRewardEnv()
    learn_steps = 10
    algo_args = {
        "state_dims": [(env.observation_space[agent].n,) for agent in env.agents],
        "action_dims": [env.action_space[agent].n for agent in env.agents],
        "one_hot": True,
        "n_agents": env.num_agents,
        "agent_ids": env.possible_agents,
        "max_action": [(1.0,), (1.0,)],
        "min_action": [(0.0,), (0.0,)],
        "discrete_actions": True,
        "net_config": {"arch": "mlp", "h_size": [32, 32]},
        "batch_size": 256,
    }
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        memory_size=10000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        agent_ids=algo_args["agent_ids"],
        device=device,
    )

    check_policy_q_learning_with_probe_env(
        env, MADDPG, algo_args, memory, learn_steps, device
    )


def test_policy_q_learning_with_probe_env_mlp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FixedObsPolicyContActionsEnv()
    learn_steps = 10
    algo_args = {
        "state_dims": [(env.observation_space[agent].n,) for agent in env.agents],
        "action_dims": [env.action_space[agent].shape[0] for agent in env.agents],
        "one_hot": True,
        "n_agents": env.num_agents,
        "agent_ids": env.possible_agents,
        "max_action": [(1.0,), (1.0,)],
        "min_action": [(0.0,), (0.0,)],
        "discrete_actions": False,
        "net_config": {"arch": "mlp", "h_size": [32, 32]},
        "batch_size": 256,
    }
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        memory_size=10000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        agent_ids=algo_args["agent_ids"],
        device=device,
    )

    check_policy_q_learning_with_probe_env(
        env, MADDPG, algo_args, memory, learn_steps, device
    )


def test_policy_q_learning_with_probe_env_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FixedObsPolicyContActionsImageEnv()
    learn_steps = 10
    algo_args = {
        "state_dims": [(env.observation_space[agent].shape) for agent in env.agents],
        "action_dims": [env.action_space[agent].shape[0] for agent in env.agents],
        "one_hot": False,
        "n_agents": env.num_agents,
        "agent_ids": env.possible_agents,
        "max_action": [(1.0,), (1.0,)],
        "min_action": [(0.0,), (0.0,)],
        "discrete_actions": False,
        "net_config": {
            "arch": "cnn",  # Network architecture
            "h_size": [32],  # Network hidden size
            "c_size": [32, 32],  # CNN channel size
            "k_size": [4, 3],  # CNN kernel size
            "s_size": [4, 2],  # CNN stride size
            "normalize": False,  # Normalize image from range [0,255] to [0,1]
        },
        "lr_actor": 1e-4,
        "lr_critic": 1e-3,
        "batch_size": 128,
    }
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        memory_size=10000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        agent_ids=algo_args["agent_ids"],
        device=device,
    )

    check_policy_q_learning_with_probe_env(
        env, MADDPG, algo_args, memory, learn_steps, device
    )
