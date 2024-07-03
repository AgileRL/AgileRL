import numpy as np
import pytest
import torch

from agilerl.algorithms.ddpg import DDPG
from agilerl.algorithms.dqn import DQN
from agilerl.algorithms.ppo import PPO
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.utils.probe_envs import (
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
    ObsDependentRewardContActionsEnv,
    ObsDependentRewardContActionsImageEnv,
    ObsDependentRewardEnv,
    ObsDependentRewardImageEnv,
    PolicyContActionsEnv,
    PolicyContActionsImageEnv,
    PolicyContActionsImageEnvSimple,
    PolicyEnv,
    PolicyImageEnv,
    check_policy_on_policy_with_probe_env,
    check_policy_q_learning_with_probe_env,
    check_q_learning_with_probe_env,
)


@pytest.mark.parametrize(
    "env_class, exp_state, exp_reward, exp_terminated, exp_truncated, exp_info",
    [
        (ConstantRewardEnv, 0, 1, True, False, {}),
        (ConstantRewardImageEnv, 0, 1, True, False, {}),
        (ConstantRewardContActionsEnv, 0, 1, True, False, {}),
        (ConstantRewardContActionsImageEnv, 0, 1, True, False, {}),
    ],
)
def test_constant_reward_envs(
    env_class, exp_state, exp_reward, exp_terminated, exp_truncated, exp_info
):
    env = env_class()
    state, info = env.reset()
    for _ in range(20):
        action = [env.action_space.sample()]
        state, reward, terminated, truncated, info = env.step(action)

        assert int(np.mean(np.array(state))) == exp_state
        assert reward == exp_reward
        assert terminated == exp_terminated
        assert truncated == exp_truncated
        assert info == exp_info

        if terminated or truncated:
            state, info = env.reset()


@pytest.mark.parametrize(
    "env_class, exp_state, exp_reward, exp_terminated, exp_truncated, exp_info",
    [
        (ObsDependentRewardEnv, 0, -1, True, False, {}),
        (ObsDependentRewardEnv, 1, 1, True, False, {}),
        (ObsDependentRewardImageEnv, 0, -1, True, False, {}),
        (ObsDependentRewardImageEnv, 1, 1, True, False, {}),
        (ObsDependentRewardContActionsEnv, 0, -1, True, False, {}),
        (ObsDependentRewardContActionsEnv, 1, 1, True, False, {}),
        (ObsDependentRewardContActionsImageEnv, 0, -1, True, False, {}),
        (ObsDependentRewardContActionsImageEnv, 1, 1, True, False, {}),
    ],
)
def test_observation_dependent_reward_envs(
    env_class, exp_state, exp_reward, exp_terminated, exp_truncated, exp_info
):
    env = env_class()
    state, info = env.reset()
    for _ in range(20):
        action = [env.action_space.sample()]
        state, reward, terminated, truncated, info = env.step(action)

        if int(np.mean(np.array(state))) == exp_state:
            assert reward == exp_reward
        assert terminated == exp_terminated
        assert truncated == exp_truncated
        assert info == exp_info

        if terminated or truncated:
            state, info = env.reset()


@pytest.mark.parametrize(
    "env_class, exp_state, exp_reward, exp_truncated, exp_info",
    [
        (DiscountedRewardEnv, 0, 0, False, {}),
        (DiscountedRewardEnv, 1, 1, False, {}),
        (DiscountedRewardImageEnv, 0, 0, False, {}),
        (DiscountedRewardImageEnv, 1, 1, False, {}),
        (DiscountedRewardContActionsEnv, 0, 0, False, {}),
        (DiscountedRewardContActionsEnv, 1, 1, False, {}),
        (DiscountedRewardContActionsImageEnv, 0, 0, False, {}),
        (DiscountedRewardContActionsImageEnv, 1, 1, False, {}),
    ],
)
def test_discounted_reward_envs(
    env_class, exp_state, exp_reward, exp_truncated, exp_info
):
    env = env_class()
    state, info = env.reset()
    for _ in range(20):
        action = [env.action_space.sample()]
        next_state, reward, terminated, truncated, info = env.step(action)

        if int(np.mean(np.array(state))) == exp_state:
            assert reward == exp_reward
        assert terminated == int(np.mean(np.array(state)))
        assert truncated == exp_truncated
        assert info == exp_info

        state = next_state

        if terminated or truncated:
            state, info = env.reset()


@pytest.mark.parametrize(
    "env_class, exp_state, exp_action, exp_reward, exp_terminated, exp_truncated, exp_info",
    [
        (FixedObsPolicyEnv, 0, 0, -1, True, False, {}),
        (FixedObsPolicyEnv, 0, 1, 1, True, False, {}),
        (FixedObsPolicyImageEnv, 0, 0, -1, True, False, {}),
        (FixedObsPolicyImageEnv, 0, 1, 1, True, False, {}),
    ],
)
def test_discrete_actions_fixed_observation_policy_reward_envs(
    env_class,
    exp_state,
    exp_action,
    exp_reward,
    exp_terminated,
    exp_truncated,
    exp_info,
):
    env = env_class()
    state, info = env.reset()
    for _ in range(20):
        action = [env.action_space.sample()]
        state, reward, terminated, truncated, info = env.step(action)

        assert int(np.mean(np.array(state))) == exp_state
        if action == exp_action:
            assert reward == exp_reward
        assert terminated == exp_terminated
        assert truncated == exp_truncated
        assert info == exp_info

        if terminated or truncated:
            state, info = env.reset()


@pytest.mark.parametrize(
    "env_class, exp_state, exp_action, exp_terminated, exp_truncated, exp_info",
    [
        (FixedObsPolicyContActionsEnv, 0, 1, True, False, {}),
        (FixedObsPolicyContActionsImageEnv, 0, 1, True, False, {}),
    ],
)
def test_continuous_actions_fixed_observation_policy_reward_envs(
    env_class, exp_state, exp_action, exp_terminated, exp_truncated, exp_info
):
    env = env_class()
    state, info = env.reset()
    for _ in range(20):
        action = [env.action_space.sample()]
        state, reward, terminated, truncated, info = env.step(action)

        assert int(np.mean(np.array(state))) == exp_state
        assert reward == -((exp_action - action[0]) ** 2)
        assert terminated == exp_terminated
        assert truncated == exp_truncated
        assert info == exp_info

        if terminated or truncated:
            state, info = env.reset()


@pytest.mark.parametrize(
    "env_class, same_reward, diff_reward, exp_terminated, exp_truncated, exp_info",
    [
        (PolicyEnv, 1, -1, True, False, {}),
        (PolicyImageEnv, 1, -1, True, False, {}),
    ],
)
def test_discrete_actions_policy_envs(
    env_class, same_reward, diff_reward, exp_terminated, exp_truncated, exp_info
):
    env = env_class()
    state, info = env.reset()
    for _ in range(20):
        action = [env.action_space.sample()]
        state, reward, terminated, truncated, info = env.step(action)

        if int(np.mean(np.array(state))) == action:
            assert reward == same_reward
        else:
            assert reward == diff_reward
        assert terminated == exp_terminated
        assert truncated == exp_truncated
        assert info == exp_info

        if terminated or truncated:
            state, info = env.reset()


@pytest.mark.parametrize(
    "env_class, reward_goal_0, reward_goal_1, exp_terminated, exp_truncated, exp_info",
    [
        (PolicyContActionsEnv, 0, 1, True, False, {}),
        (PolicyContActionsImageEnv, 0, 1, True, False, {}),
    ],
)
def test_continuous_actions_policy_envs(
    env_class, reward_goal_0, reward_goal_1, exp_terminated, exp_truncated, exp_info
):
    env = env_class()
    state, info = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

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

        if terminated or truncated:
            state, info = env.reset()


@pytest.mark.parametrize(
    "env_class, reward_goal_0, reward_goal_1, exp_terminated, exp_truncated, exp_info",
    [
        (PolicyContActionsImageEnvSimple, 1, 0, True, False, {}),
    ],
)
def test_continuous_actions_policy_envs_simple(
    env_class, reward_goal_0, reward_goal_1, exp_terminated, exp_truncated, exp_info
):
    env = env_class()
    state, info = env.reset()
    for _ in range(20):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

        if int(np.mean(np.array(state))):
            assert reward == -((reward_goal_0 - action[0]) ** 2)
        else:
            assert reward == -((reward_goal_1 - action[0]) ** 2)
        assert terminated == exp_terminated
        assert truncated == exp_truncated
        assert info == exp_info

        if terminated or truncated:
            state, info = env.reset()


def test_q_learning_with_probe_env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConstantRewardEnv()
    learn_steps = 1000
    algo_args = {
        "state_dim": (env.observation_space.n,),
        "action_dim": env.action_space.n,
        "one_hot": True if env.observation_space.n > 1 else False,
        "lr": 1e-2,
    }
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=1000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )
    check_q_learning_with_probe_env(env, DQN, algo_args, memory, learn_steps, device)


def test_policy_q_learning_with_probe_env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConstantRewardContActionsEnv()
    learn_steps = 1000
    algo_args = {
        "state_dim": (env.observation_space.n,),
        "action_dim": env.action_space.shape[0],
        "one_hot": True if env.observation_space.n > 1 else False,
        "max_action": 1.0,
        "min_action": 0.0,
        "lr_actor": 1e-2,
        "lr_critic": 1e-2,
    }
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=1000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )
    check_policy_q_learning_with_probe_env(
        env, DDPG, algo_args, memory, learn_steps, device
    )


def test_policy_q_learning_with_probe_env_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FixedObsPolicyContActionsImageEnv()
    learn_steps = 3005
    algo_args = {
        "state_dim": (env.observation_space.shape),
        "action_dim": env.action_space.shape[0],
        "one_hot": False,
        "net_config": {
            "arch": "cnn",  # Network architecture
            "hidden_size": [64],  # Network hidden size
            "channel_size": [32, 32],  # CNN channel size
            "kernel_size": [8, 4],  # CNN kernel size
            "stride_size": [4, 2],  # CNN stride size
            "normalize": False,  # Normalize image from range [0,255] to [0,1]
        },
        "max_action": 1.0,
        "min_action": 0.0,
        "policy_freq": 2,
        "lr_actor": 0.1,
        "lr_critic": 0.1,
    }
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=1000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )
    check_policy_q_learning_with_probe_env(
        env, DDPG, algo_args, memory, learn_steps, device
    )


def test_policy_on_policy_with_probe_env():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConstantRewardContActionsEnv()
    learn_steps = 100
    algo_args = {
        "state_dim": (env.observation_space.n,),
        "action_dim": env.action_space.shape[0],
        "one_hot": True if env.observation_space.n > 1 else False,
        "lr": 0.01,
        "discrete_actions": False,
    }
    check_policy_on_policy_with_probe_env(env, PPO, algo_args, learn_steps, device)


def test_policy_on_policy_with_probe_env_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = FixedObsPolicyContActionsImageEnv()
    learn_steps = 100
    algo_args = {
        "state_dim": (env.observation_space.shape),
        "action_dim": env.action_space.shape[0],
        "one_hot": False,
        "net_config": {
            "arch": "cnn",  # Network architecture
            "hidden_size": [64],  # Network hidden size
            "channel_size": [32, 32],  # CNN channel size
            "kernel_size": [8, 4],  # CNN kernel size
            "stride_size": [4, 2],  # CNN stride size
            "normalize": False,  # Normalize image from range [0,255] to [0,1]
        },
        "discrete_actions": False,
        "lr": 0.01,
    }
    check_policy_on_policy_with_probe_env(env, PPO, algo_args, learn_steps, device)
