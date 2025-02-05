import numpy as np
import pytest
import torch

from agilerl.algorithms.ddpg import DDPG
from agilerl.algorithms.dqn import DQN
from agilerl.algorithms.ppo import PPO
from agilerl.components.replay_buffer import ReplayBuffer
from agilerl.utils.probe_envs import (
    ConstantRewardContActionsDictEnv,
    ConstantRewardContActionsEnv,
    ConstantRewardContActionsImageEnv,
    ConstantRewardDictEnv,
    ConstantRewardEnv,
    ConstantRewardImageEnv,
    DiscountedRewardContActionsDictEnv,
    DiscountedRewardContActionsEnv,
    DiscountedRewardContActionsImageEnv,
    DiscountedRewardDictEnv,
    DiscountedRewardEnv,
    DiscountedRewardImageEnv,
    FixedObsPolicyContActionsDictEnv,
    FixedObsPolicyContActionsEnv,
    FixedObsPolicyContActionsImageEnv,
    FixedObsPolicyDictEnv,
    FixedObsPolicyEnv,
    FixedObsPolicyImageEnv,
    ObsDependentRewardContActionsDictEnv,
    ObsDependentRewardContActionsEnv,
    ObsDependentRewardContActionsImageEnv,
    ObsDependentRewardDictEnv,
    ObsDependentRewardEnv,
    ObsDependentRewardImageEnv,
    PolicyContActionsDictEnv,
    PolicyContActionsEnv,
    PolicyContActionsImageEnv,
    PolicyContActionsImageEnvSimple,
    PolicyDictEnv,
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
        (ConstantRewardDictEnv, {"discrete": 0, "box": 0}, 1, True, False, {}),
        (
            ConstantRewardContActionsDictEnv,
            {"discrete": 0, "box": 0},
            1,
            True,
            False,
            {},
        ),
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

        if isinstance(exp_state, dict):
            assert int(np.mean(np.array(state["box"]))) == exp_state["box"]
            assert state["discrete"] == exp_state["discrete"]
        else:
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
        (ObsDependentRewardDictEnv, {"discrete": 0, "box": 1}, -1, True, False, {}),
        (ObsDependentRewardDictEnv, {"discrete": 0, "box": 0}, 1, True, False, {}),
        (ObsDependentRewardDictEnv, {"discrete": 1, "box": 0}, -1, True, False, {}),
        (ObsDependentRewardDictEnv, {"discrete": 1, "box": 1}, 1, True, False, {}),
        (ObsDependentRewardContActionsEnv, 0, -1, True, False, {}),
        (ObsDependentRewardContActionsEnv, 1, 1, True, False, {}),
        (ObsDependentRewardContActionsImageEnv, 0, -1, True, False, {}),
        (ObsDependentRewardContActionsImageEnv, 1, 1, True, False, {}),
        (
            ObsDependentRewardContActionsDictEnv,
            {"discrete": 0, "box": 1},
            -1,
            True,
            False,
            {},
        ),
        (
            ObsDependentRewardContActionsDictEnv,
            {"discrete": 0, "box": 0},
            1,
            True,
            False,
            {},
        ),
        (
            ObsDependentRewardContActionsDictEnv,
            {"discrete": 1, "box": 0},
            -1,
            True,
            False,
            {},
        ),
        (
            ObsDependentRewardContActionsDictEnv,
            {"discrete": 1, "box": 1},
            1,
            True,
            False,
            {},
        ),
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

        if isinstance(exp_state, dict):
            if all(
                [
                    int(np.mean(np.array(state[key]))) == exp_state[key]
                    for key in exp_state.keys()
                ]
            ):
                assert reward == exp_reward
        else:
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
        (DiscountedRewardDictEnv, {"discrete": 0, "box": 1}, 1, False, {}),
        (DiscountedRewardDictEnv, {"discrete": 0, "box": 0}, 0, False, {}),
        (DiscountedRewardDictEnv, {"discrete": 1, "box": 0}, 1, False, {}),
        (DiscountedRewardDictEnv, {"discrete": 1, "box": 1}, 2, False, {}),
        (DiscountedRewardContActionsEnv, 0, 0, False, {}),
        (DiscountedRewardContActionsEnv, 1, 1, False, {}),
        (DiscountedRewardContActionsImageEnv, 0, 0, False, {}),
        (DiscountedRewardContActionsImageEnv, 1, 1, False, {}),
        (DiscountedRewardContActionsDictEnv, {"discrete": 0, "box": 1}, 1, False, {}),
        (DiscountedRewardContActionsDictEnv, {"discrete": 0, "box": 0}, 0, False, {}),
        (DiscountedRewardContActionsDictEnv, {"discrete": 1, "box": 0}, 1, False, {}),
        (DiscountedRewardContActionsDictEnv, {"discrete": 1, "box": 1}, 2, False, {}),
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

        if isinstance(exp_state, dict):
            if all(
                [
                    int(np.mean(np.array(state[key]))) == exp_state[key]
                    for key in exp_state.keys()
                ]
            ):
                assert reward == exp_reward
            assert terminated == int(np.mean(state["box"]))
        else:
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
        (FixedObsPolicyDictEnv, {"discrete": 0, "box": 0}, 0, -1, True, False, {}),
        (FixedObsPolicyDictEnv, {"discrete": 0, "box": 0}, 1, 1, True, False, {}),
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

        if isinstance(exp_state, dict):
            assert int(np.mean(np.array(state["box"]))) == exp_state["box"]
            assert state["discrete"] == exp_state["discrete"]
        else:
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
        (
            FixedObsPolicyContActionsDictEnv,
            {"discrete": 0, "box": 0},
            1,
            True,
            False,
            {},
        ),
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

        if isinstance(exp_state, dict):
            assert int(np.mean(np.array(state["box"]))) == exp_state["box"]
            assert state["discrete"] == exp_state["discrete"]
        else:
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
        (PolicyDictEnv, 1, -1, True, False, {}),
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

        if isinstance(state, dict):
            if (
                int(np.mean(np.array(state["box"]))) == action[0]
                and state["discrete"] == action[0]
            ):
                assert (
                    reward == same_reward
                ), f"action: {action}, state: {state} box: {int(np.mean(state['box']))}"
            else:
                assert reward == diff_reward
        else:
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
        (PolicyContActionsDictEnv, 0, 1, True, False, {}),
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

        if isinstance(state, dict):
            if int(np.mean(np.array(state["box"]))) and int(np.mean(state["discrete"])):
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
        else:
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
    learn_steps = 100
    algo_args = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr": 1e-2,
    }
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=1000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )
    check_q_learning_with_probe_env(env, DQN, algo_args, memory, learn_steps, device)


def test_q_learning_with_probe_env_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConstantRewardImageEnv()
    learn_steps = 100
    algo_args = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "net_config": {
            "encoder_config": {
                "channel_size": [32],  # CNN channel size
                "kernel_size": [3],  # CNN kernel size
                "stride_size": [1],  # CNN stride size
            }
        },
        "normalize_images": False,
        "lr": 1e-2,
    }
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = ReplayBuffer(
        memory_size=1000,  # Max replay buffer size
        field_names=field_names,  # Field names to store in memory
        device=device,
    )
    check_q_learning_with_probe_env(env, DQN, algo_args, memory, learn_steps, device)


def test_q_learning_with_probe_env_dict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConstantRewardDictEnv()
    learn_steps = 100
    algo_args = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "net_config": {
            "encoder_config": {
                "channel_size": [32],  # CNN channel size
                "kernel_size": [3],  # CNN kernel size
                "stride_size": [1],  # CNN stride size
                "hidden_size": [64],  # Network hidden size
                "latent_dim": 16,  # Latent dimension
            }
        },
        "normalize_images": False,
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
    learn_steps = 100
    algo_args = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
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
    env = ConstantRewardContActionsImageEnv()
    learn_steps = 100
    algo_args = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "net_config": {
            "encoder_config": {
                "channel_size": [32],  # CNN channel size
                "kernel_size": [3],  # CNN kernel size
                "stride_size": [1],  # CNN stride size
            },
            "head_config": {
                "hidden_size": [64],  # Network hidden size
            },
        },
        "normalize_images": False,
        "policy_freq": 2,
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


def test_policy_q_learning_with_probe_env_dict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConstantRewardContActionsDictEnv()
    learn_steps = 100
    algo_args = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "net_config": {
            "encoder_config": {
                "hidden_size": [64],  # Network hidden size
                "latent_dim": 16,  # Latent dimension
                "channel_size": [32],  # CNN channel size
                "kernel_size": [3],  # CNN kernel size
                "stride_size": [1],  # CNN stride size
            },
            "head_config": {
                "hidden_size": [64],  # Network hidden size
            },
        },
        "normalize_images": False,
        "policy_freq": 2,
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


def test_policy_on_policy_with_probe_env():
    device = torch.device("cpu")
    env = ConstantRewardContActionsEnv()
    learn_steps = 100
    algo_args = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr": 0.01,
    }
    check_policy_on_policy_with_probe_env(env, PPO, algo_args, learn_steps, device)


def test_policy_on_policy_with_probe_env_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConstantRewardContActionsImageEnv()  # FixedObsPolicyContActionsImageEnv()
    learn_steps = 100
    algo_args = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "net_config": {
            "encoder_config": {
                "channel_size": [32],  # CNN channel size
                "kernel_size": [3],  # CNN kernel size
                "stride_size": [1],  # CNN stride size
            },
            "head_config": {
                "hidden_size": [64],  # Network hidden size
            },
        },
        "normalize_images": False,
        "lr": 0.01,
    }
    check_policy_on_policy_with_probe_env(env, PPO, algo_args, learn_steps, device)


def test_policy_on_policy_with_probe_env_dict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConstantRewardContActionsDictEnv()  # FixedObsPolicyContActionsDictEnv()
    learn_steps = 100
    algo_args = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "net_config": {
            "encoder_config": {
                "hidden_size": [64],  # Network hidden size
                "latent_dim": 16,  # Latent dimension
                "channel_size": [32],  # CNN channel size
                "kernel_size": [3],  # CNN kernel size
                "stride_size": [1],  # CNN stride size
            },
            "head_config": {
                "hidden_size": [64],  # Network hidden size
            },
        },
        "normalize_images": False,
        "lr": 0.01,
    }
    check_policy_on_policy_with_probe_env(env, PPO, algo_args, learn_steps, device)
