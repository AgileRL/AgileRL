import copy
from pathlib import Path

import dill
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer

from agilerl.algorithms.td3 import TD3
from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.wrappers.make_evolvable import MakeEvolvable


class DummyTD3(TD3):
    def __init__(self, state_dim, action_dim, one_hot, *args, **kwargs):
        super().__init__(state_dim, action_dim, one_hot, *args, **kwargs)

        self.tensor_test = torch.randn(1)


class DummyEnv:
    def __init__(self, state_size, vect=True, num_envs=2):
        self.state_size = state_size
        self.vect = vect
        if self.vect:
            self.state_size = (num_envs,) + self.state_size
            self.n_envs = num_envs
            self.num_envs = num_envs
        else:
            self.n_envs = 1

    def reset(self):
        return np.random.rand(*self.state_size), {}

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.randint(0, 5, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            {},
        )


@pytest.fixture
def simple_mlp():
    network = nn.Sequential(
        nn.Linear(4, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Tanh(),
    )
    return network


@pytest.fixture
def simple_mlp_critic():
    network = nn.Sequential(
        nn.Linear(6, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Tanh(),
    )
    return network


@pytest.fixture
def simple_cnn():
    network = nn.Sequential(
        nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        ),  # Input channels: 3 (for RGB images), Output channels: 16
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        ),  # Input channels: 16, Output channels: 32
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),  # Flatten the 2D feature map to a 1D vector
        nn.Linear(32 * 16 * 16, 128),  # Fully connected layer with 128 output features
        nn.ReLU(),
        nn.Linear(128, 2),  # Output layer with num_classes output features
    )
    return network


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1
        )  # Input channels: 3 (for RGB images), Output channels: 16
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            16, 32, kernel_size=3, stride=1, padding=1
        )  # Input channels: 16, Output channels: 32
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()  # Flatten the 2D feature map to a 1D vector
        self.linear1 = nn.Linear(
            32 * 16 * 16, 128
        )  # Fully connected layer with 128 output features
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(
            128, 128
        )  # Fully connected layer with 128 output features

    def forward(self, x, xc):
        x = self.mp1(self.relu1(self.conv1(x)))
        x = self.mp2(self.relu2(self.conv2(x)))
        x = self.flat(x)
        x = self.relu3(self.linear1(x))
        x = self.relu3(self.linear2(x))
        return x


# initialize td3 with valid parameters
def test_initialize_td3_with_minimum_parameters():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    max_action = 1

    td3 = TD3(state_dim, action_dim, one_hot, max_action)

    assert td3.state_dim == state_dim
    assert td3.action_dim == action_dim
    assert td3.one_hot == one_hot
    assert td3.max_action == max_action
    assert td3.net_config == {
        "arch": "mlp",
        "hidden_size": [64, 64],
        "mlp_output_activation": "Tanh",
    }
    assert td3.batch_size == 64
    assert td3.lr_actor == 0.0001
    assert td3.lr_critic == 0.001
    assert td3.learn_step == 5
    assert td3.gamma == 0.99
    assert td3.tau == 0.005
    assert td3.mut is None
    assert td3.device == "cpu"
    assert td3.accelerator is None
    assert td3.index == 0
    assert td3.scores == []
    assert td3.fitness == []
    assert td3.steps == [0]
    assert td3.actor_network is None
    assert isinstance(td3.actor, EvolvableMLP)
    assert isinstance(td3.actor_target, EvolvableMLP)
    assert isinstance(td3.actor_optimizer, optim.Adam)
    assert isinstance(td3.critic_1, EvolvableMLP)
    assert isinstance(td3.critic_target_1, EvolvableMLP)
    assert isinstance(td3.critic_1_optimizer, optim.Adam)
    assert isinstance(td3.critic_2, EvolvableMLP)
    assert isinstance(td3.critic_target_2, EvolvableMLP)
    assert isinstance(td3.critic_2_optimizer, optim.Adam)
    assert td3.arch == "mlp"
    assert isinstance(td3.criterion, nn.MSELoss)


# Initializes actor network with EvolvableCNN based on net_config and Accelerator.
def test_initialize_td3_with_cnn_accelerator():
    state_dim = [3, 32, 32]
    action_dim = 2
    one_hot = False
    max_action = 1
    index = 0
    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
        "normalize": False,
    }
    batch_size = 64
    lr_actor = 1e-4
    lr_critic = 1e-3
    learn_step = 5
    gamma = 0.99
    tau = 1e-3
    mut = None
    actor_network = None
    accelerator = Accelerator()
    wrap = True

    td3 = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        max_action=max_action,
        min_action=0,
        index=index,
        net_config=net_config_cnn,
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        learn_step=learn_step,
        gamma=gamma,
        tau=tau,
        mut=mut,
        actor_network=actor_network,
        accelerator=accelerator,
        wrap=wrap,
    )

    assert td3.state_dim == state_dim
    assert td3.action_dim == action_dim
    assert td3.one_hot == one_hot
    assert td3.max_action == max_action
    assert td3.net_config == net_config_cnn
    assert td3.batch_size == batch_size
    assert td3.lr_actor == lr_actor
    assert td3.lr_critic == lr_critic
    assert td3.learn_step == learn_step
    assert td3.gamma == gamma
    assert td3.tau == tau
    assert td3.mut == mut
    assert td3.accelerator == accelerator
    assert td3.index == index
    assert td3.scores == []
    assert td3.fitness == []
    assert td3.steps == [0]
    assert td3.actor_network is None
    assert isinstance(td3.actor, EvolvableCNN)
    assert isinstance(td3.actor_target, EvolvableCNN)
    assert isinstance(td3.critic_1, EvolvableCNN)
    assert isinstance(td3.critic_target_1, EvolvableCNN)
    assert isinstance(td3.critic_2, EvolvableCNN)
    assert isinstance(td3.critic_target_2, EvolvableCNN)
    assert td3.arch == "cnn"
    assert isinstance(td3.actor_optimizer, AcceleratedOptimizer)
    assert isinstance(td3.critic_1_optimizer, AcceleratedOptimizer)
    assert isinstance(td3.critic_2_optimizer, AcceleratedOptimizer)
    assert isinstance(td3.criterion, nn.MSELoss)


# Can initialize td3 with an actor network
@pytest.mark.parametrize(
    "state_dim, actor_network, critic_1_network, critic_2_network, input_tensor, input_tensor_critic",
    [
        (
            [4],
            "simple_mlp",
            "simple_mlp_critic",
            "simple_mlp_critic",
            torch.randn(1, 4),
            torch.randn(1, 6),
        ),
    ],
)
def test_initialize_td3_with_actor_network(
    state_dim,
    actor_network,
    critic_1_network,
    critic_2_network,
    input_tensor,
    input_tensor_critic,
    request,
):
    action_dim = 2
    one_hot = False
    max_action = 1
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_1_network = request.getfixturevalue(critic_1_network)
    critic_1_network = MakeEvolvable(critic_1_network, input_tensor_critic)
    critic_2_network = request.getfixturevalue(critic_2_network)
    critic_2_network = MakeEvolvable(critic_2_network, input_tensor_critic)

    td3 = TD3(
        state_dim,
        action_dim,
        one_hot,
        max_action,
        expl_noise=np.zeros((1, action_dim)),
        actor_network=actor_network,
        critic_networks=[critic_1_network, critic_2_network],
    )

    assert td3.state_dim == state_dim
    assert td3.action_dim == action_dim
    assert td3.one_hot == one_hot
    assert td3.max_action == max_action
    assert td3.net_config is None
    assert td3.batch_size == 64
    assert td3.lr_actor == 0.0001
    assert td3.lr_critic == 0.001
    assert td3.learn_step == 5
    assert td3.gamma == 0.99
    assert td3.tau == 0.005
    assert td3.mut is None
    assert td3.device == "cpu"
    assert td3.accelerator is None
    assert td3.index == 0
    assert td3.scores == []
    assert td3.fitness == []
    assert td3.steps == [0]
    assert td3.actor_network == actor_network
    assert td3.actor == actor_network
    assert td3.critic_networks == [critic_1_network, critic_2_network]
    assert isinstance(td3.actor_optimizer, optim.Adam)
    assert isinstance(td3.critic_1_optimizer, optim.Adam)
    assert isinstance(td3.critic_2_optimizer, optim.Adam)
    assert td3.arch == actor_network.arch
    assert isinstance(td3.criterion, nn.MSELoss)


# Can initialize td3 with an actor network
@pytest.mark.parametrize(
    "state_dim, actor_network, critic_1_network, critic_2_network, input_tensor, input_tensor_critic",
    [
        (
            [4],
            "simple_mlp",
            "simple_mlp_critic",
            "simple_mlp_critic",
            torch.randn(1, 4),
            torch.randn(1, 6),
        ),
    ],
)

# Can initialize td3 with an actor network but no critics - should trigger warning
def test_initialize_td3_with_actor_network_no_critics(
    state_dim,
    actor_network,
    critic_1_network,
    critic_2_network,
    input_tensor,
    input_tensor_critic,
    request,
):
    action_dim = 2
    one_hot = False
    max_action = 1
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    td3 = TD3(
        state_dim,
        action_dim,
        one_hot,
        max_action,
        actor_network=actor_network,
        critic_networks=None,
    )

    assert td3.state_dim == state_dim
    assert td3.action_dim == action_dim
    assert td3.one_hot == one_hot
    assert td3.max_action == max_action
    assert td3.net_config is not None
    assert td3.batch_size == 64
    assert td3.lr_actor == 0.0001
    assert td3.lr_critic == 0.001
    assert td3.learn_step == 5
    assert td3.gamma == 0.99
    assert td3.tau == 0.005
    assert td3.mut is None
    assert td3.device == "cpu"
    assert td3.accelerator is None
    assert td3.index == 0
    assert td3.scores == []
    assert td3.fitness == []
    assert td3.steps == [0]
    assert td3.actor != actor_network
    assert td3.critic_networks is None
    assert isinstance(td3.actor_optimizer, optim.Adam)
    assert isinstance(td3.critic_1_optimizer, optim.Adam)
    assert isinstance(td3.critic_2_optimizer, optim.Adam)
    assert isinstance(td3.criterion, nn.MSELoss)


# Can initialize td3 with an actor network
@pytest.mark.parametrize(
    "state_dim, actor_network, input_tensor",
    [
        ([3, 64, 64], "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
def test_initialize_td3_with_actor_network_cnn(
    state_dim, actor_network, input_tensor, request
):
    action_dim = 2
    one_hot = False
    max_action = 1
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_1_network = SimpleCNN()
    critic_1_network = MakeEvolvable(
        critic_1_network,
        input_tensor,
        torch.randn(1, action_dim),
    )
    critic_2_network = SimpleCNN()
    critic_2_network = MakeEvolvable(
        critic_2_network,
        input_tensor,
        torch.randn(1, action_dim),
    )

    td3 = TD3(
        state_dim,
        action_dim,
        one_hot,
        max_action,
        actor_network=actor_network,
        critic_networks=[critic_1_network, critic_2_network],
    )

    assert td3.state_dim == state_dim
    assert td3.action_dim == action_dim
    assert td3.one_hot == one_hot
    assert td3.max_action == max_action
    assert td3.net_config is None
    assert td3.batch_size == 64
    assert td3.lr_actor == 0.0001
    assert td3.lr_critic == 0.001
    assert td3.learn_step == 5
    assert td3.gamma == 0.99
    assert td3.tau == 0.005
    assert td3.mut is None
    assert td3.device == "cpu"
    assert td3.accelerator is None
    assert td3.index == 0
    assert td3.scores == []
    assert td3.fitness == []
    assert td3.steps == [0]
    assert td3.actor_network == actor_network
    assert td3.actor == actor_network
    assert td3.critic_networks == [critic_1_network, critic_2_network]
    assert isinstance(td3.actor_optimizer, optim.Adam)
    assert isinstance(td3.critic_1_optimizer, optim.Adam)
    assert isinstance(td3.critic_2_optimizer, optim.Adam)
    assert td3.arch == actor_network.arch
    assert isinstance(td3.criterion, nn.MSELoss)


# Returns the expected action when given a state observation and epsilon=0 or 1.
def test_returns_expected_action_training():
    accelerator = Accelerator()
    state_dim = [4]
    action_dim = 2
    max_action = 1

    td3 = TD3(state_dim, action_dim, one_hot=False, max_action=max_action)
    state = np.array([1, 2, 3, 4])
    training = False
    action = td3.get_action(state, training)[0]

    assert len(action) == action_dim
    for act in action:
        assert isinstance(act, np.float32)
        assert -max_action <= act <= max_action

    td3 = TD3(
        state_dim,
        action_dim,
        one_hot=True,
        max_action=max_action,
        accelerator=accelerator,
    )
    state = np.array([1])
    training = True
    action = td3.get_action(state, training)[0]

    assert len(action) == action_dim
    for act in action:
        assert isinstance(act, np.float32)
        assert -max_action <= act <= max_action

    td3 = TD3(
        state_dim,
        action_dim,
        one_hot=True,
        O_U_noise=False,
        max_action=max_action,
        accelerator=accelerator,
    )
    state = np.array([1])
    training = True
    action = td3.get_action(state, training)[0]

    assert len(action) == action_dim
    for act in action:
        assert isinstance(act, np.float32)
        assert -max_action <= act <= max_action


# learns from experiences and updates network parameters
@pytest.mark.parametrize(
    "min_action, max_action", [(-1, 1), ([-1, 0], 1), (-1, [0, 1]), ([-1, -2], [1, 0])]
)
def test_learns_from_experiences(min_action, max_action):
    state_dim = (3, 32, 32)
    action_dim = 2
    one_hot = False
    max_action = 1
    batch_size = 64
    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
        "normalize": False,
    }

    # Create an instance of the td3 class
    td3 = TD3(
        state_dim,
        action_dim,
        one_hot,
        min_action=min_action,
        max_action=max_action,
        net_config=net_config_cnn,
        batch_size=batch_size,
        policy_freq=2,
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, *state_dim)
    actions = torch.randint(0, 2, (batch_size, action_dim)).float()
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, *state_dim)
    dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [states, actions, rewards, next_states, dones]

    # Copy state dict before learning - should be different to after updating weights
    actor = td3.actor
    actor_target = td3.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(td3.actor.state_dict()))
    critic_1 = td3.critic_1
    critic_target_1 = td3.critic_target_1
    critic_1_pre_learn_sd = str(copy.deepcopy(td3.critic_1.state_dict()))
    critic_2 = td3.critic_2
    critic_target_2 = td3.critic_target_2
    critic_2_pre_learn_sd = str(copy.deepcopy(td3.critic_2.state_dict()))

    td3.scores = [0]

    actor_loss, critic_loss = td3.learn(experiences)
    assert actor_loss is None
    assert isinstance(critic_loss, float)
    assert critic_loss >= 0.0

    td3.scores = [0, 0]
    # Call the learn method
    actor_loss, critic_loss = td3.learn(experiences)

    assert isinstance(actor_loss, float)
    assert isinstance(critic_loss, float)
    assert critic_loss >= 0.0
    assert actor == td3.actor
    assert actor_target == td3.actor_target
    assert actor_pre_learn_sd != str(td3.actor.state_dict())
    assert critic_1 == td3.critic_1
    assert critic_target_1 == td3.critic_target_1
    assert critic_1_pre_learn_sd != str(td3.critic_1.state_dict())
    assert critic_2 == td3.critic_2
    assert critic_target_2 == td3.critic_target_2
    assert critic_2_pre_learn_sd != str(td3.critic_2.state_dict())


# learns from experiences and updates network parameters
def test_learns_from_experiences_with_accelerator():
    accelerator = Accelerator()
    state_dim = [4]
    action_dim = 2
    one_hot = True
    max_action = 1
    batch_size = 64

    # Create an instance of the td3 class
    td3 = TD3(
        state_dim,
        action_dim,
        one_hot,
        max_action,
        batch_size=batch_size,
        policy_freq=1,
        accelerator=accelerator,
    )

    # Create a batch of experiences
    states = torch.randint(0, state_dim[0], (batch_size, 1)).float()
    actions = torch.randint(0, 2, (batch_size, action_dim)).float()
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randint(0, state_dim[0], (batch_size, 1)).float()
    dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [states, actions, rewards, next_states, dones]

    # Copy state dict before learning - should be different to after updating weights
    actor = td3.actor
    actor_target = td3.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(td3.actor.state_dict()))
    critic_1 = td3.critic_1
    critic_target_1 = td3.critic_target_1
    critic_1_pre_learn_sd = str(copy.deepcopy(td3.critic_1.state_dict()))
    critic_2 = td3.critic_2
    critic_target_2 = td3.critic_target_2
    critic_2_pre_learn_sd = str(copy.deepcopy(td3.critic_2.state_dict()))

    td3.scores = [0]
    # Call the learn method
    td3.learn(experiences)

    assert actor == td3.actor
    assert actor_target == td3.actor_target
    assert actor_pre_learn_sd != str(td3.actor.state_dict())
    assert critic_1 == td3.critic_1
    assert critic_target_1 == td3.critic_target_1
    assert critic_1_pre_learn_sd != str(td3.critic_1.state_dict())
    assert critic_2 == td3.critic_2
    assert critic_target_2 == td3.critic_target_2
    assert critic_2_pre_learn_sd != str(td3.critic_2.state_dict())


# Updates target network parameters with soft update
def test_soft_update():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    max_action = 1
    net_config = {"arch": "mlp", "hidden_size": [64, 64]}
    batch_size = 64
    lr_actor = 1e-4
    lr_critic = 1e-3
    learn_step = 5
    gamma = 0.99
    tau = 1e-3
    mut = None
    actor_network = None
    device = "cpu"
    accelerator = None
    wrap = True

    td3 = TD3(
        state_dim,
        action_dim,
        one_hot,
        max_action,
        net_config=net_config,
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        learn_step=learn_step,
        gamma=gamma,
        tau=tau,
        mut=mut,
        actor_network=actor_network,
        device=device,
        accelerator=accelerator,
        wrap=wrap,
    )

    td3.soft_update(td3.actor, td3.actor_target)

    eval_params = list(td3.actor.parameters())
    target_params = list(td3.actor_target.parameters())
    expected_params = [
        td3.tau * eval_param + (1.0 - td3.tau) * target_param
        for eval_param, target_param in zip(eval_params, target_params)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(expected_params, target_params)
    )

    td3.soft_update(td3.critic_1, td3.critic_target_1)

    eval_params = list(td3.critic_1.parameters())
    target_params = list(td3.critic_target_1.parameters())
    expected_params = [
        td3.tau * eval_param + (1.0 - td3.tau) * target_param
        for eval_param, target_param in zip(eval_params, target_params)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(expected_params, target_params)
    )

    td3.soft_update(td3.critic_2, td3.critic_target_2)

    eval_params = list(td3.critic_2.parameters())
    target_params = list(td3.critic_target_2.parameters())
    expected_params = [
        td3.tau * eval_param + (1.0 - td3.tau) * target_param
        for eval_param, target_param in zip(eval_params, target_params)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(expected_params, target_params)
    )


# Runs algorithm test loop
def test_algorithm_test_loop():
    state_dim = (4,)
    action_dim = 2
    num_envs = 3

    env = DummyEnv(state_size=state_dim, vect=True, num_envs=num_envs)

    # env = make_vect_envs("CartPole-v1", num_envs=num_envs)
    agent = TD3(state_dim=state_dim, action_dim=action_dim, one_hot=False, max_action=1)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorised env
def test_algorithm_test_loop_unvectorized():
    state_dim = (4,)
    action_dim = 2

    env = DummyEnv(state_size=state_dim, vect=False)

    agent = TD3(state_dim=state_dim, action_dim=action_dim, one_hot=False, max_action=1)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with images
def test_algorithm_test_loop_images():
    state_dim = (3, 32, 32)
    action_dim = 2

    env = DummyEnv(state_size=state_dim, vect=True)

    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
        "normalize": False,
    }

    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=False,
        max_action=1,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorized images
def test_algorithm_test_loop_images_unvectorized():
    state_dim = (32, 32, 3)
    action_dim = 2

    env = DummyEnv(state_size=state_dim, vect=False)

    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
        "normalize": False,
    }

    agent = TD3(
        state_dim=(3, 32, 32),
        action_dim=action_dim,
        one_hot=False,
        max_action=1,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10, swap_channels=True)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
def test_clone_returns_identical_agent():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    max_action = 1

    td3 = DummyTD3(state_dim, action_dim, one_hot, max_action)
    td3.fitness = [200, 200, 200]
    td3.scores = [94, 94, 94]
    td3.steps = [2500]
    td3.tensor_attribute = torch.randn(1)
    clone_agent = td3.clone()

    assert clone_agent.state_dim == td3.state_dim
    assert clone_agent.action_dim == td3.action_dim
    assert clone_agent.one_hot == td3.one_hot
    assert clone_agent.max_action == td3.max_action
    assert clone_agent.net_config == td3.net_config
    assert clone_agent.actor_network == td3.actor_network
    assert clone_agent.critic_networks == td3.critic_networks
    assert clone_agent.batch_size == td3.batch_size
    assert clone_agent.lr_actor == td3.lr_actor
    assert clone_agent.lr_critic == td3.lr_critic
    assert clone_agent.learn_step == td3.learn_step
    assert clone_agent.gamma == td3.gamma
    assert clone_agent.tau == td3.tau
    assert clone_agent.mut == td3.mut
    assert clone_agent.device == td3.device
    assert clone_agent.accelerator == td3.accelerator
    assert str(clone_agent.actor.state_dict()) == str(td3.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        td3.actor_target.state_dict()
    )
    assert str(clone_agent.critic_1.state_dict()) == str(td3.critic_1.state_dict())
    assert str(clone_agent.critic_target_1.state_dict()) == str(
        td3.critic_target_1.state_dict()
    )
    assert str(clone_agent.critic_2.state_dict()) == str(td3.critic_2.state_dict())
    assert str(clone_agent.critic_target_2.state_dict()) == str(
        td3.critic_target_2.state_dict()
    )
    assert str(clone_agent.actor_optimizer.state_dict()) == str(
        td3.actor_optimizer.state_dict()
    )
    assert str(clone_agent.critic_1_optimizer.state_dict()) == str(
        td3.critic_1_optimizer.state_dict()
    )
    assert str(clone_agent.critic_2_optimizer.state_dict()) == str(
        td3.critic_2_optimizer.state_dict()
    )
    assert clone_agent.fitness == td3.fitness
    assert clone_agent.steps == td3.steps
    assert clone_agent.scores == td3.scores
    assert clone_agent.tensor_attribute == td3.tensor_attribute
    assert clone_agent.tensor_test == td3.tensor_test

    accelerator = Accelerator()
    td3 = TD3(state_dim, action_dim, one_hot, max_action, accelerator=accelerator)
    clone_agent = td3.clone()

    assert clone_agent.state_dim == td3.state_dim
    assert clone_agent.action_dim == td3.action_dim
    assert clone_agent.one_hot == td3.one_hot
    assert clone_agent.max_action == td3.max_action
    assert clone_agent.net_config == td3.net_config
    assert clone_agent.actor_network == td3.actor_network
    assert clone_agent.critic_networks == td3.critic_networks
    assert clone_agent.batch_size == td3.batch_size
    assert clone_agent.lr_actor == td3.lr_actor
    assert clone_agent.lr_critic == td3.lr_critic
    assert clone_agent.learn_step == td3.learn_step
    assert clone_agent.gamma == td3.gamma
    assert clone_agent.tau == td3.tau
    assert clone_agent.mut == td3.mut
    assert clone_agent.device == td3.device
    assert clone_agent.accelerator == td3.accelerator
    assert str(clone_agent.actor.state_dict()) == str(td3.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        td3.actor_target.state_dict()
    )
    assert str(clone_agent.critic_1.state_dict()) == str(td3.critic_1.state_dict())
    assert str(clone_agent.critic_target_1.state_dict()) == str(
        td3.critic_target_1.state_dict()
    )
    assert str(clone_agent.critic_2.state_dict()) == str(td3.critic_2.state_dict())
    assert str(clone_agent.critic_target_2.state_dict()) == str(
        td3.critic_target_2.state_dict()
    )
    assert str(clone_agent.actor_optimizer.state_dict()) == str(
        td3.actor_optimizer.state_dict()
    )
    assert str(clone_agent.critic_1_optimizer.state_dict()) == str(
        td3.critic_1_optimizer.state_dict()
    )
    assert str(clone_agent.critic_2_optimizer.state_dict()) == str(
        td3.critic_2_optimizer.state_dict()
    )
    assert clone_agent.fitness == td3.fitness
    assert clone_agent.steps == td3.steps
    assert clone_agent.scores == td3.scores

    accelerator = Accelerator()
    td3 = TD3(
        state_dim, action_dim, one_hot, max_action, accelerator=accelerator, wrap=False
    )
    clone_agent = td3.clone(wrap=False)

    assert clone_agent.state_dim == td3.state_dim
    assert clone_agent.action_dim == td3.action_dim
    assert clone_agent.one_hot == td3.one_hot
    assert clone_agent.max_action == td3.max_action
    assert clone_agent.net_config == td3.net_config
    assert clone_agent.actor_network == td3.actor_network
    assert clone_agent.critic_networks == td3.critic_networks
    assert clone_agent.batch_size == td3.batch_size
    assert clone_agent.lr_actor == td3.lr_actor
    assert clone_agent.lr_critic == td3.lr_critic
    assert clone_agent.learn_step == td3.learn_step
    assert clone_agent.gamma == td3.gamma
    assert clone_agent.tau == td3.tau
    assert clone_agent.mut == td3.mut
    assert clone_agent.device == td3.device
    assert clone_agent.accelerator == td3.accelerator
    assert str(clone_agent.actor.state_dict()) == str(td3.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        td3.actor_target.state_dict()
    )
    assert str(clone_agent.critic_1.state_dict()) == str(td3.critic_1.state_dict())
    assert str(clone_agent.critic_target_1.state_dict()) == str(
        td3.critic_target_1.state_dict()
    )
    assert str(clone_agent.critic_2.state_dict()) == str(td3.critic_2.state_dict())
    assert str(clone_agent.critic_target_2.state_dict()) == str(
        td3.critic_target_2.state_dict()
    )
    assert str(clone_agent.actor_optimizer.state_dict()) == str(
        td3.actor_optimizer.state_dict()
    )
    assert str(clone_agent.critic_1_optimizer.state_dict()) == str(
        td3.critic_1_optimizer.state_dict()
    )
    assert str(clone_agent.critic_2_optimizer.state_dict()) == str(
        td3.critic_2_optimizer.state_dict()
    )
    assert clone_agent.fitness == td3.fitness
    assert clone_agent.steps == td3.steps
    assert clone_agent.scores == td3.scores


def test_clone_new_index():
    state_dim = [4]
    action_dim = 2
    one_hot = False

    td3 = TD3(state_dim, action_dim, one_hot)
    clone_agent = td3.clone(index=100)

    assert clone_agent.index == 100


def test_clone_after_learning():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    batch_size = 8
    td3 = TD3(state_dim, action_dim, one_hot)

    states = torch.randn(batch_size, state_dim[0])
    actions = torch.randn(batch_size, action_dim)
    rewards = torch.rand(batch_size, 1)
    next_states = torch.randn(batch_size, state_dim[0])
    dones = torch.zeros(batch_size, 1)

    experiences = states, actions, rewards, next_states, dones
    td3.learn(experiences)
    clone_agent = td3.clone()

    assert clone_agent.state_dim == td3.state_dim
    assert clone_agent.action_dim == td3.action_dim
    assert clone_agent.one_hot == td3.one_hot
    assert clone_agent.net_config == td3.net_config
    assert clone_agent.actor_network == td3.actor_network
    assert clone_agent.critic_networks == td3.critic_networks
    assert clone_agent.batch_size == td3.batch_size
    assert clone_agent.lr_actor == td3.lr_actor
    assert clone_agent.lr_critic == td3.lr_critic
    assert clone_agent.learn_step == td3.learn_step
    assert clone_agent.gamma == td3.gamma
    assert clone_agent.tau == td3.tau
    assert clone_agent.mut == td3.mut
    assert clone_agent.device == td3.device
    assert clone_agent.accelerator == td3.accelerator
    assert str(clone_agent.actor.state_dict()) == str(td3.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        td3.actor_target.state_dict()
    )
    assert str(clone_agent.critic_1.state_dict()) == str(td3.critic_1.state_dict())
    assert str(clone_agent.critic_target_1.state_dict()) == str(
        td3.critic_target_1.state_dict()
    )
    assert str(clone_agent.critic_2.state_dict()) == str(td3.critic_2.state_dict())
    assert str(clone_agent.critic_target_2.state_dict()) == str(
        td3.critic_target_2.state_dict()
    )
    assert str(clone_agent.actor_optimizer.state_dict()) == str(
        td3.actor_optimizer.state_dict()
    )
    assert str(clone_agent.critic_1_optimizer.state_dict()) == str(
        td3.critic_1_optimizer.state_dict()
    )
    assert str(clone_agent.critic_2_optimizer.state_dict()) == str(
        td3.critic_2_optimizer.state_dict()
    )
    assert clone_agent.fitness == td3.fitness
    assert clone_agent.steps == td3.steps
    assert clone_agent.scores == td3.scores


# The method successfully unwraps the actor and actor_target models when an accelerator is present.
def test_unwrap_models():
    td3 = TD3(
        state_dim=[4],
        action_dim=2,
        one_hot=False,
        max_action=1,
        accelerator=Accelerator(),
    )
    td3.unwrap_models()
    assert isinstance(td3.actor, nn.Module)
    assert isinstance(td3.actor_target, nn.Module)
    assert isinstance(td3.critic_1, nn.Module)
    assert isinstance(td3.critic_target_1, nn.Module)
    assert isinstance(td3.critic_2, nn.Module)
    assert isinstance(td3.critic_target_2, nn.Module)


# The saved checkpoint file contains the correct data and format.
def test_save_load_checkpoint_correct_data_and_format(tmpdir):
    # Initialize the td3 agent
    td3 = TD3(state_dim=[4], action_dim=2, one_hot=False, max_action=1)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    td3.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "actor_target_init_dict" in checkpoint
    assert "actor_target_state_dict" in checkpoint
    assert "actor_optimizer_state_dict" in checkpoint
    assert "critic_1_init_dict" in checkpoint
    assert "critic_1_state_dict" in checkpoint
    assert "critic_target_1_init_dict" in checkpoint
    assert "critic_target_1_state_dict" in checkpoint
    assert "critic_1_optimizer_state_dict" in checkpoint
    assert "critic_2_init_dict" in checkpoint
    assert "critic_2_state_dict" in checkpoint
    assert "critic_target_2_init_dict" in checkpoint
    assert "critic_target_2_state_dict" in checkpoint
    assert "critic_2_optimizer_state_dict" in checkpoint
    assert "max_action" in checkpoint
    assert "net_config" in checkpoint
    assert "batch_size" in checkpoint
    assert "lr_actor" in checkpoint
    assert "lr_critic" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    td3 = TD3(state_dim=[4], action_dim=2, one_hot=False, max_action=1)
    # Load checkpoint
    td3.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert td3.net_config == {
        "arch": "mlp",
        "hidden_size": [64, 64],
        "mlp_output_activation": "Tanh",
    }
    assert isinstance(td3.actor, EvolvableMLP)
    assert isinstance(td3.actor_target, EvolvableMLP)
    assert isinstance(td3.critic_1, EvolvableMLP)
    assert isinstance(td3.critic_target_1, EvolvableMLP)
    assert isinstance(td3.critic_2, EvolvableMLP)
    assert isinstance(td3.critic_target_2, EvolvableMLP)
    assert td3.lr_actor == 1e-4
    assert td3.lr_critic == 1e-3
    assert str(td3.actor.state_dict()) == str(td3.actor_target.state_dict())
    assert str(td3.critic_1.state_dict()) == str(td3.critic_target_1.state_dict())
    assert str(td3.critic_2.state_dict()) == str(td3.critic_target_2.state_dict())
    assert td3.max_action == 1
    assert td3.batch_size == 64
    assert td3.learn_step == 5
    assert td3.gamma == 0.99
    assert td3.tau == 0.005
    assert td3.mut is None
    assert td3.index == 0
    assert td3.scores == []
    assert td3.fitness == []
    assert td3.steps == [0]


def test_save_load_checkpoint_correct_data_and_format_cnn(tmpdir):
    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
        "normalize": False,
    }

    # Initialize the td3 agent
    td3 = TD3(
        state_dim=[3, 32, 32],
        action_dim=2,
        one_hot=False,
        max_action=1,
        net_config=net_config_cnn,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    td3.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "actor_target_init_dict" in checkpoint
    assert "actor_target_state_dict" in checkpoint
    assert "actor_optimizer_state_dict" in checkpoint
    assert "critic_1_init_dict" in checkpoint
    assert "critic_1_state_dict" in checkpoint
    assert "critic_target_1_init_dict" in checkpoint
    assert "critic_target_1_state_dict" in checkpoint
    assert "critic_1_optimizer_state_dict" in checkpoint
    assert "critic_2_init_dict" in checkpoint
    assert "critic_2_state_dict" in checkpoint
    assert "critic_target_2_init_dict" in checkpoint
    assert "critic_target_2_state_dict" in checkpoint
    assert "critic_2_optimizer_state_dict" in checkpoint
    assert "max_action" in checkpoint
    assert "net_config" in checkpoint
    assert "batch_size" in checkpoint
    assert "lr_actor" in checkpoint
    assert "lr_critic" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    td3 = TD3(state_dim=[4], action_dim=2, one_hot=False, max_action=1)
    # Load checkpoint
    td3.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert td3.net_config == net_config_cnn
    assert isinstance(td3.actor, EvolvableCNN)
    assert isinstance(td3.actor_target, EvolvableCNN)
    assert isinstance(td3.critic_1, EvolvableCNN)
    assert isinstance(td3.critic_target_1, EvolvableCNN)
    assert isinstance(td3.critic_2, EvolvableCNN)
    assert isinstance(td3.critic_target_2, EvolvableCNN)
    assert td3.lr_actor == 1e-4
    assert td3.lr_critic == 1e-3
    assert str(td3.actor.state_dict()) == str(td3.actor_target.state_dict())
    assert str(td3.critic_1.state_dict()) == str(td3.critic_target_1.state_dict())
    assert str(td3.critic_2.state_dict()) == str(td3.critic_target_2.state_dict())
    assert td3.max_action == 1
    assert td3.batch_size == 64
    assert td3.learn_step == 5
    assert td3.gamma == 0.99
    assert td3.tau == 0.005
    assert td3.mut is None
    assert td3.index == 0
    assert td3.scores == []
    assert td3.fitness == []
    assert td3.steps == [0]


# The saved checkpoint file contains the correct data and format.
@pytest.mark.parametrize(
    "actor_network, input_tensor",
    [
        ("simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
def test_save_load_checkpoint_correct_data_and_format_cnn_network(
    actor_network, input_tensor, request, tmpdir
):
    action_dim = 2
    state_dim = input_tensor.shape
    one_hot = False
    max_action = 1

    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_1_network = SimpleCNN()
    critic_1_network = MakeEvolvable(
        critic_1_network,
        input_tensor,
        torch.randn(1, action_dim),
    )
    critic_2_network = SimpleCNN()
    critic_2_network = MakeEvolvable(
        critic_2_network,
        input_tensor,
        torch.randn(1, action_dim),
    )

    td3 = TD3(
        state_dim,
        action_dim,
        one_hot,
        max_action,
        actor_network=actor_network,
        critic_networks=[critic_1_network, critic_2_network],
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    td3.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "actor_target_init_dict" in checkpoint
    assert "actor_target_state_dict" in checkpoint
    assert "actor_optimizer_state_dict" in checkpoint
    assert "critic_1_init_dict" in checkpoint
    assert "critic_1_state_dict" in checkpoint
    assert "critic_target_1_init_dict" in checkpoint
    assert "critic_target_1_state_dict" in checkpoint
    assert "critic_1_optimizer_state_dict" in checkpoint
    assert "critic_2_init_dict" in checkpoint
    assert "critic_2_state_dict" in checkpoint
    assert "critic_target_2_init_dict" in checkpoint
    assert "critic_target_2_state_dict" in checkpoint
    assert "critic_2_optimizer_state_dict" in checkpoint
    assert "max_action" in checkpoint
    assert "net_config" in checkpoint
    assert "batch_size" in checkpoint
    assert "lr_actor" in checkpoint
    assert "lr_critic" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    td3 = TD3(state_dim=[4], action_dim=2, one_hot=False, max_action=1)
    # Load checkpoint
    td3.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert td3.net_config is None
    assert isinstance(td3.actor, nn.Module)
    assert isinstance(td3.actor_target, nn.Module)
    assert isinstance(td3.critic_1, nn.Module)
    assert isinstance(td3.critic_target_1, nn.Module)
    assert isinstance(td3.critic_2, nn.Module)
    assert isinstance(td3.critic_target_2, nn.Module)
    assert td3.lr_actor == 1e-4
    assert td3.lr_critic == 1e-3
    assert str(td3.actor.state_dict()) == str(td3.actor_target.state_dict())
    assert str(td3.critic_1.state_dict()) == str(td3.critic_target_1.state_dict())
    assert str(td3.critic_2.state_dict()) == str(td3.critic_target_2.state_dict())
    assert td3.max_action == 1
    assert td3.batch_size == 64
    assert td3.learn_step == 5
    assert td3.gamma == 0.99
    assert td3.tau == 0.005
    assert td3.mut is None
    assert td3.index == 0
    assert td3.scores == []
    assert td3.fitness == []
    assert td3.steps == [0]


@pytest.mark.parametrize(
    "state_dim, net_type",
    [
        ([4], "mlp"),
        ([3, 64, 64], "cnn"),
    ],
)
def test_initialize_td3_with_actor_network_evo_net(state_dim, net_type):
    action_dim = 2
    one_hot = False
    max_action = 1
    if net_type == "mlp":
        actor_network = EvolvableMLP(
            num_inputs=state_dim[0],
            num_outputs=action_dim,
            hidden_size=[64, 64],
            mlp_activation="ReLU",
            mlp_output_activation="Tanh",
        )
        critic_networks = [
            EvolvableMLP(
                num_inputs=state_dim[0] + action_dim,
                num_outputs=1,
                hidden_size=[64, 64],
                mlp_activation="ReLU",
            )
            for _ in range(2)
        ]
    else:
        actor_network = EvolvableCNN(
            input_shape=state_dim,
            num_outputs=action_dim,
            channel_size=[8, 8],
            kernel_size=[2, 2],
            stride_size=[1, 1],
            hidden_size=[64, 64],
            mlp_activation="ReLU",
            mlp_output_activation="Tanh",
        )

        critic_networks = [
            EvolvableCNN(
                input_shape=state_dim,
                num_outputs=action_dim,
                channel_size=[8, 8],
                kernel_size=[2, 2],
                stride_size=[1, 1],
                hidden_size=[64, 64],
                critic=True,
                mlp_activation="ReLU",
            )
            for _ in range(2)
        ]

    td3 = TD3(
        state_dim,
        action_dim,
        one_hot,
        actor_network=actor_network,
        critic_networks=critic_networks,
        max_action=max_action,
    )

    assert td3.state_dim == state_dim
    assert td3.action_dim == action_dim
    assert td3.one_hot == one_hot
    assert td3.max_action == max_action
    assert td3.batch_size == 64
    assert td3.lr_actor == 0.0001
    assert td3.lr_critic == 0.001
    assert td3.learn_step == 5
    assert td3.gamma == 0.99
    assert td3.tau == 0.005
    assert td3.mut is None
    assert td3.device == "cpu"
    assert td3.accelerator is None
    assert td3.index == 0
    assert td3.scores == []
    assert td3.fitness == []
    assert td3.steps == [0]
    assert td3.actor_network == actor_network
    assert td3.actor == actor_network
    assert td3.critic_networks == critic_networks
    assert isinstance(td3.actor_optimizer, optim.Adam)
    assert isinstance(td3.critic_1_optimizer, optim.Adam)
    assert isinstance(td3.critic_2_optimizer, optim.Adam)
    assert td3.arch == actor_network.arch
    assert isinstance(td3.criterion, nn.MSELoss)


def test_initialize_td3_with_incorrect_actor_net():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    actor_network = "dummy"
    critic_networks = "dummy"
    with pytest.raises(AssertionError):
        td3 = TD3(
            state_dim,
            action_dim,
            one_hot,
            actor_network=actor_network,
            critic_networks=critic_networks,
            max_action=1,
        )
        assert td3


@pytest.mark.parametrize(
    "action_array_vals, min_max, activation_func",
    [
        ([0.1, 0.2, 0.3, -0.1, -0.2, -0.3], (-1, 1), "Tanh"),
        ([0.1, 0.2, 0.3, -0.1, -0.2, -0.3], (-1, 1), "Sigmoid"),
        ([0.1, 0.2, 0.3, 0], (0, 1), "Tanh"),
        ([0.1, 0.2, 0.3, -0.1, -0.2, -0.3], (-2, 2), "Sigmoid"),
        ([0.1, 0.2, 0.3, -0.1, -0.2, -0.3], (-1, 2), "Softmax"),
        ([0.1, 0.2, 0.3, 0], ([-1, 0, -1, 0], 1), "Tanh"),
        ([0.1, 0.2, 0.3, 0], (-2, [-1, 0, -1, 0]), "Tanh"),
        ([0.1, 0.2, 0.3, 0], ([-1, 0, -1, 0], [1, 2, 3, 4]), "Tanh"),
        ([0.1, 0.2, 0.3, 0], ([-1, 0, -1, 0], 1), "Sigmoid"),
        ([0.1, 0.2, 0.3, 0], (-2, [-1, 0, -1, 0]), "Sigmoid"),
        ([0.1, 0.2, 0.3, 0], ([-1, 0, -1, 0], [1, 2, 3, 4]), "Sigmoid"),
    ],
)
def test_action_scaling_td3(action_array_vals, min_max, activation_func):
    net_config = {
        "arch": "mlp",
        "hidden_size": [64, 64],
        "mlp_output_activation": activation_func,
    }
    min_action, max_action = min_max
    if activation_func == "Tanh":
        min_activation_val, max_activation_val = -1, 1
    else:
        min_activation_val, max_activation_val = 0, 1
    action = np.array(action_array_vals)
    td3 = TD3(
        state_dim=[4],
        action_dim=4,
        one_hot=False,
        max_action=max_action,
        min_action=min_action,
        net_config=net_config,
    )
    scaled_action = td3.scale_to_action_space(action)
    min_action = np.array(min_action) if isinstance(min_action, list) else min_action
    max_action = np.array(max_action) if isinstance(max_action, list) else max_action
    expected_result = min_action + (action - min_activation_val) * (
        max_action - min_action
    ) / (max_activation_val - min_activation_val)
    assert np.allclose(scaled_action, expected_result)


@pytest.mark.parametrize(
    "min, max, action, expected_result, device",
    [
        (0, 1, [1.1, 0.75, -1], [1.0, 0.75, 0], "cpu"),
        ([0.5, 0, 0.1], 1, [0, 0, 0.2], [0.5, 0, 0.2], "cpu"),
        (0, [0.75, 1.0, 0.1], [1.0, 0.75, 0.1], [0.75, 0.75, 0.1], "cpu"),
        (
            [-1, -1, -1],
            [1, 1, 1],
            [[-2, 1, 0.25], [1.5, -1, 0.75]],
            [[-1, 1, 0.25], [1, -1, 0.75]],
            "cpu",
        ),
        (0, 1, [1.1, 0.75, -1], [1.0, 0.75, 0], "cuda"),
        ([0.5, 0, 0.1], 1, [0, 0, 0.2], [0.5, 0, 0.2], "cuda"),
        (0, [0.75, 1.0, 0.1], [1.0, 0.75, 0.1], [0.75, 0.75, 0.1], "cuda"),
        (
            [-1, -1, -1],
            [1, 1, 1],
            [[-2, 1, 0.25], [1.5, -1, 0.75]],
            [[-1, 1, 0.25], [1, -1, 0.75]],
            "cuda",
        ),
    ],
)
def test_multi_dim_clamp(min, max, action, expected_result, device):
    if isinstance(min, list):
        min = np.array(min)
    if isinstance(max, list):
        max = np.array(max)
    td3 = TD3(state_dim=[4], action_dim=1, one_hot=False, device=device)
    input = torch.tensor(action, dtype=torch.float32).to(device)
    clamped_actions = td3.multi_dim_clamp(min, max, input).type(torch.float32)
    expected_result = torch.tensor(expected_result)
    assert clamped_actions.dtype == expected_result.dtype


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained(device, accelerator, tmpdir):
    # Initialize the td3 agent
    td3 = TD3(state_dim=[4], action_dim=2, one_hot=False)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    td3.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_td3 = TD3.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_td3.state_dim == td3.state_dim
    assert new_td3.action_dim == td3.action_dim
    assert new_td3.one_hot == td3.one_hot
    assert new_td3.min_action == td3.min_action
    assert new_td3.max_action == td3.max_action
    assert new_td3.net_config == td3.net_config
    assert isinstance(new_td3.actor, EvolvableMLP)
    assert isinstance(new_td3.actor_target, EvolvableMLP)
    assert isinstance(new_td3.critic_1, EvolvableMLP)
    assert isinstance(new_td3.critic_target_1, EvolvableMLP)
    assert isinstance(new_td3.critic_2, EvolvableMLP)
    assert isinstance(new_td3.critic_target_2, EvolvableMLP)
    assert new_td3.lr_actor == td3.lr_actor
    assert new_td3.lr_critic == td3.lr_critic
    assert str(new_td3.actor.to("cpu").state_dict()) == str(td3.actor.state_dict())
    assert str(new_td3.actor_target.to("cpu").state_dict()) == str(
        td3.actor_target.state_dict()
    )
    assert str(new_td3.critic_1.to("cpu").state_dict()) == str(
        td3.critic_1.state_dict()
    )
    assert str(new_td3.critic_target_1.to("cpu").state_dict()) == str(
        td3.critic_target_1.state_dict()
    )
    assert str(new_td3.critic_2.to("cpu").state_dict()) == str(
        td3.critic_2.state_dict()
    )
    assert str(new_td3.critic_target_2.to("cpu").state_dict()) == str(
        td3.critic_target_2.state_dict()
    )
    assert new_td3.batch_size == td3.batch_size
    assert new_td3.learn_step == td3.learn_step
    assert new_td3.gamma == td3.gamma
    assert new_td3.tau == td3.tau
    assert new_td3.policy_freq == td3.policy_freq
    assert new_td3.mut == td3.mut
    assert new_td3.index == td3.index
    assert new_td3.scores == td3.scores
    assert new_td3.fitness == td3.fitness
    assert new_td3.steps == td3.steps


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_cnn(device, accelerator, tmpdir):
    # Initialize the td3 agent
    td3 = TD3(
        state_dim=[3, 32, 32],
        action_dim=2,
        one_hot=False,
        net_config={
            "arch": "cnn",
            "hidden_size": [8],
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
            "normalize": False,
        },
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    td3.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_td3 = TD3.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_td3.state_dim == td3.state_dim
    assert new_td3.action_dim == td3.action_dim
    assert new_td3.one_hot == td3.one_hot
    assert new_td3.min_action == td3.min_action
    assert new_td3.max_action == td3.max_action
    assert new_td3.net_config == td3.net_config
    assert isinstance(new_td3.actor, EvolvableCNN)
    assert isinstance(new_td3.actor_target, EvolvableCNN)
    assert isinstance(new_td3.critic_1, EvolvableCNN)
    assert isinstance(new_td3.critic_target_1, EvolvableCNN)
    assert isinstance(new_td3.critic_2, EvolvableCNN)
    assert isinstance(new_td3.critic_target_2, EvolvableCNN)
    assert new_td3.lr_actor == td3.lr_actor
    assert new_td3.lr_critic == td3.lr_critic
    assert str(new_td3.actor.to("cpu").state_dict()) == str(td3.actor.state_dict())
    assert str(new_td3.actor_target.to("cpu").state_dict()) == str(
        td3.actor_target.state_dict()
    )
    assert str(new_td3.critic_1.to("cpu").state_dict()) == str(
        td3.critic_1.state_dict()
    )
    assert str(new_td3.critic_target_1.to("cpu").state_dict()) == str(
        td3.critic_target_1.state_dict()
    )
    assert str(new_td3.critic_2.to("cpu").state_dict()) == str(
        td3.critic_2.state_dict()
    )
    assert str(new_td3.critic_target_2.to("cpu").state_dict()) == str(
        td3.critic_target_2.state_dict()
    )
    assert new_td3.batch_size == td3.batch_size
    assert new_td3.learn_step == td3.learn_step
    assert new_td3.gamma == td3.gamma
    assert new_td3.tau == td3.tau
    assert new_td3.policy_freq == td3.policy_freq
    assert new_td3.mut == td3.mut
    assert new_td3.index == td3.index
    assert new_td3.scores == td3.scores
    assert new_td3.fitness == td3.fitness
    assert new_td3.steps == td3.steps


@pytest.mark.parametrize(
    "state_dim, actor_network, input_tensor",
    [
        ([4], "simple_mlp", torch.randn(1, 4)),
        ([3, 64, 64], "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_networks(
    state_dim, actor_network, input_tensor, request, tmpdir
):
    action_dim = 2
    one_hot = False
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    # Initialize the td3 agent
    td3 = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        actor_network=actor_network,
        critic_networks=[copy.deepcopy(actor_network), copy.deepcopy(actor_network)],
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    td3.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_td3 = TD3.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_td3.state_dim == td3.state_dim
    assert new_td3.action_dim == td3.action_dim
    assert new_td3.one_hot == td3.one_hot
    assert new_td3.min_action == td3.min_action
    assert new_td3.max_action == td3.max_action
    assert new_td3.net_config == td3.net_config
    assert isinstance(new_td3.actor, nn.Module)
    assert isinstance(new_td3.actor_target, nn.Module)
    assert isinstance(new_td3.critic_1, nn.Module)
    assert isinstance(new_td3.critic_target_1, nn.Module)
    assert isinstance(new_td3.critic_2, nn.Module)
    assert isinstance(new_td3.critic_target_2, nn.Module)
    assert new_td3.lr_actor == td3.lr_actor
    assert new_td3.lr_critic == td3.lr_critic
    assert str(new_td3.actor.to("cpu").state_dict()) == str(td3.actor.state_dict())
    assert str(new_td3.actor_target.to("cpu").state_dict()) == str(
        td3.actor_target.state_dict()
    )
    assert str(new_td3.critic_1.to("cpu").state_dict()) == str(
        td3.critic_1.state_dict()
    )
    assert str(new_td3.critic_target_1.to("cpu").state_dict()) == str(
        td3.critic_target_1.state_dict()
    )
    assert str(new_td3.critic_2.to("cpu").state_dict()) == str(
        td3.critic_2.state_dict()
    )
    assert str(new_td3.critic_target_2.to("cpu").state_dict()) == str(
        td3.critic_target_2.state_dict()
    )
    assert new_td3.batch_size == td3.batch_size
    assert new_td3.learn_step == td3.learn_step
    assert new_td3.gamma == td3.gamma
    assert new_td3.tau == td3.tau
    assert new_td3.policy_freq == td3.policy_freq
    assert new_td3.mut == td3.mut
    assert new_td3.index == td3.index
    assert new_td3.scores == td3.scores
    assert new_td3.fitness == td3.fitness
    assert new_td3.steps == td3.steps
