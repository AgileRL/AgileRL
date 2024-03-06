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

from agilerl.algorithms.ddpg import DDPG
from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.wrappers.make_evolvable import MakeEvolvable


class DummyDDPG(DDPG):
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
        return np.random.rand(*self.state_size), "info_string"

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.randint(0, 5, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            "info_string",
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
        nn.Linear(128, 1),  # Output layer with num_classes output features
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


# initialize ddpg with valid parameters
def test_initialize_ddpg_with_minimum_parameters():
    state_dim = [4]
    action_dim = 2
    one_hot = False

    ddpg = DDPG(state_dim, action_dim, one_hot)

    assert ddpg.state_dim == state_dim
    assert ddpg.action_dim == action_dim
    assert ddpg.one_hot == one_hot
    assert ddpg.net_config == {"arch": "mlp", "h_size": [64, 64]}
    assert ddpg.batch_size == 64
    assert ddpg.lr_actor == 0.0001
    assert ddpg.lr_critic == 0.001
    assert ddpg.learn_step == 5
    assert ddpg.gamma == 0.99
    assert ddpg.tau == 0.001
    assert ddpg.mut is None
    assert ddpg.device == "cpu"
    assert ddpg.accelerator is None
    assert ddpg.index == 0
    assert ddpg.scores == []
    assert ddpg.fitness == []
    assert ddpg.steps == [0]
    assert ddpg.actor_network is None
    assert isinstance(ddpg.actor, EvolvableMLP)
    assert isinstance(ddpg.actor_target, EvolvableMLP)
    assert isinstance(ddpg.actor_optimizer, optim.Adam)
    assert isinstance(ddpg.critic, EvolvableMLP)
    assert isinstance(ddpg.critic_target, EvolvableMLP)
    assert isinstance(ddpg.critic_optimizer, optim.Adam)
    assert ddpg.arch == "mlp"
    assert isinstance(ddpg.criterion, nn.MSELoss)


# Initializes actor network with EvolvableCNN based on net_config and Accelerator.
def test_initialize_ddpg_with_cnn_accelerator():
    state_dim = [3, 32, 32]
    action_dim = 2
    one_hot = False
    index = 0
    net_config_cnn = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
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

    ddpg = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
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

    assert ddpg.state_dim == state_dim
    assert ddpg.action_dim == action_dim
    assert ddpg.one_hot == one_hot
    assert ddpg.net_config == net_config_cnn
    assert ddpg.batch_size == batch_size
    assert ddpg.lr_actor == lr_actor
    assert ddpg.learn_step == learn_step
    assert ddpg.gamma == gamma
    assert ddpg.tau == tau
    assert ddpg.mut == mut
    assert ddpg.accelerator == accelerator
    assert ddpg.index == index
    assert ddpg.scores == []
    assert ddpg.fitness == []
    assert ddpg.steps == [0]
    assert ddpg.actor_network is None
    assert isinstance(ddpg.actor, EvolvableCNN)
    assert isinstance(ddpg.actor_target, EvolvableCNN)
    assert isinstance(ddpg.critic, EvolvableCNN)
    assert isinstance(ddpg.critic_target, EvolvableCNN)
    assert ddpg.arch == "cnn"
    assert isinstance(ddpg.actor_optimizer, AcceleratedOptimizer)
    assert isinstance(ddpg.critic_optimizer, AcceleratedOptimizer)
    assert isinstance(ddpg.criterion, nn.MSELoss)


# Can initialize ddpg with an actor network
@pytest.mark.parametrize(
    "state_dim, actor_network, critic_network, input_tensor, input_tensor_critic",
    [
        ([4], "simple_mlp", "simple_mlp_critic", torch.randn(1, 4), torch.randn(1, 6)),
    ],
)
def test_initialize_ddpg_with_actor_network(
    state_dim, actor_network, critic_network, input_tensor, input_tensor_critic, request
):
    action_dim = 2
    one_hot = False
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_network = request.getfixturevalue(critic_network)
    critic_network = MakeEvolvable(critic_network, input_tensor_critic)

    ddpg = DDPG(
        state_dim,
        action_dim,
        one_hot,
        actor_network=actor_network,
        critic_network=critic_network,
    )

    assert ddpg.state_dim == state_dim
    assert ddpg.action_dim == action_dim
    assert ddpg.one_hot == one_hot
    assert ddpg.net_config is None
    assert ddpg.batch_size == 64
    assert ddpg.lr_actor == 0.0001
    assert ddpg.lr_critic == 0.001
    assert ddpg.learn_step == 5
    assert ddpg.gamma == 0.99
    assert ddpg.tau == 0.001
    assert ddpg.mut is None
    assert ddpg.device == "cpu"
    assert ddpg.accelerator is None
    assert ddpg.index == 0
    assert ddpg.scores == []
    assert ddpg.fitness == []
    assert ddpg.steps == [0]
    assert ddpg.actor_network == actor_network
    assert ddpg.actor == actor_network
    assert ddpg.critic_network == critic_network
    assert ddpg.critic == critic_network
    assert isinstance(ddpg.actor_optimizer, optim.Adam)
    assert isinstance(ddpg.critic_optimizer, optim.Adam)
    assert ddpg.arch == actor_network.arch
    assert isinstance(ddpg.criterion, nn.MSELoss)


# Can initialize ddpg with an actor network but no critic - should trigger warning
@pytest.mark.parametrize(
    "state_dim, actor_network, critic_network, input_tensor, input_tensor_critic",
    [
        ([4], "simple_mlp", "simple_mlp_critic", torch.randn(1, 4), torch.randn(1, 6)),
    ],
)
def test_initialize_ddpg_with_actor_network_no_critic(
    state_dim, actor_network, critic_network, input_tensor, input_tensor_critic, request
):
    action_dim = 2
    one_hot = False
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    ddpg = DDPG(
        state_dim,
        action_dim,
        one_hot,
        actor_network=actor_network,
        critic_network=None,
    )

    assert ddpg.state_dim == state_dim
    assert ddpg.action_dim == action_dim
    assert ddpg.one_hot == one_hot
    assert ddpg.net_config is not None
    assert ddpg.batch_size == 64
    assert ddpg.lr_actor == 0.0001
    assert ddpg.lr_critic == 0.001
    assert ddpg.learn_step == 5
    assert ddpg.gamma == 0.99
    assert ddpg.tau == 0.001
    assert ddpg.mut is None
    assert ddpg.device == "cpu"
    assert ddpg.accelerator is None
    assert ddpg.index == 0
    assert ddpg.scores == []
    assert ddpg.fitness == []
    assert ddpg.steps == [0]
    assert ddpg.actor != actor_network
    assert ddpg.critic_network is None
    assert isinstance(ddpg.actor_optimizer, optim.Adam)
    assert isinstance(ddpg.critic_optimizer, optim.Adam)
    assert isinstance(ddpg.criterion, nn.MSELoss)


# Returns the expected action when given a state observation and epsilon=0 or 1.
def test_returns_expected_action_epsilon_greedy():
    accelerator = Accelerator()
    state_dim = [4]
    action_dim = 2

    ddpg = DDPG(state_dim, action_dim, one_hot=False)
    state = np.array([1, 2, 3, 4])

    epsilon = 0
    action = ddpg.getAction(state, epsilon)[0]

    assert len(action) == action_dim
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1

    ddpg = DDPG(state_dim, action_dim, one_hot=True, accelerator=accelerator)
    state = np.array([1])
    epsilon = 1
    action = ddpg.getAction(state, epsilon)[0]

    assert len(action) == action_dim
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1


# learns from experiences and updates network parameters
def test_learns_from_experiences():
    state_dim = (3, 32, 32)
    action_dim = 2
    one_hot = False
    batch_size = 4
    policy_freq = 4
    net_config_cnn = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }

    # Create an instance of the ddpg class
    ddpg = DDPG(
        state_dim,
        action_dim,
        one_hot,
        net_config=net_config_cnn,
        batch_size=batch_size,
        policy_freq=policy_freq,
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = ddpg.actor
    actor_target = ddpg.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(ddpg.actor.state_dict()))
    critic = ddpg.critic
    critic_target = ddpg.critic_target
    critic_pre_learn_sd = str(copy.deepcopy(ddpg.critic.state_dict()))

    for i in range(policy_freq * 2):
        # Create a batch of experiences
        states = torch.rand(batch_size, *state_dim)
        actions = torch.randint(0, 2, (batch_size, action_dim)).float()
        rewards = torch.randn((batch_size, 1))
        next_states = torch.rand(batch_size, *state_dim)
        dones = torch.randint(0, 2, (batch_size, 1))

        experiences = [states, actions, rewards, next_states, dones]

        ddpg.scores.append(0)

        # Call the learn method
        actor_loss, critic_loss = ddpg.learn(experiences)

    assert isinstance(actor_loss, float)
    assert isinstance(critic_loss, float)
    assert critic_loss >= 0.0
    assert actor == ddpg.actor
    assert actor_target == ddpg.actor_target
    assert actor_pre_learn_sd != str(ddpg.actor.state_dict())
    assert critic == ddpg.critic
    assert critic_target == ddpg.critic_target
    assert critic_pre_learn_sd != str(ddpg.critic.state_dict())


# learns from experiences and updates network parameters
def test_learns_from_experiences_with_accelerator():
    accelerator = Accelerator()
    state_dim = [4]
    action_dim = 2
    one_hot = True
    batch_size = 64

    # Create an instance of the ddpg class
    ddpg = DDPG(
        state_dim,
        action_dim,
        one_hot,
        batch_size=batch_size,
        policy_freq=2,
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
    actor = ddpg.actor
    actor_target = ddpg.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(ddpg.actor.state_dict()))
    critic = ddpg.critic
    critic_target = ddpg.critic_target
    critic_pre_learn_sd = str(copy.deepcopy(ddpg.critic.state_dict()))

    ddpg.scores = [0]
    # Call the learn method
    actor_loss, critic_loss = ddpg.learn(experiences)

    assert actor_loss is None
    assert isinstance(critic_loss, float)
    assert critic_loss >= 0.0

    ddpg.scores = [0, 0]
    # Call the learn method
    actor_loss, critic_loss = ddpg.learn(experiences)

    assert isinstance(actor_loss, float)
    assert isinstance(critic_loss, float)
    assert critic_loss >= 0.0
    assert actor == ddpg.actor
    assert actor_target == ddpg.actor_target
    assert actor_pre_learn_sd != str(ddpg.actor.state_dict())
    assert critic == ddpg.critic
    assert critic_target == ddpg.critic_target
    assert critic_pre_learn_sd != str(ddpg.critic.state_dict())


# Updates target network parameters with soft update
def test_soft_update():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [64, 64]}
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

    ddpg = DDPG(
        state_dim,
        action_dim,
        one_hot,
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

    ddpg.softUpdate(ddpg.actor, ddpg.actor_target)

    eval_params = list(ddpg.actor.parameters())
    target_params = list(ddpg.actor_target.parameters())
    expected_params = [
        ddpg.tau * eval_param + (1.0 - ddpg.tau) * target_param
        for eval_param, target_param in zip(eval_params, target_params)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(expected_params, target_params)
    )

    ddpg.softUpdate(ddpg.critic, ddpg.critic_target)

    eval_params = list(ddpg.critic.parameters())
    target_params = list(ddpg.critic_target.parameters())
    expected_params = [
        ddpg.tau * eval_param + (1.0 - ddpg.tau) * target_param
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

    # env = makeVectEnvs("CartPole-v1", num_envs=num_envs)
    agent = DDPG(state_dim=state_dim, action_dim=action_dim, one_hot=False)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorised env
def test_algorithm_test_loop_unvectorized():
    state_dim = (4,)
    action_dim = 2

    env = DummyEnv(state_size=state_dim, vect=False)

    agent = DDPG(state_dim=state_dim, action_dim=action_dim, one_hot=False)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with images
def test_algorithm_test_loop_images():
    state_dim = (3, 32, 32)
    action_dim = 2

    env = DummyEnv(state_size=state_dim, vect=True)

    net_config_cnn = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }

    agent = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=False,
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
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }

    agent = DDPG(
        state_dim=(3, 32, 32),
        action_dim=action_dim,
        one_hot=False,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10, swap_channels=True)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
def test_clone_returns_identical_agent():
    state_dim = [4]
    action_dim = 2
    one_hot = False

    ddpg = DDPG(state_dim, action_dim, one_hot)
    ddpg.fitness = [200, 200, 200]
    ddpg.scores = [94, 94, 94]
    ddpg.steps = [2500]
    ddpg.tensor_attribute = torch.randn(1)
    clone_agent = ddpg.clone()

    assert clone_agent.state_dim == ddpg.state_dim
    assert clone_agent.action_dim == ddpg.action_dim
    assert clone_agent.one_hot == ddpg.one_hot
    assert clone_agent.net_config == ddpg.net_config
    assert clone_agent.actor_network == ddpg.actor_network
    assert clone_agent.critic_network == ddpg.critic_network
    assert clone_agent.batch_size == ddpg.batch_size
    assert clone_agent.lr_actor == ddpg.lr_actor
    assert clone_agent.lr_critic == ddpg.lr_critic
    assert clone_agent.learn_step == ddpg.learn_step
    assert clone_agent.gamma == ddpg.gamma
    assert clone_agent.tau == ddpg.tau
    assert clone_agent.mut == ddpg.mut
    assert clone_agent.device == ddpg.device
    assert clone_agent.accelerator == ddpg.accelerator
    assert str(clone_agent.actor.state_dict()) == str(ddpg.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        ddpg.actor_target.state_dict()
    )
    assert str(clone_agent.critic.state_dict()) == str(ddpg.critic.state_dict())
    assert str(clone_agent.critic_target.state_dict()) == str(
        ddpg.critic_target.state_dict()
    )
    assert str(clone_agent.actor_optimizer.state_dict()) == str(
        ddpg.actor_optimizer.state_dict()
    )
    assert str(clone_agent.critic_optimizer.state_dict()) == str(
        ddpg.critic_optimizer.state_dict()
    )
    assert clone_agent.fitness == ddpg.fitness
    assert clone_agent.steps == ddpg.steps
    assert clone_agent.scores == ddpg.scores
    assert clone_agent.tensor_attribute == ddpg.tensor_attribute

    accelerator = Accelerator()
    ddpg = DDPG(state_dim, action_dim, one_hot, accelerator=accelerator)
    clone_agent = ddpg.clone()

    assert clone_agent.state_dim == ddpg.state_dim
    assert clone_agent.action_dim == ddpg.action_dim
    assert clone_agent.one_hot == ddpg.one_hot
    assert clone_agent.net_config == ddpg.net_config
    assert clone_agent.actor_network == ddpg.actor_network
    assert clone_agent.critic_network == ddpg.critic_network
    assert clone_agent.batch_size == ddpg.batch_size
    assert clone_agent.lr_actor == ddpg.lr_actor
    assert clone_agent.lr_critic == ddpg.lr_critic
    assert clone_agent.learn_step == ddpg.learn_step
    assert clone_agent.gamma == ddpg.gamma
    assert clone_agent.tau == ddpg.tau
    assert clone_agent.mut == ddpg.mut
    assert clone_agent.device == ddpg.device
    assert clone_agent.accelerator == ddpg.accelerator
    assert str(clone_agent.actor.state_dict()) == str(ddpg.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        ddpg.actor_target.state_dict()
    )
    assert str(clone_agent.critic.state_dict()) == str(ddpg.critic.state_dict())
    assert str(clone_agent.critic_target.state_dict()) == str(
        ddpg.critic_target.state_dict()
    )
    assert str(clone_agent.actor_optimizer.state_dict()) == str(
        ddpg.actor_optimizer.state_dict()
    )
    assert str(clone_agent.critic_optimizer.state_dict()) == str(
        ddpg.critic_optimizer.state_dict()
    )
    assert clone_agent.fitness == ddpg.fitness
    assert clone_agent.steps == ddpg.steps
    assert clone_agent.scores == ddpg.scores

    accelerator = Accelerator()
    ddpg = DDPG(state_dim, action_dim, one_hot, accelerator=accelerator, wrap=False)
    clone_agent = ddpg.clone(wrap=False)

    assert clone_agent.state_dim == ddpg.state_dim
    assert clone_agent.action_dim == ddpg.action_dim
    assert clone_agent.one_hot == ddpg.one_hot
    assert clone_agent.net_config == ddpg.net_config
    assert clone_agent.actor_network == ddpg.actor_network
    assert clone_agent.critic_network == ddpg.critic_network
    assert clone_agent.batch_size == ddpg.batch_size
    assert clone_agent.lr_actor == ddpg.lr_actor
    assert clone_agent.lr_critic == ddpg.lr_critic
    assert clone_agent.learn_step == ddpg.learn_step
    assert clone_agent.gamma == ddpg.gamma
    assert clone_agent.tau == ddpg.tau
    assert clone_agent.mut == ddpg.mut
    assert clone_agent.device == ddpg.device
    assert clone_agent.accelerator == ddpg.accelerator
    assert str(clone_agent.actor.state_dict()) == str(ddpg.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        ddpg.actor_target.state_dict()
    )
    assert str(clone_agent.critic.state_dict()) == str(ddpg.critic.state_dict())
    assert str(clone_agent.critic_target.state_dict()) == str(
        ddpg.critic_target.state_dict()
    )
    assert str(clone_agent.actor_optimizer.state_dict()) == str(
        ddpg.actor_optimizer.state_dict()
    )
    assert str(clone_agent.critic_optimizer.state_dict()) == str(
        ddpg.critic_optimizer.state_dict()
    )
    assert clone_agent.fitness == ddpg.fitness
    assert clone_agent.steps == ddpg.steps
    assert clone_agent.scores == ddpg.scores


def test_clone_after_learning():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    batch_size = 8
    ddpg = DDPG(state_dim, action_dim, one_hot)

    states = torch.randn(batch_size, state_dim[0])
    actions = torch.randn(batch_size, action_dim)
    rewards = torch.rand(batch_size, 1)
    next_states = torch.randn(batch_size, state_dim[0])
    dones = torch.zeros(batch_size, 1)

    experiences = states, actions, rewards, next_states, dones
    ddpg.learn(experiences)
    clone_agent = ddpg.clone()
    
    assert clone_agent.state_dim == ddpg.state_dim
    assert clone_agent.action_dim == ddpg.action_dim
    assert clone_agent.one_hot == ddpg.one_hot
    assert clone_agent.net_config == ddpg.net_config
    assert clone_agent.actor_network == ddpg.actor_network
    assert clone_agent.critic_network == ddpg.critic_network
    assert clone_agent.batch_size == ddpg.batch_size
    assert clone_agent.lr_actor == ddpg.lr_actor
    assert clone_agent.lr_critic == ddpg.lr_critic
    assert clone_agent.learn_step == ddpg.learn_step
    assert clone_agent.gamma == ddpg.gamma
    assert clone_agent.tau == ddpg.tau
    assert clone_agent.mut == ddpg.mut
    assert clone_agent.device == ddpg.device
    assert clone_agent.accelerator == ddpg.accelerator
    assert str(clone_agent.actor.state_dict()) == str(ddpg.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        ddpg.actor_target.state_dict()
    )
    assert str(clone_agent.critic.state_dict()) == str(ddpg.critic.state_dict())
    assert str(clone_agent.critic_target.state_dict()) == str(
        ddpg.critic_target.state_dict()
    )
    assert str(clone_agent.actor_optimizer.state_dict()) == str(
        ddpg.actor_optimizer.state_dict()
    )
    assert str(clone_agent.critic_optimizer.state_dict()) == str(
        ddpg.critic_optimizer.state_dict()
    )
    assert clone_agent.fitness == ddpg.fitness
    assert clone_agent.steps == ddpg.steps
    assert clone_agent.scores == ddpg.scores


# The method successfully unwraps the actor and actor_target models when an accelerator is present.
def test_unwrap_models():
    ddpg = DDPG(state_dim=[4], action_dim=2, one_hot=False, accelerator=Accelerator())
    ddpg.unwrap_models()
    assert isinstance(ddpg.actor, nn.Module)
    assert isinstance(ddpg.actor_target, nn.Module)
    assert isinstance(ddpg.critic, nn.Module)
    assert isinstance(ddpg.critic_target, nn.Module)


# The saved checkpoint file contains the correct data and format.
def test_save_load_checkpoint_correct_data_and_format(tmpdir):
    # Initialize the ddpg agent
    ddpg = DDPG(state_dim=[4], action_dim=2, one_hot=False)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ddpg.saveCheckpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "actor_target_init_dict" in checkpoint
    assert "actor_target_state_dict" in checkpoint
    assert "actor_optimizer_state_dict" in checkpoint
    assert "critic_init_dict" in checkpoint
    assert "critic_state_dict" in checkpoint
    assert "critic_target_init_dict" in checkpoint
    assert "critic_target_state_dict" in checkpoint
    assert "critic_optimizer_state_dict" in checkpoint
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

    ddpg = DDPG(
        state_dim=[3, 32, 32],
        action_dim=2,
        one_hot=False,
    )
    # Load checkpoint
    ddpg.loadCheckpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert ddpg.net_config == {"arch": "mlp", "h_size": [64, 64]}
    assert isinstance(ddpg.actor, EvolvableMLP)
    assert isinstance(ddpg.actor_target, EvolvableMLP)
    assert isinstance(ddpg.critic, EvolvableMLP)
    assert isinstance(ddpg.critic_target, EvolvableMLP)
    assert ddpg.lr_actor == 1e-4
    assert ddpg.lr_critic == 1e-3
    assert str(ddpg.actor.state_dict()) == str(ddpg.actor_target.state_dict())
    assert str(ddpg.critic.state_dict()) == str(ddpg.critic_target.state_dict())
    assert ddpg.batch_size == 64
    assert ddpg.learn_step == 5
    assert ddpg.gamma == 0.99
    assert ddpg.tau == 1e-3
    assert ddpg.mut is None
    assert ddpg.index == 0
    assert ddpg.scores == []
    assert ddpg.fitness == []
    assert ddpg.steps == [0]


def test_save_load_checkpoint_correct_data_and_format_cnn(tmpdir):
    net_config_cnn = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }

    # Initialize the ddpg agent
    ddpg = DDPG(
        state_dim=[3, 32, 32], action_dim=2, one_hot=False, net_config=net_config_cnn
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ddpg.saveCheckpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "actor_target_init_dict" in checkpoint
    assert "actor_target_state_dict" in checkpoint
    assert "actor_optimizer_state_dict" in checkpoint
    assert "critic_init_dict" in checkpoint
    assert "critic_state_dict" in checkpoint
    assert "critic_target_init_dict" in checkpoint
    assert "critic_target_state_dict" in checkpoint
    assert "critic_optimizer_state_dict" in checkpoint
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

    assert checkpoint["net_config"] == net_config_cnn

    # Load checkpoint
    ddpg = DDPG(
        state_dim=[3, 32, 32],
        action_dim=2,
        one_hot=False,
        net_config={"arch": "mlp", "h_size": [64, 64]},
    )
    ddpg.loadCheckpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert ddpg.net_config == net_config_cnn
    assert isinstance(ddpg.actor, EvolvableCNN)
    assert isinstance(ddpg.actor_target, EvolvableCNN)
    assert isinstance(ddpg.critic, EvolvableCNN)
    assert isinstance(ddpg.critic_target, EvolvableCNN)
    assert ddpg.lr_actor == 1e-4
    assert ddpg.lr_critic == 1e-3
    assert str(ddpg.actor.state_dict()) == str(ddpg.actor_target.state_dict())
    assert str(ddpg.critic.state_dict()) == str(ddpg.critic_target.state_dict())
    assert ddpg.batch_size == 64
    assert ddpg.learn_step == 5
    assert ddpg.gamma == 0.99
    assert ddpg.tau == 1e-3
    assert ddpg.mut is None
    assert ddpg.index == 0
    assert ddpg.scores == []
    assert ddpg.fitness == []
    assert ddpg.steps == [0]


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
    state_dim = input_tensor.shape
    action_dim = 2

    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_network = SimpleCNN()
    critic_network = MakeEvolvable(
        critic_network,
        input_tensor,
        torch.randn(1, action_dim),
        extra_critic_dims=action_dim,
    )

    # Initialize the ddpg agent
    ddpg = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=False,
        actor_network=actor_network,
        critic_network=critic_network,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ddpg.saveCheckpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "actor_target_init_dict" in checkpoint
    assert "actor_target_state_dict" in checkpoint
    assert "actor_optimizer_state_dict" in checkpoint
    assert "critic_init_dict" in checkpoint
    assert "critic_state_dict" in checkpoint
    assert "critic_target_init_dict" in checkpoint
    assert "critic_target_state_dict" in checkpoint
    assert "critic_optimizer_state_dict" in checkpoint
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

    ddpg = DDPG(
        state_dim=[3, 32, 32],
        action_dim=2,
        one_hot=False,
    )
    # Load checkpoint
    ddpg.loadCheckpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert ddpg.net_config is None
    assert isinstance(ddpg.actor, nn.Module)
    assert isinstance(ddpg.actor_target, nn.Module)
    assert isinstance(ddpg.critic, nn.Module)
    assert isinstance(ddpg.critic_target, nn.Module)
    assert ddpg.lr_actor == 1e-4
    assert ddpg.lr_critic == 1e-3
    assert str(ddpg.actor.state_dict()) == str(ddpg.actor_target.state_dict())
    assert str(ddpg.critic.state_dict()) == str(ddpg.critic_target.state_dict())
    assert ddpg.batch_size == 64
    assert ddpg.learn_step == 5
    assert ddpg.gamma == 0.99
    assert ddpg.tau == 1e-3
    assert ddpg.mut is None
    assert ddpg.index == 0
    assert ddpg.scores == []
    assert ddpg.fitness == []
    assert ddpg.steps == [0]


# Returns the input action scaled to the action space defined by self.min_action and self.max_action.
def test_action_scaling():
    action = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    ddpg = DDPG(state_dim=[4], action_dim=1, one_hot=False, max_action=1, min_action=-1)
    scaled_action = ddpg.scale_to_action_space(action)
    assert np.array_equal(scaled_action, np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3]))

    action = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    ddpg = DDPG(state_dim=[4], action_dim=1, one_hot=False, max_action=2, min_action=-2)
    scaled_action = ddpg.scale_to_action_space(action)
    assert np.array_equal(scaled_action, np.array([0.2, 0.4, 0.6, -0.2, -0.4, -0.6]))

    action = np.array([0.1, 0.2, 0.3, 0])
    ddpg = DDPG(state_dim=[4], action_dim=1, one_hot=False, max_action=1, min_action=0)
    scaled_action = ddpg.scale_to_action_space(action)
    assert np.array_equal(scaled_action, np.array([0.1, 0.2, 0.3, 0]))

    action = np.array([0.1, 0.2, 0.3, 0])
    ddpg = DDPG(state_dim=[4], action_dim=1, one_hot=False, max_action=2, min_action=0)
    scaled_action = ddpg.scale_to_action_space(action)
    assert np.array_equal(scaled_action, np.array([0.2, 0.4, 0.6, 0]))

    action = np.array([0.1, 0.2, 0.3, -0.1, -0.2, -0.3])
    ddpg = DDPG(state_dim=[4], action_dim=1, one_hot=False, max_action=2, min_action=-1)
    scaled_action = ddpg.scale_to_action_space(action)
    assert np.array_equal(scaled_action, np.array([0.2, 0.4, 0.6, -0.1, -0.2, -0.3]))


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained(device, accelerator, tmpdir):
    # Initialize the ddpg agent
    ddpg = DDPG(state_dim=[4], action_dim=2, one_hot=False)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ddpg.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_ddpg = DDPG.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_ddpg.state_dim == ddpg.state_dim
    assert new_ddpg.action_dim == ddpg.action_dim
    assert new_ddpg.one_hot == ddpg.one_hot
    assert new_ddpg.min_action == ddpg.min_action
    assert new_ddpg.max_action == ddpg.max_action
    assert new_ddpg.net_config == ddpg.net_config
    assert isinstance(new_ddpg.actor, EvolvableMLP)
    assert isinstance(new_ddpg.actor_target, EvolvableMLP)
    assert isinstance(new_ddpg.critic, EvolvableMLP)
    assert isinstance(new_ddpg.critic_target, EvolvableMLP)
    assert new_ddpg.lr_actor == ddpg.lr_actor
    assert new_ddpg.lr_critic == ddpg.lr_critic
    assert str(new_ddpg.actor.to("cpu").state_dict()) == str(ddpg.actor.state_dict())
    assert str(new_ddpg.actor_target.to("cpu").state_dict()) == str(
        ddpg.actor_target.state_dict()
    )
    assert str(new_ddpg.critic.to("cpu").state_dict()) == str(ddpg.critic.state_dict())
    assert str(new_ddpg.critic_target.to("cpu").state_dict()) == str(
        ddpg.critic_target.state_dict()
    )
    assert new_ddpg.batch_size == ddpg.batch_size
    assert new_ddpg.learn_step == ddpg.learn_step
    assert new_ddpg.gamma == ddpg.gamma
    assert new_ddpg.tau == ddpg.tau
    assert new_ddpg.policy_freq == ddpg.policy_freq
    assert new_ddpg.mut == ddpg.mut
    assert new_ddpg.index == ddpg.index
    assert new_ddpg.scores == ddpg.scores
    assert new_ddpg.fitness == ddpg.fitness
    assert new_ddpg.steps == ddpg.steps


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_cnn(device, accelerator, tmpdir):
    # Initialize the ddpg agent
    ddpg = DDPG(
        state_dim=[3, 32, 32],
        action_dim=2,
        one_hot=False,
        net_config={
            "arch": "cnn",
            "h_size": [8],
            "c_size": [3],
            "k_size": [3],
            "s_size": [1],
            "normalize": False,
        },
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ddpg.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_ddpg = DDPG.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_ddpg.state_dim == ddpg.state_dim
    assert new_ddpg.action_dim == ddpg.action_dim
    assert new_ddpg.one_hot == ddpg.one_hot
    assert new_ddpg.min_action == ddpg.min_action
    assert new_ddpg.max_action == ddpg.max_action
    assert new_ddpg.net_config == ddpg.net_config
    assert isinstance(new_ddpg.actor, EvolvableCNN)
    assert isinstance(new_ddpg.actor_target, EvolvableCNN)
    assert isinstance(new_ddpg.critic, EvolvableCNN)
    assert isinstance(new_ddpg.critic_target, EvolvableCNN)
    assert new_ddpg.lr_actor == ddpg.lr_actor
    assert new_ddpg.lr_critic == ddpg.lr_critic
    assert str(new_ddpg.actor.to("cpu").state_dict()) == str(ddpg.actor.state_dict())
    assert str(new_ddpg.actor_target.to("cpu").state_dict()) == str(
        ddpg.actor_target.state_dict()
    )
    assert str(new_ddpg.critic.to("cpu").state_dict()) == str(ddpg.critic.state_dict())
    assert str(new_ddpg.critic_target.to("cpu").state_dict()) == str(
        ddpg.critic_target.state_dict()
    )
    assert new_ddpg.batch_size == ddpg.batch_size
    assert new_ddpg.learn_step == ddpg.learn_step
    assert new_ddpg.gamma == ddpg.gamma
    assert new_ddpg.tau == ddpg.tau
    assert new_ddpg.policy_freq == ddpg.policy_freq
    assert new_ddpg.mut == ddpg.mut
    assert new_ddpg.index == ddpg.index
    assert new_ddpg.scores == ddpg.scores
    assert new_ddpg.fitness == ddpg.fitness
    assert new_ddpg.steps == ddpg.steps


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

    # Initialize the ddpg agent
    ddpg = DDPG(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        actor_network=actor_network,
        critic_network=copy.deepcopy(actor_network),
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ddpg.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_ddpg = DDPG.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_ddpg.state_dim == ddpg.state_dim
    assert new_ddpg.action_dim == ddpg.action_dim
    assert new_ddpg.one_hot == ddpg.one_hot
    assert new_ddpg.min_action == ddpg.min_action
    assert new_ddpg.max_action == ddpg.max_action
    assert new_ddpg.net_config == ddpg.net_config
    assert isinstance(new_ddpg.actor, nn.Module)
    assert isinstance(new_ddpg.actor_target, nn.Module)
    assert isinstance(new_ddpg.critic, nn.Module)
    assert isinstance(new_ddpg.critic_target, nn.Module)
    assert new_ddpg.lr_actor == ddpg.lr_actor
    assert new_ddpg.lr_critic == ddpg.lr_critic
    assert str(new_ddpg.actor.to("cpu").state_dict()) == str(ddpg.actor.state_dict())
    assert str(new_ddpg.actor_target.to("cpu").state_dict()) == str(
        ddpg.actor_target.state_dict()
    )
    assert str(new_ddpg.critic.to("cpu").state_dict()) == str(ddpg.critic.state_dict())
    assert str(new_ddpg.critic_target.to("cpu").state_dict()) == str(
        ddpg.critic_target.state_dict()
    )
    assert new_ddpg.batch_size == ddpg.batch_size
    assert new_ddpg.learn_step == ddpg.learn_step
    assert new_ddpg.gamma == ddpg.gamma
    assert new_ddpg.tau == ddpg.tau
    assert new_ddpg.policy_freq == ddpg.policy_freq
    assert new_ddpg.mut == ddpg.mut
    assert new_ddpg.index == ddpg.index
    assert new_ddpg.scores == ddpg.scores
    assert new_ddpg.fitness == ddpg.fitness
    assert new_ddpg.steps == ddpg.steps
