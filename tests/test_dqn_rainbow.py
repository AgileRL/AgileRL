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

from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.networks.evolvable_cnn import EvolvableCNN
from agilerl.networks.evolvable_mlp import EvolvableMLP
from agilerl.wrappers.make_evolvable import MakeEvolvable


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


# initialize DQN with valid parameters
def test_initialize_dqn_with_minimum_parameters():
    state_dim = [4]
    action_dim = 2
    one_hot = False

    dqn = RainbowDQN(state_dim, action_dim, one_hot)

    assert dqn.state_dim == state_dim
    assert dqn.action_dim == action_dim
    assert dqn.one_hot == one_hot
    assert dqn.net_config == {"arch": "mlp", "h_size": [64, 64]}
    assert dqn.batch_size == 64
    assert dqn.lr == 0.0001
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 0.001
    assert dqn.mut is None
    assert dqn.device == "cpu"
    assert dqn.accelerator is None
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]
    assert dqn.actor_network is None
    assert isinstance(dqn.actor, EvolvableMLP)
    assert isinstance(dqn.actor_target, EvolvableMLP)
    assert isinstance(dqn.optimizer_type, optim.Adam)
    assert dqn.arch == "mlp"
    assert dqn.optimizer == dqn.optimizer_type


# Initializes actor network with EvolvableCNN based on net_config and Accelerator.
def test_initialize_dqn_with_cnn_accelerator():
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
    lr = 1e-4
    learn_step = 5
    gamma = 0.99
    tau = 1e-3
    mut = None
    actor_network = None
    accelerator = Accelerator()
    wrap = True

    dqn = RainbowDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        index=index,
        net_config=net_config_cnn,
        batch_size=batch_size,
        lr=lr,
        learn_step=learn_step,
        gamma=gamma,
        tau=tau,
        mut=mut,
        actor_network=actor_network,
        accelerator=accelerator,
        wrap=wrap,
    )

    assert dqn.state_dim == state_dim
    assert dqn.action_dim == action_dim
    assert dqn.one_hot == one_hot
    assert dqn.net_config == net_config_cnn
    assert dqn.batch_size == batch_size
    assert dqn.lr == lr
    assert dqn.learn_step == learn_step
    assert dqn.gamma == gamma
    assert dqn.tau == tau
    assert dqn.mut == mut
    assert dqn.accelerator == accelerator
    assert dqn.index == index
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]
    assert dqn.actor_network is None
    assert isinstance(dqn.actor, EvolvableCNN)
    assert isinstance(dqn.actor_target, EvolvableCNN)
    assert isinstance(dqn.optimizer_type, optim.Adam)
    assert dqn.arch == "cnn"
    assert isinstance(dqn.optimizer, AcceleratedOptimizer)


# Can initialize DQN with an actor network
@pytest.mark.parametrize(
    "state_dim, actor_network, input_tensor",
    [
        ([4], "simple_mlp", torch.randn(1, 4)),
        ([3, 64, 64], "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
def test_initialize_dqn_with_actor_network(
    state_dim, actor_network, input_tensor, request
):
    action_dim = 2
    one_hot = False
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    dqn = RainbowDQN(state_dim, action_dim, one_hot, actor_network=actor_network)

    assert dqn.state_dim == state_dim
    assert dqn.action_dim == action_dim
    assert dqn.one_hot == one_hot
    assert dqn.net_config is None
    assert dqn.batch_size == 64
    assert dqn.lr == 0.0001
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 0.001
    assert dqn.mut is None
    assert dqn.device == "cpu"
    assert dqn.accelerator is None
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]
    assert dqn.actor_network == actor_network
    assert dqn.actor == actor_network
    assert isinstance(dqn.optimizer_type, optim.Adam)
    assert dqn.arch == actor_network.arch
    assert dqn.optimizer == dqn.optimizer_type


@pytest.mark.parametrize(
    "accelerator",
    [
        None,
        Accelerator(),
    ],
)
# Returns the expected action when given a state observation and action mask
def test_returns_expected_action(accelerator):
    state_dim = [4]
    action_dim = 2
    one_hot = False

    dqn = RainbowDQN(state_dim, action_dim, one_hot, accelerator=accelerator)
    state = np.array([1, 2, 3, 4])

    action_mask = None

    action = dqn.getAction(state, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_dim

    action_mask = np.array([0, 1])

    action = dqn.getAction(state, action_mask)[0]

    assert action.is_integer()
    assert action == 1


@pytest.mark.parametrize(
    "accelerator",
    [
        None,
        Accelerator(),
    ],
)
# Returns the expected action when given a state observation and action mask
def test_returns_expected_action_one_hot(accelerator):
    state_dim = [4]
    action_dim = 2
    one_hot = True

    dqn = RainbowDQN(state_dim, action_dim, one_hot, accelerator=accelerator)
    state = np.array([1])

    action_mask = None

    action = dqn.getAction(state, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_dim

    action_mask = np.array([0, 1])

    action = dqn.getAction(state, action_mask)[0]

    assert action.is_integer()
    assert action == 1


@pytest.mark.parametrize(
    "accelerator",
    [
        None,
        Accelerator(),
    ],
)
# learns from experiences and updates network parameters
def test_learns_from_experiences(accelerator):
    state_dim = [4]
    action_dim = 2
    one_hot = False
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        state_dim, action_dim, one_hot, batch_size=batch_size, accelerator=accelerator
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, state_dim[0])
    actions = torch.randint(0, action_dim, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, state_dim[0])
    dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [states, actions, rewards, next_states, dones]

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(dqn.actor.state_dict()))
    actor_target_pre_learn_sd = str(copy.deepcopy(dqn.actor_target.state_dict()))

    # Call the learn method
    new_idxs, new_priorities = dqn.learn(experiences, n_step=False, per=False)

    assert new_idxs is None
    assert new_priorities is None
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert actor_pre_learn_sd != str(dqn.actor.state_dict())
    assert actor_target_pre_learn_sd != str(dqn.actor_target.state_dict())


@pytest.mark.parametrize(
    "accelerator",
    [
        None,
        Accelerator(),
    ],
)
# learns from experiences and updates network parameters
def test_learns_from_experiences_one_hot(accelerator):
    state_dim = [4]
    action_dim = 2
    one_hot = True
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        state_dim, action_dim, one_hot, batch_size=batch_size, accelerator=accelerator
    )

    # Create a batch of experiences
    states = torch.randint(0, state_dim[0], (batch_size, 1))
    actions = torch.randint(0, action_dim, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randint(0, state_dim[0], (batch_size, 1))
    dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [states, actions, rewards, next_states, dones]

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(dqn.actor.state_dict()))
    actor_target_pre_learn_sd = str(copy.deepcopy(dqn.actor_target.state_dict()))

    # Call the learn method
    new_idxs, new_priorities = dqn.learn(experiences, n_step=False, per=False)

    assert new_idxs is None
    assert new_priorities is None
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert actor_pre_learn_sd != str(dqn.actor.state_dict())
    assert actor_target_pre_learn_sd != str(dqn.actor_target.state_dict())


@pytest.mark.parametrize(
    "accelerator",
    [
        None,
        Accelerator(),
    ],
)
# learns from experiences and updates network parameters
def test_learns_from_experiences_n_step(accelerator):
    state_dim = [4]
    action_dim = 2
    one_hot = False
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        state_dim, action_dim, one_hot, batch_size=batch_size, accelerator=accelerator
    )

    # Create a batch of experiences
    # Create a batch of experiences
    states = torch.randn(batch_size, state_dim[0])
    actions = torch.randint(0, action_dim, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, state_dim[0])
    dones = torch.randint(0, 2, (batch_size, 1))
    idxs = np.arange(batch_size)
    n_states = torch.randn(batch_size, state_dim[0])
    n_actions = torch.randint(0, action_dim, (batch_size, 1))
    n_rewards = torch.randn((batch_size, 1))
    n_next_states = torch.randn(batch_size, state_dim[0])
    n_dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [
        states,
        actions,
        rewards,
        next_states,
        dones,
        idxs,
        n_states,
        n_actions,
        n_rewards,
        n_next_states,
        n_dones,
    ]

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(dqn.actor.state_dict()))
    actor_target_pre_learn_sd = str(copy.deepcopy(dqn.actor_target.state_dict()))

    # Call the learn method
    new_idxs, new_priorities = dqn.learn(experiences, n_step=True, per=False)

    assert new_idxs is not None
    assert new_priorities is None
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert actor_pre_learn_sd != str(dqn.actor.state_dict())
    assert actor_target_pre_learn_sd != str(dqn.actor_target.state_dict())


@pytest.mark.parametrize(
    "accelerator",
    [
        None,
        Accelerator(),
    ],
)
# learns from experiences and updates network parameters
def test_learns_from_experiences_per(accelerator):
    state_dim = [4]
    action_dim = 2
    one_hot = False
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        state_dim, action_dim, one_hot, batch_size=batch_size, accelerator=accelerator
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, state_dim[0])
    actions = torch.randint(0, action_dim, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, state_dim[0])
    dones = torch.randint(0, 2, (batch_size, 1))
    weights = torch.rand(batch_size)
    idxs = np.arange(batch_size)

    experiences = [states, actions, rewards, next_states, dones, weights, idxs]

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(dqn.actor.state_dict()))
    actor_target_pre_learn_sd = str(copy.deepcopy(dqn.actor_target.state_dict()))

    # Call the learn method
    new_idxs, new_priorities = dqn.learn(experiences, n_step=False, per=True)

    assert isinstance(new_idxs, np.ndarray)
    assert isinstance(new_priorities, np.ndarray)
    assert np.array_equal(new_idxs, idxs)
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert actor_pre_learn_sd != str(dqn.actor.state_dict())
    assert actor_target_pre_learn_sd != str(dqn.actor_target.state_dict())


@pytest.mark.parametrize(
    "accelerator",
    [
        None,
        Accelerator(),
    ],
)
# learns from experiences and updates network parameters
def test_learns_from_experiences_per_n_step(accelerator):
    state_dim = [4]
    action_dim = 2
    one_hot = False
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        state_dim, action_dim, one_hot, batch_size=batch_size, accelerator=accelerator
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, state_dim[0])
    actions = torch.randint(0, action_dim, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, state_dim[0])
    dones = torch.randint(0, 2, (batch_size, 1))
    weights = torch.rand(batch_size)
    idxs = np.arange(batch_size)
    n_states = torch.randn(batch_size, state_dim[0])
    n_actions = torch.randint(0, action_dim, (batch_size, 1))
    n_rewards = torch.randn((batch_size, 1))
    n_next_states = torch.randn(batch_size, state_dim[0])
    n_dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [
        states,
        actions,
        rewards,
        next_states,
        dones,
        weights,
        idxs,
        n_states,
        n_actions,
        n_rewards,
        n_next_states,
        n_dones,
    ]

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(dqn.actor.state_dict()))
    actor_target_pre_learn_sd = str(copy.deepcopy(dqn.actor_target.state_dict()))

    # Call the learn method
    new_idxs, new_priorities = dqn.learn(experiences, n_step=True, per=True)

    assert isinstance(new_idxs, np.ndarray)
    assert isinstance(new_priorities, np.ndarray)
    assert np.array_equal(new_idxs, idxs)
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert actor_pre_learn_sd != str(dqn.actor.state_dict())
    assert actor_target_pre_learn_sd != str(dqn.actor_target.state_dict())


# Updates target network parameters with soft update
def test_soft_update():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [64, 64]}
    batch_size = 64
    lr = 1e-4
    learn_step = 5
    gamma = 0.99
    tau = 1e-3
    mut = None
    actor_network = None
    device = "cpu"
    accelerator = None
    wrap = True

    dqn = RainbowDQN(
        state_dim,
        action_dim,
        one_hot,
        net_config=net_config,
        batch_size=batch_size,
        lr=lr,
        learn_step=learn_step,
        gamma=gamma,
        tau=tau,
        mut=mut,
        actor_network=actor_network,
        device=device,
        accelerator=accelerator,
        wrap=wrap,
    )

    dqn.softUpdate()

    eval_params = list(dqn.actor.parameters())
    target_params = list(dqn.actor_target.parameters())
    expected_params = [
        dqn.tau * eval_param + (1.0 - dqn.tau) * target_param
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
    agent = RainbowDQN(state_dim=state_dim, action_dim=action_dim, one_hot=False)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorised env
def test_algorithm_test_loop_unvectorized():
    state_dim = (4,)
    action_dim = 2

    env = DummyEnv(state_size=state_dim, vect=False)

    agent = RainbowDQN(state_dim=state_dim, action_dim=action_dim, one_hot=False)
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

    agent = RainbowDQN(
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

    agent = RainbowDQN(
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

    dqn = RainbowDQN(state_dim, action_dim, one_hot)
    dqn.fitness = [200, 200, 200]
    dqn.scores = [94, 94, 94]
    dqn.steps = [2500]
    clone_agent = dqn.clone()

    assert clone_agent.state_dim == dqn.state_dim
    assert clone_agent.action_dim == dqn.action_dim
    assert clone_agent.one_hot == dqn.one_hot
    assert clone_agent.net_config == dqn.net_config
    assert clone_agent.actor_network == dqn.actor_network
    assert clone_agent.batch_size == dqn.batch_size
    assert clone_agent.lr == dqn.lr
    assert clone_agent.learn_step == dqn.learn_step
    assert clone_agent.gamma == dqn.gamma
    assert clone_agent.tau == dqn.tau
    assert clone_agent.mut == dqn.mut
    assert clone_agent.device == dqn.device
    assert clone_agent.accelerator == dqn.accelerator
    assert str(clone_agent.actor.state_dict()) == str(dqn.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        dqn.actor_target.state_dict()
    )
    assert str(clone_agent.optimizer.state_dict()) == str(dqn.optimizer.state_dict())
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores

    accelerator = Accelerator()
    dqn = RainbowDQN(state_dim, action_dim, one_hot, accelerator=accelerator)
    clone_agent = dqn.clone()

    assert clone_agent.state_dim == dqn.state_dim
    assert clone_agent.action_dim == dqn.action_dim
    assert clone_agent.one_hot == dqn.one_hot
    assert clone_agent.net_config == dqn.net_config
    assert clone_agent.actor_network == dqn.actor_network
    assert clone_agent.batch_size == dqn.batch_size
    assert clone_agent.lr == dqn.lr
    assert clone_agent.learn_step == dqn.learn_step
    assert clone_agent.gamma == dqn.gamma
    assert clone_agent.tau == dqn.tau
    assert clone_agent.mut == dqn.mut
    assert clone_agent.device == dqn.device
    assert clone_agent.accelerator == dqn.accelerator
    assert str(clone_agent.actor.state_dict()) == str(dqn.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        dqn.actor_target.state_dict()
    )
    assert str(clone_agent.optimizer.state_dict()) == str(dqn.optimizer.state_dict())
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores

    accelerator = Accelerator()
    dqn = RainbowDQN(
        state_dim, action_dim, one_hot, accelerator=accelerator, wrap=False
    )
    clone_agent = dqn.clone(wrap=False)

    assert clone_agent.state_dim == dqn.state_dim
    assert clone_agent.action_dim == dqn.action_dim
    assert clone_agent.one_hot == dqn.one_hot
    assert clone_agent.net_config == dqn.net_config
    assert clone_agent.actor_network == dqn.actor_network
    assert clone_agent.batch_size == dqn.batch_size
    assert clone_agent.lr == dqn.lr
    assert clone_agent.learn_step == dqn.learn_step
    assert clone_agent.gamma == dqn.gamma
    assert clone_agent.tau == dqn.tau
    assert clone_agent.mut == dqn.mut
    assert clone_agent.device == dqn.device
    assert clone_agent.accelerator == dqn.accelerator
    assert str(clone_agent.actor.state_dict()) == str(dqn.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        dqn.actor_target.state_dict()
    )
    assert str(clone_agent.optimizer.state_dict()) == str(dqn.optimizer.state_dict())
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores


# The method successfully unwraps the actor and actor_target models when an accelerator is present.
def test_unwrap_models():
    dqn = RainbowDQN(
        state_dim=[4], action_dim=2, one_hot=False, accelerator=Accelerator()
    )
    dqn.unwrap_models()
    assert isinstance(dqn.actor, nn.Module)
    assert isinstance(dqn.actor_target, nn.Module)


# The saved checkpoint file contains the correct data and format.
def test_save_load_checkpoint_correct_data_and_format(tmpdir):
    # Initialize the DQN agent
    dqn = RainbowDQN(state_dim=[4], action_dim=2, one_hot=False)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.saveCheckpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "actor_target_init_dict" in checkpoint
    assert "actor_target_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "net_config" in checkpoint
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    # Load checkpoint
    dqn.loadCheckpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert dqn.net_config == {"arch": "mlp", "h_size": [64, 64]}
    assert isinstance(dqn.actor, EvolvableMLP)
    assert isinstance(dqn.actor_target, EvolvableMLP)
    assert dqn.lr == 1e-4
    assert str(dqn.actor.state_dict()) == str(dqn.actor_target.state_dict())
    assert str(dqn.optimizer.state_dict()) == str(dqn.optimizer_type.state_dict())
    assert dqn.batch_size == 64
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 1e-3
    assert dqn.mut is None
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]


def test_save_load_checkpoint_correct_data_and_format_cnn(tmpdir):
    net_config_cnn = {
        "arch": "cnn",
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }

    # Initialize the DQN agent
    dqn = RainbowDQN(
        state_dim=[3, 32, 32], action_dim=2, one_hot=False, net_config=net_config_cnn
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.saveCheckpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "actor_target_init_dict" in checkpoint
    assert "actor_target_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "net_config" in checkpoint
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    # Load checkpoint
    dqn.loadCheckpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert dqn.net_config == net_config_cnn
    assert isinstance(dqn.actor, EvolvableCNN)
    assert isinstance(dqn.actor_target, EvolvableCNN)
    assert dqn.lr == 1e-4
    assert str(dqn.actor.state_dict()) == str(dqn.actor_target.state_dict())
    assert str(dqn.optimizer.state_dict()) == str(dqn.optimizer_type.state_dict())
    assert dqn.batch_size == 64
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 1e-3
    assert dqn.mut is None
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]


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
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    # Initialize the DQN agent
    dqn = RainbowDQN(
        state_dim=[3, 64, 64], action_dim=2, one_hot=False, actor_network=actor_network
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.saveCheckpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "actor_target_init_dict" in checkpoint
    assert "actor_target_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "net_config" in checkpoint
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "tau" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    # Load checkpoint
    dqn.loadCheckpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert dqn.net_config is None
    assert isinstance(dqn.actor, nn.Module)
    assert isinstance(dqn.actor_target, nn.Module)
    assert dqn.lr == 1e-4
    assert str(dqn.actor.state_dict()) == str(dqn.actor_target.state_dict())
    assert str(dqn.optimizer.state_dict()) == str(dqn.optimizer_type.state_dict())
    assert dqn.batch_size == 64
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 1e-3
    assert dqn.mut is None
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained(device, accelerator, tmpdir):
    # Initialize the DQN agent
    dqn = RainbowDQN(state_dim=[4], action_dim=2, one_hot=False)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_dqn = RainbowDQN.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_dqn.state_dim == dqn.state_dim
    assert new_dqn.action_dim == dqn.action_dim
    assert new_dqn.one_hot == dqn.one_hot
    assert new_dqn.net_config == dqn.net_config
    assert isinstance(new_dqn.actor, EvolvableMLP)
    assert isinstance(new_dqn.actor_target, EvolvableMLP)
    assert new_dqn.lr == dqn.lr
    assert str(new_dqn.actor.to("cpu").state_dict()) == str(dqn.actor.state_dict())
    assert str(new_dqn.actor_target.to("cpu").state_dict()) == str(
        dqn.actor_target.state_dict()
    )
    assert new_dqn.batch_size == dqn.batch_size
    assert new_dqn.learn_step == dqn.learn_step
    assert new_dqn.gamma == dqn.gamma
    assert new_dqn.tau == dqn.tau
    assert new_dqn.mut == dqn.mut
    assert new_dqn.index == dqn.index
    assert new_dqn.scores == dqn.scores
    assert new_dqn.fitness == dqn.fitness
    assert new_dqn.steps == dqn.steps


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_cnn(device, accelerator, tmpdir):
    # Initialize the DQN agent
    dqn = RainbowDQN(
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
    dqn.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_dqn = RainbowDQN.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_dqn.state_dim == dqn.state_dim
    assert new_dqn.action_dim == dqn.action_dim
    assert new_dqn.one_hot == dqn.one_hot
    assert new_dqn.net_config == dqn.net_config
    assert isinstance(new_dqn.actor, EvolvableCNN)
    assert isinstance(new_dqn.actor_target, EvolvableCNN)
    assert new_dqn.lr == dqn.lr
    assert str(new_dqn.actor.to("cpu").state_dict()) == str(dqn.actor.state_dict())
    assert str(new_dqn.actor_target.to("cpu").state_dict()) == str(
        dqn.actor_target.state_dict()
    )
    assert new_dqn.batch_size == dqn.batch_size
    assert new_dqn.learn_step == dqn.learn_step
    assert new_dqn.gamma == dqn.gamma
    assert new_dqn.tau == dqn.tau
    assert new_dqn.mut == dqn.mut
    assert new_dqn.index == dqn.index
    assert new_dqn.scores == dqn.scores
    assert new_dqn.fitness == dqn.fitness
    assert new_dqn.steps == dqn.steps


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

    # Initialize the DQN agent
    dqn = RainbowDQN(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        actor_network=actor_network,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_dqn = RainbowDQN.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_dqn.state_dim == dqn.state_dim
    assert new_dqn.action_dim == dqn.action_dim
    assert new_dqn.one_hot == dqn.one_hot
    assert new_dqn.net_config == dqn.net_config
    assert isinstance(new_dqn.actor, nn.Module)
    assert isinstance(new_dqn.actor_target, nn.Module)
    assert new_dqn.lr == dqn.lr
    assert str(new_dqn.actor.to("cpu").state_dict()) == str(dqn.actor.state_dict())
    assert str(new_dqn.actor_target.to("cpu").state_dict()) == str(
        dqn.actor_target.state_dict()
    )
    assert new_dqn.batch_size == dqn.batch_size
    assert new_dqn.learn_step == dqn.learn_step
    assert new_dqn.gamma == dqn.gamma
    assert new_dqn.tau == dqn.tau
    assert new_dqn.mut == dqn.mut
    assert new_dqn.index == dqn.index
    assert new_dqn.scores == dqn.scores
    assert new_dqn.fitness == dqn.fitness
    assert new_dqn.steps == dqn.steps
