from pathlib import Path

import dill
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces

from agilerl.algorithms.cqn import CQN
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.wrappers.make_evolvable import MakeEvolvable

class DummyCQN(CQN):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs)

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


# initialize CQN with valid parameters
def test_initialize_cqn_with_minimum_parameters():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)

    cqn = CQN(observation_space, action_space)

    assert cqn.observation_space == observation_space
    assert cqn.action_space == action_space
    assert cqn.net_config == {"arch": "mlp", "hidden_size": [64, 64]}
    assert cqn.batch_size == 64
    assert cqn.lr == 0.0001
    assert cqn.learn_step == 5
    assert cqn.gamma == 0.99
    assert cqn.tau == 0.001
    assert cqn.mut is None
    assert cqn.device == "cpu"
    assert cqn.accelerator is None
    assert cqn.index == 0
    assert cqn.scores == []
    assert cqn.fitness == []
    assert cqn.steps == [0]
    assert cqn.double is False
    assert isinstance(cqn.actor, EvolvableMLP)
    assert isinstance(cqn.actor_target, EvolvableMLP)
    assert isinstance(cqn.optimizer, optim.Adam)
    assert cqn.arch == "mlp"
    assert isinstance(cqn.criterion, nn.MSELoss)


# Initializes actor network with EvolvableCNN based on net_config and Accelerator.
def test_initialize_cqn_with_cnn_accelerator():
    observation_space = spaces.Box(0, 255, shape=(3, 32, 32))
    action_space = spaces.Discrete(2)
    index = 0
    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
    }
    batch_size = 64
    lr = 1e-4
    learn_step = 5
    gamma = 0.99
    tau = 1e-3
    mut = None
    double = True
    actor_network = None
    accelerator = Accelerator()
    wrap = True

    cqn = CQN(
        observation_space=observation_space,
        action_space=action_space,
        index=index,
        net_config=net_config_cnn,
        batch_size=batch_size,
        lr=lr,
        learn_step=learn_step,
        gamma=gamma,
        tau=tau,
        mut=mut,
        double=double,
        actor_network=actor_network,
        accelerator=accelerator,
        wrap=wrap,
    )

    assert cqn.observation_space == observation_space
    assert cqn.action_space == action_space
    assert cqn.net_config == net_config_cnn
    assert cqn.batch_size == batch_size
    assert cqn.lr == lr
    assert cqn.learn_step == learn_step
    assert cqn.gamma == gamma
    assert cqn.tau == tau
    assert cqn.mut == mut
    assert cqn.accelerator == accelerator
    assert cqn.index == index
    assert cqn.scores == []
    assert cqn.fitness == []
    assert cqn.steps == [0]
    assert cqn.double is True
    assert isinstance(cqn.actor, EvolvableCNN)
    assert isinstance(cqn.actor_target, EvolvableCNN)
    assert cqn.arch == "cnn"
    assert isinstance(cqn.optimizer, AcceleratedOptimizer)
    assert isinstance(cqn.criterion, nn.MSELoss)


# Can initialize cqn with an actor network
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (spaces.Box(0, 1, shape=(4,)), "simple_mlp", torch.randn(1, 4)),
        (spaces.Box(0, 255, shape=(3, 32, 32)), "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
def test_initialize_cqn_with_make_evo(observation_space, actor_network, input_tensor, request):
    action_space = spaces.Discrete(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    cqn = CQN(observation_space, action_space, actor_network=actor_network)

    assert cqn.observation_space == observation_space
    assert cqn.action_space == action_space
    assert cqn.net_config is None
    assert cqn.batch_size == 64
    assert cqn.lr == 0.0001
    assert cqn.learn_step == 5
    assert cqn.gamma == 0.99
    assert cqn.tau == 0.001
    assert cqn.mut is None
    assert cqn.device == "cpu"
    assert cqn.accelerator is None
    assert cqn.index == 0
    assert cqn.scores == []
    assert cqn.fitness == []
    assert cqn.steps == [0]
    assert cqn.double is False
    assert cqn.actor_network == actor_network
    assert cqn.actor == actor_network
    assert isinstance(cqn.optimizer, optim.Adam)
    assert cqn.arch == actor_network.arch
    assert isinstance(cqn.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_space, net_type",
    [
        (spaces.Box(0, 1, shape=(4,)), "mlp"),
        (spaces.Box(0, 255, shape=(3, 32, 32)), "cnn"),
    ],
)
def test_initialize_cqn_with_actor_network_evo_net(observation_space, net_type):
    action_space = spaces.Discrete(2)
    if net_type == "mlp":
        actor_network = EvolvableMLP(
            num_inputs=observation_space.shape[0],
            num_outputs=action_space.n,
            hidden_size=[64, 64],
            mlp_activation="ReLU",
        )
    else:
        actor_network = EvolvableCNN(
            input_shape=observation_space.shape,
            num_outputs=action_space.n,
            channel_size=[8, 8],
            kernel_size=[2, 2],
            stride_size=[1, 1],
            hidden_size=[64, 64],
            mlp_activation="ReLU",
        )

    cqn = CQN(observation_space, action_space, actor_network=actor_network)

    assert cqn.observation_space == observation_space
    assert cqn.action_space == action_space
    assert cqn.net_config is not None
    assert cqn.batch_size == 64
    assert cqn.lr == 0.0001
    assert cqn.learn_step == 5
    assert cqn.gamma == 0.99
    assert cqn.tau == 0.001
    assert cqn.mut is None
    assert cqn.device == "cpu"
    assert cqn.accelerator is None
    assert cqn.index == 0
    assert cqn.scores == []
    assert cqn.fitness == []
    assert cqn.steps == [0]
    assert cqn.double is False
    assert cqn.actor_network is None
    assert cqn.actor == actor_network
    assert isinstance(cqn.optimizer, optim.Adam)
    assert cqn.arch == actor_network.arch
    assert isinstance(cqn.criterion, nn.MSELoss)


def test_init_with_incorrect_actor_net():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)
    actor_network = "String"

    with pytest.raises(AssertionError) as e:
        cqn = CQN(observation_space, action_space, actor_network=actor_network)
        assert cqn
        assert (
            e
            == "'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableMLP, EvolvableCNN or MakeEvolvable"
        )


# Returns the expected action when given a state observation and epsilon=0 or 1.
def test_returns_expected_action_epsilon_greedy():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)

    cqn = CQN(observation_space, action_space)
    state = np.array([1, 2, 3, 4])

    action_mask = None

    epsilon = 0
    action = cqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_space.n

    epsilon = 1
    action = cqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_space.n

# Returns the expected action when given a state observation and action mask.
def test_returns_expected_action_mask():
    accelerator = Accelerator()
    observation_space = spaces.Discrete(4)
    action_space = spaces.Discrete(2)

    cqn = CQN(observation_space, action_space, accelerator=accelerator)
    state = np.array([1])

    action_mask = np.array([0, 1])

    epsilon = 0
    action = cqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action == 1

    epsilon = 1
    action = cqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action == 1


# learns from experiences and updates network parameters
def test_learns_from_experiences():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)
    batch_size = 64

    # Create an instance of the cqn class
    cqn = CQN(observation_space, action_space, batch_size=batch_size)

    # Create a batch of experiences
    states = torch.randn(batch_size, observation_space.shape[0])
    actions = torch.randint(0, action_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, observation_space.shape[0])
    dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [states, actions, rewards, next_states, dones]

    # Copy state dict before learning - should be different to after updating weights
    actor = cqn.actor
    actor_target = cqn.actor_target
    actor_pre_learn_sd = str(cqn.actor.state_dict())
    actor_target_pre_learn_sd = str(cqn.actor_target.state_dict())

    # Call the learn method
    cqn.learn(experiences)

    assert actor == cqn.actor
    assert actor_target == cqn.actor_target
    assert actor_pre_learn_sd != str(cqn.actor.state_dict())
    assert actor_target_pre_learn_sd != str(cqn.actor_target.state_dict())


# handles double Q-learning
def test_handles_double_q_learning():
    accelerator = Accelerator()
    observation_space = spaces.Discrete(4)
    action_space = spaces.Discrete(2)
    double = True
    batch_size = 64

    # Create an instance of the cqn class
    cqn = CQN(
        observation_space,
        action_space,
        double=double,
        batch_size=batch_size,
        accelerator=accelerator,
    )

    # Create a batch of experiences
    states = torch.randint(0, observation_space.n, (batch_size, 1))
    actions = torch.randint(0, action_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randint(0, observation_space.n, (batch_size, 1))
    dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [states, actions, rewards, next_states, dones]

    # Copy state dict before learning - should be different to after updating weights
    actor = cqn.actor
    actor_target = cqn.actor_target
    actor_pre_learn_sd = str(cqn.actor.state_dict())
    actor_target_pre_learn_sd = str(cqn.actor_target.state_dict())

    # Call the learn method
    cqn.learn(experiences)

    assert actor == cqn.actor
    assert actor_target == cqn.actor_target
    assert actor_pre_learn_sd != str(cqn.actor.state_dict())
    assert actor_target_pre_learn_sd != str(cqn.actor_target.state_dict())


# Updates target network parameters with soft update
def test_soft_update():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)
    net_config = {"arch": "mlp", "hidden_size": [64, 64]}
    batch_size = 64
    lr = 1e-4
    learn_step = 5
    gamma = 0.99
    tau = 1e-3
    mut = None
    double = False
    actor_network = None
    device = "cpu"
    accelerator = None
    wrap = True

    cqn = CQN(
        observation_space,
        action_space,
        net_config=net_config,
        batch_size=batch_size,
        lr=lr,
        learn_step=learn_step,
        gamma=gamma,
        tau=tau,
        mut=mut,
        double=double,
        actor_network=actor_network,
        device=device,
        accelerator=accelerator,
        wrap=wrap,
    )

    cqn.soft_update()

    eval_params = list(cqn.actor.parameters())
    target_params = list(cqn.actor_target.parameters())
    expected_params = [
        cqn.tau * eval_param + (1.0 - cqn.tau) * target_param
        for eval_param, target_param in zip(eval_params, target_params)
    ]

    assert all(
        torch.allclose(expected_param, target_param)
        for expected_param, target_param in zip(expected_params, target_params)
    )

# Runs algorithm test loop
def test_algorithm_test_loop():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)
    num_envs = 3

    env = DummyEnv(state_size=observation_space.shape, vect=True, num_envs=num_envs)

    # env = make_vect_envs("CartPole-v1", num_envs=num_envs)
    agent = CQN(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorised env
def test_algorithm_test_loop_unvectorized():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)

    env = DummyEnv(state_size=observation_space.shape, vect=False)

    agent = CQN(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with images
def test_algorithm_test_loop_images():
    observation_space = spaces.Box(0, 1, shape= (3, 32, 32))
    action_space = spaces.Discrete(2)

    env = DummyEnv(state_size=observation_space.shape, vect=True)

    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
    }

    agent = CQN(
        observation_space=observation_space,
        action_space=action_space,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorized images
def test_algorithm_test_loop_images_unvectorized():
    observation_space = spaces.Box(0, 1, shape= (32, 32, 3))
    action_space = spaces.Discrete(2)

    env = DummyEnv(state_size=observation_space.shape, vect=False)

    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
    }

    agent = CQN(
        observation_space=spaces.Box(0, 1, shape= (3, 32, 32)),
        action_space=action_space,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10, swap_channels=True)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
def test_clone_returns_identical_agent():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)

    cqn = DummyCQN(observation_space, action_space)
    cqn.tensor_attribute = torch.randn(1)
    clone_agent = cqn.clone()

    assert clone_agent.observation_space == cqn.observation_space
    assert clone_agent.action_space == cqn.action_space
    assert clone_agent.net_config == cqn.net_config
    assert clone_agent.actor_network == cqn.actor_network
    assert clone_agent.batch_size == cqn.batch_size
    assert clone_agent.lr == cqn.lr
    assert clone_agent.learn_step == cqn.learn_step
    assert clone_agent.gamma == cqn.gamma
    assert clone_agent.tau == cqn.tau
    assert clone_agent.mut == cqn.mut
    assert clone_agent.device == cqn.device
    assert clone_agent.accelerator == cqn.accelerator
    assert str(clone_agent.actor.state_dict()) == str(cqn.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        cqn.actor_target.state_dict()
    )
    assert str(clone_agent.optimizer.state_dict()) == str(cqn.optimizer.state_dict())
    assert clone_agent.fitness == cqn.fitness
    assert clone_agent.steps == cqn.steps
    assert clone_agent.scores == cqn.scores
    assert clone_agent.tensor_attribute == cqn.tensor_attribute
    assert clone_agent.tensor_test == cqn.tensor_test

    accelerator = Accelerator()
    cqn = CQN(observation_space, action_space, accelerator=accelerator)
    clone_agent = cqn.clone()

    assert clone_agent.observation_space == cqn.observation_space
    assert clone_agent.action_space == cqn.action_space
    assert clone_agent.net_config == cqn.net_config
    assert clone_agent.actor_network == cqn.actor_network
    assert clone_agent.batch_size == cqn.batch_size
    assert clone_agent.lr == cqn.lr
    assert clone_agent.learn_step == cqn.learn_step
    assert clone_agent.gamma == cqn.gamma
    assert clone_agent.tau == cqn.tau
    assert clone_agent.mut == cqn.mut
    assert clone_agent.device == cqn.device
    assert clone_agent.accelerator == cqn.accelerator
    assert str(clone_agent.actor.state_dict()) == str(cqn.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        cqn.actor_target.state_dict()
    )
    assert str(clone_agent.optimizer.state_dict()) == str(cqn.optimizer.state_dict())
    assert clone_agent.fitness == cqn.fitness
    assert clone_agent.steps == cqn.steps
    assert clone_agent.scores == cqn.scores

    accelerator = Accelerator()
    cqn = CQN(observation_space, action_space, accelerator=accelerator, wrap=False)
    clone_agent = cqn.clone(wrap=False)

    assert clone_agent.observation_space == cqn.observation_space
    assert clone_agent.action_space == cqn.action_space
    assert clone_agent.net_config == cqn.net_config
    assert clone_agent.actor_network == cqn.actor_network
    assert clone_agent.batch_size == cqn.batch_size
    assert clone_agent.lr == cqn.lr
    assert clone_agent.learn_step == cqn.learn_step
    assert clone_agent.gamma == cqn.gamma
    assert clone_agent.tau == cqn.tau
    assert clone_agent.mut == cqn.mut
    assert clone_agent.device == cqn.device
    assert clone_agent.accelerator == cqn.accelerator
    assert str(clone_agent.actor.state_dict()) == str(cqn.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        cqn.actor_target.state_dict()
    )
    assert str(clone_agent.optimizer.state_dict()) == str(cqn.optimizer.state_dict())
    assert clone_agent.fitness == cqn.fitness
    assert clone_agent.steps == cqn.steps
    assert clone_agent.scores == cqn.scores


def test_clone_new_index():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)

    cqn = CQN(observation_space, action_space)
    clone_agent = cqn.clone(index=100)

    assert clone_agent.index == 100


# The method successfully unwraps the actor and actor_target models when an accelerator is present.
def test_unwrap_models():
    cqn = CQN(observation_space=spaces.Box(0, 1, shape=(4,)), action_space=spaces.Discrete(2), accelerator=Accelerator())
    cqn.unwrap_models()
    assert isinstance(cqn.actor, nn.Module)
    assert isinstance(cqn.actor_target, nn.Module)


# The saved checkpoint file contains the correct data and format.
def test_save_load_checkpoint_correct_data_and_format(tmpdir):
    # Initialize the cqn agent
    cqn = CQN(observation_space=spaces.Box(0, 1, shape=(4,)), action_space=spaces.Discrete(2))

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    cqn.save_checkpoint(checkpoint_path)

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

    cqn = CQN(observation_space=spaces.Box(0, 1, shape=(4,)), action_space=spaces.Discrete(2))
    # Load checkpoint
    cqn.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert cqn.net_config == {"arch": "mlp", "hidden_size": [64, 64]}
    assert isinstance(cqn.actor, EvolvableMLP)
    assert isinstance(cqn.actor_target, EvolvableMLP)
    assert cqn.lr == 1e-4
    assert str(cqn.actor.state_dict()) == str(cqn.actor_target.state_dict())
    assert cqn.batch_size == 64
    assert cqn.learn_step == 5
    assert cqn.gamma == 0.99
    assert cqn.tau == 1e-3
    assert cqn.mut is None
    assert cqn.index == 0
    assert cqn.scores == []
    assert cqn.fitness == []
    assert cqn.steps == [0]


def test_save_load_checkpoint_correct_data_and_format_cnn(tmpdir):
    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
    }

    # Initialize the cqn agent
    cqn = CQN(
        observation_space=spaces.Box(0, 255, shape=(3, 32, 32)), action_space=spaces.Discrete(2), net_config=net_config_cnn
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    cqn.save_checkpoint(checkpoint_path)

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

    cqn = CQN(observation_space=spaces.Box(0, 1, shape=(4,)), action_space=spaces.Discrete(2))
    # Load checkpoint
    cqn.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert cqn.net_config == net_config_cnn
    assert isinstance(cqn.actor, EvolvableCNN)
    assert isinstance(cqn.actor_target, EvolvableCNN)
    assert cqn.lr == 1e-4
    assert str(cqn.actor.state_dict()) == str(cqn.actor_target.state_dict())
    assert cqn.batch_size == 64
    assert cqn.learn_step == 5
    assert cqn.gamma == 0.99
    assert cqn.tau == 1e-3
    assert cqn.mut is None
    assert cqn.index == 0
    assert cqn.scores == []
    assert cqn.fitness == []
    assert cqn.steps == [0]


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

    # Initialize the cqn agent
    cqn = CQN(
        observation_space=spaces.Box(0, 255, shape=(3, 32, 32)), action_space=spaces.Discrete(2), actor_network=actor_network
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    cqn.save_checkpoint(checkpoint_path)

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

    cqn = CQN(observation_space=spaces.Box(0, 1, shape=(4,)), action_space=spaces.Discrete(2))
    # Load checkpoint
    cqn.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert cqn.net_config is None
    assert isinstance(cqn.actor, nn.Module)
    assert isinstance(cqn.actor_target, nn.Module)
    assert cqn.lr == 1e-4
    assert str(cqn.actor.state_dict()) == str(cqn.actor_target.state_dict())
    assert cqn.batch_size == 64
    assert cqn.learn_step == 5
    assert cqn.gamma == 0.99
    assert cqn.tau == 1e-3
    assert cqn.mut is None
    assert cqn.index == 0
    assert cqn.scores == []
    assert cqn.fitness == []
    assert cqn.steps == [0]


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained(device, accelerator, tmpdir):
    # Initialize the cqn agent
    cqn = CQN(observation_space=spaces.Box(0, 1, shape=(4,)), action_space=spaces.Discrete(2))

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    cqn.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_cqn = CQN.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_cqn.observation_space == cqn.observation_space
    assert new_cqn.action_space == cqn.action_space
    assert new_cqn.net_config == cqn.net_config
    assert isinstance(new_cqn.actor, EvolvableMLP)
    assert isinstance(new_cqn.actor_target, EvolvableMLP)
    assert new_cqn.lr == cqn.lr
    assert str(new_cqn.actor.to("cpu").state_dict()) == str(cqn.actor.state_dict())
    assert str(new_cqn.actor_target.to("cpu").state_dict()) == str(
        cqn.actor_target.state_dict()
    )
    assert new_cqn.batch_size == cqn.batch_size
    assert new_cqn.learn_step == cqn.learn_step
    assert new_cqn.gamma == cqn.gamma
    assert new_cqn.tau == cqn.tau
    assert new_cqn.mut == cqn.mut
    assert new_cqn.index == cqn.index
    assert new_cqn.scores == cqn.scores
    assert new_cqn.fitness == cqn.fitness
    assert new_cqn.steps == cqn.steps


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_cnn(device, accelerator, tmpdir):
    # Initialize the cqn agent
    cqn = CQN(
        observation_space=spaces.Box(0, 255, shape=(3, 32, 32)),
        action_space=spaces.Discrete(2),
        net_config={
            "arch": "cnn",
            "hidden_size": [8],
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        },
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    cqn.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_cqn = CQN.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_cqn.observation_space == cqn.observation_space
    assert new_cqn.action_space == cqn.action_space
    assert new_cqn.net_config == cqn.net_config
    assert isinstance(new_cqn.actor, EvolvableCNN)
    assert isinstance(new_cqn.actor_target, EvolvableCNN)
    assert new_cqn.lr == cqn.lr
    assert str(new_cqn.actor.to("cpu").state_dict()) == str(cqn.actor.state_dict())
    assert str(new_cqn.actor_target.to("cpu").state_dict()) == str(
        cqn.actor_target.state_dict()
    )
    assert new_cqn.batch_size == cqn.batch_size
    assert new_cqn.learn_step == cqn.learn_step
    assert new_cqn.gamma == cqn.gamma
    assert new_cqn.tau == cqn.tau
    assert new_cqn.mut == cqn.mut
    assert new_cqn.index == cqn.index
    assert new_cqn.scores == cqn.scores
    assert new_cqn.fitness == cqn.fitness
    assert new_cqn.steps == cqn.steps


@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (spaces.Box(0, 1, shape=(4,)), "simple_mlp", torch.randn(1, 4)),
        (spaces.Box(0, 255, shape=(3, 32, 32)), "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_networks(
    observation_space, actor_network, input_tensor, request, tmpdir
):
    action_space = spaces.Discrete(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    # Initialize the cqn agent
    cqn = CQN(
        observation_space=observation_space,
        action_space=action_space,
        actor_network=actor_network,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    cqn.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_cqn = CQN.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_cqn.observation_space == cqn.observation_space
    assert new_cqn.action_space == cqn.action_space
    assert new_cqn.net_config == cqn.net_config
    assert isinstance(new_cqn.actor, nn.Module)
    assert isinstance(new_cqn.actor_target, nn.Module)
    assert new_cqn.lr == cqn.lr
    assert str(new_cqn.actor.to("cpu").state_dict()) == str(cqn.actor.state_dict())
    assert str(new_cqn.actor_target.to("cpu").state_dict()) == str(
        cqn.actor_target.state_dict()
    )
    assert new_cqn.batch_size == cqn.batch_size
    assert new_cqn.learn_step == cqn.learn_step
    assert new_cqn.gamma == cqn.gamma
    assert new_cqn.tau == cqn.tau
    assert new_cqn.mut == cqn.mut
    assert new_cqn.index == cqn.index
    assert new_cqn.scores == cqn.scores
    assert new_cqn.fitness == cqn.fitness
    assert new_cqn.steps == cqn.steps
