from pathlib import Path

import dill
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer

from agilerl.algorithms.cqn import CQN
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


# initialize CQN with valid parameters
def test_initialize_cqn_with_minimum_parameters():
    state_dim = [4]
    action_dim = 2
    one_hot = False

    cqn = CQN(state_dim, action_dim, one_hot)

    assert cqn.state_dim == state_dim
    assert cqn.action_dim == action_dim
    assert cqn.one_hot == one_hot
    assert cqn.net_config == {"arch": "mlp", "h_size": [64, 64]}
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
    assert isinstance(cqn.actor, EvolvableMLP)
    assert isinstance(cqn.actor_target, EvolvableMLP)
    assert isinstance(cqn.optimizer_type, optim.Adam)
    assert cqn.arch == "mlp"
    assert cqn.optimizer == cqn.optimizer_type
    assert isinstance(cqn.criterion, nn.MSELoss)


# Initializes actor network with EvolvableCNN based on net_config and Accelerator.
def test_initialize_cqn_with_cnn_accelerator():
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
    mutation = None
    double = True
    actor_network = None
    accelerator = Accelerator()
    wrap = True

    cqn = CQN(
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
        mutation=mutation,
        double=double,
        actor_network=actor_network,
        accelerator=accelerator,
        wrap=wrap,
    )

    assert cqn.state_dim == state_dim
    assert cqn.action_dim == action_dim
    assert cqn.one_hot == one_hot
    assert cqn.net_config == net_config_cnn
    assert cqn.batch_size == batch_size
    assert cqn.lr == lr
    assert cqn.learn_step == learn_step
    assert cqn.gamma == gamma
    assert cqn.tau == tau
    assert cqn.mut == mutation
    assert cqn.accelerator == accelerator
    assert cqn.index == index
    assert cqn.scores == []
    assert cqn.fitness == []
    assert cqn.steps == [0]
    assert cqn.double is True
    assert cqn.actor_network is None
    assert isinstance(cqn.actor, EvolvableCNN)
    assert isinstance(cqn.actor_target, EvolvableCNN)
    assert isinstance(cqn.optimizer_type, optim.Adam)
    assert cqn.arch == "cnn"
    assert isinstance(cqn.optimizer, AcceleratedOptimizer)
    assert isinstance(cqn.criterion, nn.MSELoss)


# Can initialize cqn with an actor network
@pytest.mark.parametrize(
    "state_dim, actor_network, input_tensor",
    [
        ([4], "simple_mlp", torch.randn(1, 4)),
        ([3, 64, 64], "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
def test_initialize_cqn_with_actor_network(
    state_dim, actor_network, input_tensor, request
):
    action_dim = 2
    one_hot = False
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    cqn = CQN(state_dim, action_dim, one_hot, actor_network=actor_network)

    assert cqn.state_dim == state_dim
    assert cqn.action_dim == action_dim
    assert cqn.one_hot == one_hot
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
    assert isinstance(cqn.optimizer_type, optim.Adam)
    assert cqn.arch == actor_network.arch
    assert cqn.optimizer == cqn.optimizer_type
    assert isinstance(cqn.criterion, nn.MSELoss)


# Returns the expected action when given a state observation and epsilon=0 or 1.
def test_returns_expected_action_epsilon_greedy():
    state_dim = [4]
    action_dim = 2
    one_hot = False

    cqn = CQN(state_dim, action_dim, one_hot)
    state = np.array([1, 2, 3, 4])

    action_mask = None

    epsilon = 0
    action = cqn.getAction(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_dim

    epsilon = 1
    action = cqn.getAction(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_dim


# Returns the expected action when given a state observation and action mask.
def test_returns_expected_action_mask():
    accelerator = Accelerator()
    state_dim = [4]
    action_dim = 2
    one_hot = True

    cqn = CQN(state_dim, action_dim, one_hot, accelerator=accelerator)
    state = np.array([1])

    action_mask = np.array([0, 1])

    epsilon = 0
    action = cqn.getAction(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action == 1

    epsilon = 1
    action = cqn.getAction(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action == 1


# learns from experiences and updates network parameters
def test_learns_from_experiences():
    state_dim = [4]
    action_dim = 2
    one_hot = False
    batch_size = 64

    # Create an instance of the cqn class
    cqn = CQN(state_dim, action_dim, one_hot, batch_size=batch_size)

    # Create a batch of experiences
    states = torch.randn(batch_size, state_dim[0])
    actions = torch.randint(0, action_dim, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, state_dim[0])
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
    state_dim = [4]
    action_dim = 2
    one_hot = True
    double = True
    batch_size = 64

    # Create an instance of the cqn class
    cqn = CQN(
        state_dim,
        action_dim,
        one_hot,
        double=double,
        batch_size=batch_size,
        accelerator=accelerator,
    )

    # Create a batch of experiences
    states = torch.randint(0, state_dim[0], (batch_size, 1))
    actions = torch.randint(0, action_dim, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randint(0, state_dim[0], (batch_size, 1))
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
    state_dim = [4]
    action_dim = 2
    one_hot = False
    net_config = {"arch": "mlp", "h_size": [64, 64]}
    batch_size = 64
    lr = 1e-4
    learn_step = 5
    gamma = 0.99
    tau = 1e-3
    mutation = None
    double = False
    actor_network = None
    device = "cpu"
    accelerator = None
    wrap = True

    cqn = CQN(
        state_dim,
        action_dim,
        one_hot,
        net_config=net_config,
        batch_size=batch_size,
        lr=lr,
        learn_step=learn_step,
        gamma=gamma,
        tau=tau,
        mutation=mutation,
        double=double,
        actor_network=actor_network,
        device=device,
        accelerator=accelerator,
        wrap=wrap,
    )

    cqn.softUpdate()

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
    state_dim = (4,)
    action_dim = 2
    num_envs = 3

    env = DummyEnv(state_size=state_dim, vect=True, num_envs=num_envs)

    # env = makeVectEnvs("CartPole-v1", num_envs=num_envs)
    agent = CQN(state_dim=state_dim, action_dim=action_dim, one_hot=False)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorised env
def test_algorithm_test_loop_unvectorized():
    state_dim = (4,)
    action_dim = 2

    env = DummyEnv(state_size=state_dim, vect=False)

    agent = CQN(state_dim=state_dim, action_dim=action_dim, one_hot=False)
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

    agent = CQN(
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

    agent = CQN(
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

    cqn = CQN(state_dim, action_dim, one_hot)
    clone_agent = cqn.clone()

    assert clone_agent.state_dim == cqn.state_dim
    assert clone_agent.action_dim == cqn.action_dim
    assert clone_agent.one_hot == cqn.one_hot
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
    cqn = CQN(state_dim, action_dim, one_hot, accelerator=accelerator)
    clone_agent = cqn.clone()

    assert clone_agent.state_dim == cqn.state_dim
    assert clone_agent.action_dim == cqn.action_dim
    assert clone_agent.one_hot == cqn.one_hot
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
    cqn = CQN(state_dim, action_dim, one_hot, accelerator=accelerator, wrap=False)
    clone_agent = cqn.clone(wrap=False)

    assert clone_agent.state_dim == cqn.state_dim
    assert clone_agent.action_dim == cqn.action_dim
    assert clone_agent.one_hot == cqn.one_hot
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


# The method successfully unwraps the actor and actor_target models when an accelerator is present.
def test_unwrap_models():
    cqn = CQN(state_dim=[4], action_dim=2, one_hot=False, accelerator=Accelerator())
    cqn.unwrap_models()
    assert isinstance(cqn.actor, nn.Module)
    assert isinstance(cqn.actor_target, nn.Module)


# The saved checkpoint file contains the correct data and format.
def test_save_load_checkpoint_correct_data_and_format(tmpdir):
    # Initialize the cqn agent
    cqn = CQN(state_dim=[4], action_dim=2, one_hot=False)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    cqn.saveCheckpoint(checkpoint_path)

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
    assert "mutation" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    cqn = CQN(state_dim=[4], action_dim=2, one_hot=False)
    # Load checkpoint
    cqn.loadCheckpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert cqn.net_config == {"arch": "mlp", "h_size": [64, 64]}
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
        "h_size": [8],
        "c_size": [3],
        "k_size": [3],
        "s_size": [1],
        "normalize": False,
    }

    # Initialize the cqn agent
    cqn = CQN(
        state_dim=[3, 32, 32], action_dim=2, one_hot=False, net_config=net_config_cnn
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    cqn.saveCheckpoint(checkpoint_path)

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
    assert "mutation" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    cqn = CQN(state_dim=[4], action_dim=2, one_hot=False)
    # Load checkpoint
    cqn.loadCheckpoint(checkpoint_path)

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
        state_dim=[3, 64, 64], action_dim=2, one_hot=False, actor_network=actor_network
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    cqn.saveCheckpoint(checkpoint_path)

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
    assert "mutation" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    cqn = CQN(state_dim=[4], action_dim=2, one_hot=False)
    # Load checkpoint
    cqn.loadCheckpoint(checkpoint_path)

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
    cqn = CQN(state_dim=[4], action_dim=2, one_hot=False)

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    cqn.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_cqn = CQN.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_cqn.state_dim == cqn.state_dim
    assert new_cqn.action_dim == cqn.action_dim
    assert new_cqn.one_hot == cqn.one_hot
    assert new_cqn.net_config == cqn.net_config
    assert isinstance(cqn.actor, EvolvableMLP)
    assert isinstance(cqn.actor_target, EvolvableMLP)
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
    cqn.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_cqn = CQN.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_cqn.state_dim == cqn.state_dim
    assert new_cqn.action_dim == cqn.action_dim
    assert new_cqn.one_hot == cqn.one_hot
    assert new_cqn.net_config == cqn.net_config
    assert isinstance(cqn.actor, EvolvableCNN)
    assert isinstance(cqn.actor_target, EvolvableCNN)
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

    # Initialize the cqn agent
    cqn = CQN(
        state_dim=state_dim,
        action_dim=action_dim,
        one_hot=one_hot,
        actor_network=actor_network,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    cqn.saveCheckpoint(checkpoint_path)

    # Create new agent object
    new_cqn = CQN.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_cqn.state_dim == cqn.state_dim
    assert new_cqn.action_dim == cqn.action_dim
    assert new_cqn.one_hot == cqn.one_hot
    assert new_cqn.net_config == cqn.net_config
    assert isinstance(cqn.actor, nn.Module)
    assert isinstance(cqn.actor_target, nn.Module)
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
