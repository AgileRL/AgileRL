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
from gymnasium import spaces

from agilerl.algorithms.neural_ucb_bandit import NeuralUCB
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.wrappers.make_evolvable import MakeEvolvable


@pytest.fixture(autouse=True)
def cleanup():
    yield  # Run the test first
    torch.cuda.empty_cache()  # Free up GPU memory


class DummyNeuralUCB(NeuralUCB):
    def __init__(
        self,
        observation_space,
        action_space,
        net_config=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            net_config=net_config,
            *args,
            **kwargs,
        )

        self.tensor_test = torch.randn(1)
        self.numpy_test = np.random.rand(1)


class DummyBanditEnv:
    def __init__(self, state_size, arms):
        self.arms = arms
        self.state_size = (arms,) + state_size
        self.n_envs = 1

    def reset(self):
        return np.random.rand(*self.state_size)

    def step(self, action):
        return (
            np.random.rand(*self.state_size),
            np.random.rand(self.n_envs),
        )


@pytest.fixture
def simple_mlp():
    network = nn.Sequential(
        nn.Linear(4, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )
    return network


@pytest.fixture
def simple_cnn():
    network = nn.Sequential(
        nn.Conv2d(
            3, 4, kernel_size=3, stride=1, padding=1
        ),  # Input channels: 3 (for RGB images), Output channels: 16
        nn.ReLU(),
        nn.Conv2d(
            4, 4, kernel_size=3, stride=1, padding=1
        ),  # Input channels: 16, Output channels: 32
        nn.ReLU(),
        nn.Flatten(),  # Flatten the 2D feature map to a 1D vector
        nn.Linear(32 * 4 * 4 * 32, 8),  # Fully connected layer with 128 output features
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    return network


# initialize NeuralUCB with valid parameters
def test_initialize_bandit_with_minimum_parameters():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)

    bandit = NeuralUCB(observation_space, action_space)

    assert bandit.observation_space == observation_space
    assert bandit.action_space == action_space
    assert bandit.batch_size == 64
    assert bandit.lr == 0.001
    assert bandit.learn_step == 2
    assert bandit.gamma == 1.0
    assert bandit.lamb == 1.0
    assert bandit.reg == 0.000625
    assert bandit.mut is None
    assert bandit.device == "cpu"
    assert bandit.accelerator is None
    assert bandit.index == 0
    assert bandit.scores == []
    assert bandit.fitness == []
    assert bandit.steps == [0]
    assert isinstance(bandit.actor.encoder, EvolvableMLP)
    assert isinstance(bandit.optimizer.optimizer, optim.Adam)
    assert isinstance(bandit.criterion, nn.MSELoss)


# Initializes actor network with EvolvableCNN based on net_config and Accelerator.
def test_initialize_bandit_with_cnn_accelerator():
    observation_space = spaces.Box(0, 1, shape=(3, 32, 32))
    action_space = spaces.Discrete(2)
    index = 0
    net_config_cnn = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }
    batch_size = 64
    lr = 1e-3
    learn_step = 5
    gamma = 1.0
    lamb = 1.0
    reg = 0.000625
    mut = None
    actor_network = None
    accelerator = Accelerator()
    wrap = True

    bandit = NeuralUCB(
        observation_space=observation_space,
        action_space=action_space,
        index=index,
        net_config=net_config_cnn,
        batch_size=batch_size,
        lr=lr,
        learn_step=learn_step,
        gamma=gamma,
        lamb=lamb,
        reg=reg,
        mut=mut,
        actor_network=actor_network,
        accelerator=accelerator,
        wrap=wrap,
    )

    assert bandit.observation_space == observation_space
    assert bandit.action_space == action_space
    assert bandit.batch_size == batch_size
    assert bandit.lr == lr
    assert bandit.learn_step == learn_step
    assert bandit.gamma == gamma
    assert bandit.lamb == lamb
    assert bandit.reg == reg
    assert bandit.mut == mut
    assert bandit.accelerator == accelerator
    assert bandit.index == index
    assert bandit.scores == []
    assert bandit.fitness == []
    assert bandit.steps == [0]
    assert isinstance(bandit.actor.encoder, EvolvableCNN)
    assert isinstance(bandit.optimizer.optimizer, AcceleratedOptimizer)
    assert isinstance(bandit.criterion, nn.MSELoss)


# Can initialize NeuralUCB with an actor network
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (spaces.Box(0, 1, shape=(4,)), "simple_mlp", torch.randn(1, 4)),
        (spaces.Box(0, 1, shape=(3, 64, 64)), "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
def test_initialize_bandit_with_actor_network(
    observation_space, actor_network, input_tensor, request
):
    action_space = spaces.Discrete(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    bandit = NeuralUCB(observation_space, action_space, actor_network=actor_network)

    assert bandit.observation_space == observation_space
    assert bandit.action_space == action_space
    assert bandit.batch_size == 64
    assert bandit.lr == 0.001
    assert bandit.learn_step == 2
    assert bandit.gamma == 1.0
    assert bandit.lamb == 1.0
    assert bandit.reg == 0.000625
    assert bandit.mut is None
    assert bandit.device == "cpu"
    assert bandit.accelerator is None
    assert bandit.index == 0
    assert bandit.scores == []
    assert bandit.fitness == []
    assert bandit.steps == [0]
    assert isinstance(bandit.optimizer.optimizer, optim.Adam)
    assert isinstance(bandit.criterion, nn.MSELoss)


def test_initialize_bandit_with_incorrect_actor_network():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)
    actor_network = nn.Sequential(nn.Linear(observation_space.shape[0], action_space.n))

    with pytest.raises(TypeError) as e:
        bandit = NeuralUCB(observation_space, action_space, actor_network=actor_network)

        assert bandit
        assert (
            e
            == "'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableModule."
        )


def test_initialize_bandit_with_evo_nets():  #
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)
    actor_network = EvolvableMLP(
        num_inputs=observation_space.shape[0],
        num_outputs=1,
        hidden_size=[64, 64],
        layer_norm=False,
    )

    bandit = NeuralUCB(observation_space, action_space, actor_network=actor_network)
    assert bandit.observation_space == observation_space
    assert bandit.action_space == action_space
    assert bandit.batch_size == 64
    assert bandit.lr == 0.001
    assert bandit.learn_step == 2
    assert bandit.gamma == 1.0
    assert bandit.lamb == 1.0
    assert bandit.reg == 0.000625
    assert bandit.mut is None
    assert bandit.device == "cpu"
    assert bandit.accelerator is None
    assert bandit.index == 0
    assert bandit.scores == []
    assert bandit.fitness == []
    assert bandit.steps == [0]
    assert isinstance(bandit.optimizer.optimizer, optim.Adam)
    assert isinstance(bandit.criterion, nn.MSELoss)


def test_initialize_neuralucb_with_incorrect_actor_net_type():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)
    actor_network = "dummy"

    with pytest.raises(TypeError) as a:
        bandit = NeuralUCB(observation_space, action_space, actor_network=actor_network)
        assert bandit
        assert (
            str(a.value)
            == f"'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableMLP, EvolvableCNN or MakeEvolvable"
        )


# Returns the expected action when given a state observation and epsilon=0 or 1.
def test_returns_expected_action():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)

    bandit = NeuralUCB(observation_space, action_space)
    state = np.array([1, 2, 3, 4])

    action_mask = None

    action = bandit.get_action(state, action_mask)

    assert action.is_integer()
    assert action >= 0 and action < action_space.n


# Returns the expected action when given a state observation and action mask.
def test_returns_expected_action_mask():
    accelerator = Accelerator()
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)

    bandit = NeuralUCB(observation_space, action_space, accelerator=accelerator)
    state = np.array([1, 2, 3, 4])

    action_mask = np.array([0, 1])

    action = bandit.get_action(state, action_mask)

    assert action.is_integer()
    assert action == 1


# learns from experiences and updates network parameters
def test_learns_from_experiences():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)
    batch_size = 64

    # Create an instance of the NeuralUCB class
    bandit = NeuralUCB(observation_space, action_space, batch_size=batch_size)

    # Create a batch of experiences
    states = torch.randn(batch_size, *observation_space.shape)
    rewards = torch.randn((batch_size, 1))

    experiences = [states, rewards]

    # Copy state dict before learning - should be different to after updating weights
    actor = bandit.actor
    actor_pre_learn_sd = str(copy.deepcopy(bandit.actor.state_dict()))

    # Call the learn method
    loss = bandit.learn(experiences)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert actor == bandit.actor
    assert actor_pre_learn_sd != str(bandit.actor.state_dict())


# learns from experiences and updates network parameters
def test_learns_from_experiences_if_cuda():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)
    batch_size = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the NeuralUCB class
    bandit = NeuralUCB(
        observation_space, action_space, batch_size=batch_size, device=device
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, *observation_space.shape).to(device)
    rewards = torch.randn((batch_size, 1)).to(device)

    experiences = [states, rewards]

    # Copy state dict before learning - should be different to after updating weights
    actor = bandit.actor
    actor_pre_learn_sd = str(copy.deepcopy(bandit.actor.state_dict()))

    # Call the learn method
    loss = bandit.learn(experiences)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert actor == bandit.actor
    assert actor_pre_learn_sd != str(bandit.actor.state_dict())


def test_learning_accelerator():
    accelerator = Accelerator()
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)
    batch_size = 64

    # Create an instance of the NeuralUCB class
    bandit = NeuralUCB(
        observation_space,
        action_space,
        batch_size=batch_size,
        accelerator=accelerator,
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, *observation_space.shape)
    rewards = torch.randn((batch_size, 1))

    experiences = [states, rewards]

    # Copy state dict before learning - should be different to after updating weights
    actor = bandit.actor
    actor_pre_learn_sd = str(copy.deepcopy(bandit.actor.state_dict()))

    # Call the learn method
    loss = bandit.learn(experiences)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert actor == bandit.actor
    assert actor_pre_learn_sd != str(bandit.actor.state_dict())


def test_learning_cnn():
    observation_space = spaces.Box(0, 1, shape=(3, 32, 32))
    action_space = spaces.Discrete(2)
    batch_size = 64
    net_config = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }

    # Create an instance of the NeuralUCB class
    bandit = NeuralUCB(
        observation_space,
        action_space,
        net_config=net_config,
        batch_size=batch_size,
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, *observation_space.shape)
    rewards = torch.randn((batch_size, 1))

    experiences = [states, rewards]

    # Copy state dict before learning - should be different to after updating weights
    actor = bandit.actor
    actor_pre_learn_sd = str(copy.deepcopy(bandit.actor.state_dict()))

    # Call the learn method
    loss = bandit.learn(experiences)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert actor == bandit.actor
    assert actor_pre_learn_sd != str(bandit.actor.state_dict())


# Runs algorithm test loop
def test_algorithm_test_loop():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)

    env = DummyBanditEnv(state_size=observation_space.shape, arms=action_space.n)

    agent = NeuralUCB(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with images
def test_algorithm_test_loop_images():
    observation_space = spaces.Box(0, 1, shape=(32, 32, 3))
    action_space = spaces.Discrete(2)

    env = DummyBanditEnv(state_size=observation_space.shape, arms=action_space.n)

    net_config_cnn = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }

    agent = NeuralUCB(
        observation_space=spaces.Box(0, 1, shape=(3, 32, 32)),
        action_space=action_space,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10, swap_channels=True)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
@pytest.mark.parametrize(
    "observation_space, net_config",
    [
        (
            spaces.Box(0, 1, shape=(3, 32, 32)),
            {
                "encoder_config": {
                    "channel_size": [3],
                    "kernel_size": [3],
                    "stride_size": [1],
                }
            },
        ),
        (spaces.Box(0, 1, shape=(4,)), {"encoder_config": {"hidden_size": [128]}}),
    ],
)
def test_clone_returns_identical_agent(observation_space, net_config):
    action_space = spaces.Discrete(2)

    bandit = DummyNeuralUCB(observation_space, action_space, net_config)
    print(bandit.actor)
    bandit.tensor_attribute = torch.randn(1)
    bandit.numpy_attribute = np.random.rand(1)
    clone_agent = bandit.clone()

    assert clone_agent.observation_space == bandit.observation_space
    assert clone_agent.action_space == bandit.action_space
    assert clone_agent.batch_size == bandit.batch_size
    assert clone_agent.lr == bandit.lr
    assert clone_agent.learn_step == bandit.learn_step
    assert clone_agent.gamma == bandit.gamma
    assert clone_agent.mut == bandit.mut
    assert clone_agent.device == bandit.device
    assert clone_agent.accelerator == bandit.accelerator
    assert str(clone_agent.actor.state_dict()) == str(bandit.actor.state_dict())
    assert str(clone_agent.optimizer.state_dict()) == str(bandit.optimizer.state_dict())
    assert clone_agent.fitness == bandit.fitness
    assert clone_agent.steps == bandit.steps
    assert clone_agent.scores == bandit.scores
    assert clone_agent.tensor_attribute == bandit.tensor_attribute
    assert clone_agent.tensor_test == bandit.tensor_test
    assert clone_agent.numpy_attribute == bandit.numpy_attribute
    assert clone_agent.numpy_test == bandit.numpy_test

    accelerator = Accelerator()
    bandit = NeuralUCB(observation_space, action_space, accelerator=accelerator)
    clone_agent = bandit.clone()

    assert clone_agent.observation_space == bandit.observation_space
    assert clone_agent.action_space == bandit.action_space
    assert clone_agent.batch_size == bandit.batch_size
    assert clone_agent.lr == bandit.lr
    assert clone_agent.learn_step == bandit.learn_step
    assert clone_agent.gamma == bandit.gamma
    assert clone_agent.mut == bandit.mut
    assert clone_agent.device == bandit.device
    assert clone_agent.accelerator == bandit.accelerator
    assert str(clone_agent.actor.state_dict()) == str(bandit.actor.state_dict())
    assert str(clone_agent.optimizer.state_dict()) == str(bandit.optimizer.state_dict())
    assert clone_agent.fitness == bandit.fitness
    assert clone_agent.steps == bandit.steps
    assert clone_agent.scores == bandit.scores

    accelerator = Accelerator()
    bandit = NeuralUCB(
        observation_space, action_space, accelerator=accelerator, wrap=False
    )
    clone_agent = bandit.clone(wrap=False)

    assert clone_agent.observation_space == bandit.observation_space
    assert clone_agent.action_space == bandit.action_space
    assert clone_agent.batch_size == bandit.batch_size
    assert clone_agent.lr == bandit.lr
    assert clone_agent.learn_step == bandit.learn_step
    assert clone_agent.gamma == bandit.gamma
    assert clone_agent.mut == bandit.mut
    assert clone_agent.device == bandit.device
    assert clone_agent.accelerator == bandit.accelerator
    assert str(clone_agent.actor.state_dict()) == str(bandit.actor.state_dict())
    assert str(clone_agent.optimizer.state_dict()) == str(bandit.optimizer.state_dict())
    assert clone_agent.fitness == bandit.fitness
    assert clone_agent.steps == bandit.steps
    assert clone_agent.scores == bandit.scores


def test_clone_new_index():
    observation_space = spaces.Box(0, 1, shape=(4,))
    action_space = spaces.Discrete(2)

    bandit = NeuralUCB(observation_space, action_space)
    clone_agent = bandit.clone(index=100)

    assert clone_agent.index == 100


def test_clone_after_learning():
    action_space = spaces.Discrete(2)
    observation_space = spaces.Box(0, 1, shape=(4,))
    batch_size = 4
    states = torch.randn(batch_size, observation_space.shape[0])
    rewards = torch.rand(batch_size, 1)
    experiences = states, rewards
    bandit = NeuralUCB(observation_space, action_space, batch_size=batch_size)
    bandit.learn(experiences)
    clone_agent = bandit.clone()

    assert clone_agent.observation_space == bandit.observation_space
    assert clone_agent.action_space == bandit.action_space
    assert clone_agent.batch_size == bandit.batch_size
    assert clone_agent.lr == bandit.lr
    assert clone_agent.learn_step == bandit.learn_step
    assert clone_agent.gamma == bandit.gamma
    assert clone_agent.mut == bandit.mut
    assert clone_agent.device == bandit.device
    assert clone_agent.accelerator == bandit.accelerator
    assert str(clone_agent.actor.state_dict()) == str(bandit.actor.state_dict())
    assert str(clone_agent.optimizer.state_dict()) == str(bandit.optimizer.state_dict())
    assert clone_agent.fitness == bandit.fitness
    assert clone_agent.steps == bandit.steps
    assert clone_agent.scores == bandit.scores


@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (spaces.Box(0, 1, shape=(4,)), "simple_mlp", torch.randn(1, 4)),
        (spaces.Box(0, 1, shape=(3, 64, 64)), "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
def test_clone_with_make_evo(observation_space, actor_network, input_tensor, request):
    action_space = spaces.Discrete(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    bandit = NeuralUCB(observation_space, action_space, actor_network=actor_network)
    clone_agent = bandit.clone()

    assert clone_agent.observation_space == bandit.observation_space
    assert clone_agent.action_space == bandit.action_space
    assert clone_agent.batch_size == bandit.batch_size
    assert clone_agent.lr == bandit.lr
    assert clone_agent.learn_step == bandit.learn_step
    assert clone_agent.gamma == bandit.gamma
    assert clone_agent.mut == bandit.mut
    assert clone_agent.device == bandit.device
    assert clone_agent.accelerator == bandit.accelerator
    assert str(clone_agent.actor.state_dict()) == str(bandit.actor.state_dict())
    assert str(clone_agent.optimizer.state_dict()) == str(bandit.optimizer.state_dict())
    assert clone_agent.fitness == bandit.fitness
    assert clone_agent.steps == bandit.steps
    assert clone_agent.scores == bandit.scores


# The method successfully unwraps the actor model when an accelerator is present.
def test_unwrap_models():
    bandit = NeuralUCB(
        observation_space=spaces.Box(0, 1, shape=(4,)),
        action_space=spaces.Discrete(2),
        accelerator=Accelerator(),
    )
    bandit.unwrap_models()
    assert isinstance(bandit.actor, nn.Module)


# The saved checkpoint file contains the correct data and format.
def test_save_load_checkpoint_correct_data_and_format(tmpdir):
    # Initialize the NeuralUCB agent
    bandit = NeuralUCB(
        observation_space=spaces.Box(0, 1, shape=(4,)), action_space=spaces.Discrete(2)
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    bandit.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    bandit = NeuralUCB(
        observation_space=spaces.Box(0, 1, shape=(4,)), action_space=spaces.Discrete(2)
    )
    # Load checkpoint
    bandit.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(bandit.actor.encoder, EvolvableMLP)
    assert bandit.lr == 1e-3
    assert bandit.batch_size == 64
    assert bandit.learn_step == 2
    assert bandit.gamma == 1.0
    assert bandit.lamb == 1.0
    assert bandit.reg == 0.000625
    assert bandit.mut is None
    assert bandit.index == 0
    assert bandit.scores == []
    assert bandit.fitness == []
    assert bandit.steps == [0]

    assert bandit.numel == sum(
        w.numel() for w in bandit.exp_layer.parameters() if w.requires_grad
    )
    assert torch.equal(
        bandit.theta_0,
        torch.cat(
            [w.flatten() for w in bandit.exp_layer.parameters() if w.requires_grad]
        ),
    )


def test_save_load_checkpoint_correct_data_and_format_cnn(tmpdir):
    net_config_cnn = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }

    # Initialize the NeuralUCB agent
    bandit = NeuralUCB(
        observation_space=spaces.Box(0, 1, shape=(3, 32, 32)),
        action_space=spaces.Discrete(2),
        net_config=net_config_cnn,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    bandit.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    bandit = NeuralUCB(
        observation_space=spaces.Box(0, 1, shape=(4,)), action_space=spaces.Discrete(2)
    )
    # Load checkpoint
    bandit.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(bandit.actor.encoder, EvolvableCNN)
    assert bandit.lr == 1e-3
    assert bandit.batch_size == 64
    assert bandit.learn_step == 2
    assert bandit.gamma == 1.0
    assert bandit.lamb == 1.0
    assert bandit.reg == 0.000625
    assert bandit.mut is None
    assert bandit.index == 0
    assert bandit.scores == []
    assert bandit.fitness == []
    assert bandit.steps == [0]

    assert bandit.numel == sum(
        w.numel() for w in bandit.exp_layer.parameters() if w.requires_grad
    )
    assert torch.equal(
        bandit.theta_0,
        torch.cat(
            [w.flatten() for w in bandit.exp_layer.parameters() if w.requires_grad]
        ),
    )


# The saved checkpoint file contains the correct data and format.
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (spaces.Box(0, 1, shape=(4,)), "simple_mlp", torch.randn(1, 4)),
        (spaces.Box(0, 1, shape=(3, 64, 64)), "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
def test_save_load_checkpoint_correct_data_and_format_cnn_network(
    observation_space, actor_network, input_tensor, request, tmpdir
):
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    # Initialize the NeuralUCB agent
    bandit = NeuralUCB(
        observation_space=observation_space,
        action_space=spaces.Discrete(2),
        actor_network=actor_network,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    bandit.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "learn_step" in checkpoint
    assert "gamma" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint

    bandit = NeuralUCB(
        observation_space=observation_space, action_space=spaces.Discrete(2)
    )
    # Load checkpoint
    bandit.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(bandit.actor, nn.Module)
    assert bandit.lr == 1e-3
    assert bandit.batch_size == 64
    assert bandit.learn_step == 2
    assert bandit.gamma == 1.0
    assert bandit.lamb == 1.0
    assert bandit.reg == 0.000625
    assert bandit.mut is None
    assert bandit.index == 0
    assert bandit.scores == []
    assert bandit.fitness == []
    assert bandit.steps == [0]

    assert bandit.numel == sum(
        w.numel() for w in bandit.exp_layer.parameters() if w.requires_grad
    )
    assert torch.equal(
        bandit.theta_0,
        torch.cat(
            [w.flatten() for w in bandit.exp_layer.parameters() if w.requires_grad]
        ),
    )


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained(device, accelerator, tmpdir):
    # Initialize the NeuralUCB agent
    bandit = NeuralUCB(
        observation_space=spaces.Box(0, 1, shape=(4,)), action_space=spaces.Discrete(2)
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    bandit.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_bandit = NeuralUCB.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_bandit.observation_space == bandit.observation_space
    assert new_bandit.action_space == bandit.action_space
    assert isinstance(new_bandit.actor.encoder, EvolvableMLP)
    assert new_bandit.lr == bandit.lr
    assert str(copy.deepcopy(new_bandit.actor).to("cpu").state_dict()) == str(
        bandit.actor.state_dict()
    )
    assert new_bandit.batch_size == bandit.batch_size
    assert new_bandit.learn_step == bandit.learn_step
    assert new_bandit.gamma == bandit.gamma
    assert new_bandit.lamb == bandit.lamb
    assert new_bandit.reg == bandit.reg
    assert new_bandit.mut == bandit.mut
    assert new_bandit.index == bandit.index
    assert new_bandit.scores == bandit.scores
    assert new_bandit.fitness == bandit.fitness
    assert new_bandit.steps == bandit.steps

    assert new_bandit.numel == sum(
        w.numel() for w in bandit.exp_layer.parameters() if w.requires_grad
    )
    assert torch.equal(
        new_bandit.theta_0,
        torch.cat(
            [w.flatten() for w in new_bandit.exp_layer.parameters() if w.requires_grad]
        ),
    )


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_cnn(device, accelerator, tmpdir):
    # Initialize the NeuralUCB agent
    bandit = NeuralUCB(
        observation_space=spaces.Box(0, 1, shape=(3, 32, 32)),
        action_space=spaces.Discrete(2),
        net_config={
            "encoder_config": {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            }
        },
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    bandit.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_bandit = NeuralUCB.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_bandit.observation_space == bandit.observation_space
    assert new_bandit.action_space == bandit.action_space
    assert isinstance(new_bandit.actor.encoder, EvolvableCNN)
    assert new_bandit.lr == bandit.lr
    assert str(copy.deepcopy(new_bandit.actor).to("cpu").state_dict()) == str(
        bandit.actor.state_dict()
    )
    assert new_bandit.batch_size == bandit.batch_size
    assert new_bandit.learn_step == bandit.learn_step
    assert new_bandit.gamma == bandit.gamma
    assert new_bandit.lamb == bandit.lamb
    assert new_bandit.reg == bandit.reg
    assert new_bandit.mut == bandit.mut
    assert new_bandit.index == bandit.index
    assert new_bandit.scores == bandit.scores
    assert new_bandit.fitness == bandit.fitness
    assert new_bandit.steps == bandit.steps

    assert new_bandit.numel == sum(
        w.numel() for w in bandit.exp_layer.parameters() if w.requires_grad
    )
    assert torch.equal(
        new_bandit.theta_0,
        torch.cat(
            [w.flatten() for w in new_bandit.exp_layer.parameters() if w.requires_grad]
        ),
    )


@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (spaces.Box(0, 1, shape=(4,)), "simple_mlp", torch.randn(1, 4)),
        (spaces.Box(0, 1, shape=(3, 64, 64)), "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_networks(
    observation_space, actor_network, input_tensor, request, tmpdir
):
    action_space = spaces.Discrete(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    # Initialize the NeuralUCB agent
    bandit = NeuralUCB(
        observation_space=observation_space,
        action_space=action_space,
        actor_network=actor_network,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    bandit.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_bandit = NeuralUCB.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_bandit.observation_space == bandit.observation_space
    assert new_bandit.action_space == bandit.action_space
    assert isinstance(new_bandit.actor, nn.Module)
    assert new_bandit.lr == bandit.lr
    assert str(new_bandit.actor.to("cpu").state_dict()) == str(
        bandit.actor.state_dict()
    )
    assert new_bandit.batch_size == bandit.batch_size
    assert new_bandit.learn_step == bandit.learn_step
    assert new_bandit.gamma == bandit.gamma
    assert bandit.lamb == 1.0
    assert bandit.reg == 0.000625
    assert new_bandit.mut == bandit.mut
    assert new_bandit.index == bandit.index
    assert new_bandit.scores == bandit.scores
    assert new_bandit.fitness == bandit.fitness
    assert new_bandit.steps == bandit.steps

    assert new_bandit.numel == sum(
        w.numel() for w in bandit.exp_layer.parameters() if w.requires_grad
    )
    assert torch.equal(
        new_bandit.theta_0,
        torch.cat(
            [w.flatten() for w in new_bandit.exp_layer.parameters() if w.requires_grad]
        ),
    )
