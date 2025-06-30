import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces

from agilerl.algorithms.cqn import CQN
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import assert_not_equal_state_dict, assert_state_dicts_equal


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


@pytest.fixture(scope="module")
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


@pytest.fixture(scope="module")
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
@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        ("vector_space", EvolvableMLP),
        ("image_space", EvolvableCNN),
        ("dict_space", EvolvableMultiInput),
        ("multidiscrete_space", EvolvableMLP),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_initialize_cqn(observation_space, encoder_cls, accelerator, request):
    action_space = spaces.Discrete(2)
    observation_space = request.getfixturevalue(observation_space)
    cqn = CQN(observation_space, action_space, accelerator=accelerator)

    expected_device = accelerator.device if accelerator else "cpu"
    assert cqn.observation_space == observation_space
    assert cqn.action_space == action_space
    assert cqn.batch_size == 64
    assert cqn.lr == 0.0001
    assert cqn.learn_step == 5
    assert cqn.gamma == 0.99
    assert cqn.tau == 0.001
    assert cqn.mut is None
    assert cqn.device == expected_device
    assert cqn.accelerator == accelerator
    assert cqn.index == 0
    assert cqn.scores == []
    assert cqn.fitness == []
    assert cqn.steps == [0]
    assert isinstance(cqn.actor.encoder, encoder_cls)
    assert isinstance(cqn.actor_target.encoder, encoder_cls)
    expected_opt_cls = AcceleratedOptimizer if accelerator else optim.Adam
    assert isinstance(cqn.optimizer.optimizer, expected_opt_cls)
    assert isinstance(cqn.criterion, nn.MSELoss)


# Can initialize cqn with an actor network
# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        ("vector_space", "simple_mlp", torch.randn(1, 4)),
        (
            "image_space",
            "simple_cnn",
            torch.randn(1, 3, 64, 64),
        ),
    ],
)
def test_initialize_cqn_with_make_evo(
    observation_space, actor_network, input_tensor, request
):
    action_space = spaces.Discrete(2)
    observation_space = request.getfixturevalue(observation_space)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    cqn = CQN(observation_space, action_space, actor_network=actor_network)

    assert cqn.observation_space == observation_space
    assert cqn.action_space == action_space
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
    assert isinstance(cqn.optimizer.optimizer, optim.Adam)
    assert isinstance(cqn.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_space, net_type",
    [
        ("vector_space", "mlp"),
        ("image_space", "cnn"),
    ],
)
def test_initialize_cqn_with_actor_network_evo_net(
    observation_space, net_type, request
):
    action_space = spaces.Discrete(2)
    observation_space = request.getfixturevalue(observation_space)
    if net_type == "mlp":
        actor_network = EvolvableMLP(
            num_inputs=observation_space.shape[0],
            num_outputs=action_space.n,
            hidden_size=[64, 64],
            activation="ReLU",
        )
    else:
        actor_network = EvolvableCNN(
            input_shape=observation_space.shape,
            num_outputs=action_space.n,
            channel_size=[8, 8],
            kernel_size=[2, 2],
            stride_size=[1, 1],
            activation="ReLU",
        )

    cqn = CQN(observation_space, action_space, actor_network=actor_network)

    assert cqn.observation_space == observation_space
    assert cqn.action_space == action_space
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
    assert isinstance(cqn.optimizer.optimizer, optim.Adam)
    assert isinstance(cqn.criterion, nn.MSELoss)


def test_init_with_incorrect_actor_net(vector_space):
    action_space = spaces.Discrete(2)
    actor_network = "String"

    with pytest.raises(TypeError) as e:
        cqn = CQN(vector_space, action_space, actor_network=actor_network)
        assert cqn
        assert (
            e
            == f"'actor_network' argument is of type {type(actor_network)}, but must be of type nn.Module."
        )


# Returns the expected action when given a state observation and epsilon=0 or 1.
def test_returns_expected_action_epsilon_greedy(vector_space):
    action_space = spaces.Discrete(2)

    cqn = CQN(vector_space, action_space)
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
def test_returns_expected_action_mask(vector_space):
    accelerator = Accelerator()
    action_space = spaces.Discrete(2)

    cqn = CQN(vector_space, action_space, accelerator=accelerator)
    state = np.array([1, 2, 3, 4])

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
def test_learns_from_experiences(vector_space):
    action_space = spaces.Discrete(2)
    batch_size = 64

    # Create an instance of the cqn class
    cqn = CQN(vector_space, action_space, batch_size=batch_size)

    # Create a batch of experiences
    states = torch.randn(batch_size, vector_space.shape[0])
    actions = torch.randint(0, action_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, vector_space.shape[0])
    dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [states, actions, rewards, next_states, dones]

    # Copy state dict before learning - should be different to after updating weights
    actor = cqn.actor
    actor_target = cqn.actor_target
    actor_pre_learn_sd = copy.deepcopy(cqn.actor.state_dict())
    actor_target_pre_learn_sd = copy.deepcopy(cqn.actor_target.state_dict())

    # Call the learn method
    cqn.learn(experiences)

    assert actor == cqn.actor
    assert actor_target == cqn.actor_target
    assert_not_equal_state_dict(actor_pre_learn_sd, cqn.actor.state_dict())
    assert_not_equal_state_dict(
        actor_target_pre_learn_sd, cqn.actor_target.state_dict()
    )


# handles double Q-learning
def test_handles_double_q_learning(discrete_space):
    accelerator = Accelerator()
    action_space = spaces.Discrete(2)
    double = True
    batch_size = 64

    # Create an instance of the cqn class
    cqn = CQN(
        discrete_space,
        action_space,
        double=double,
        batch_size=batch_size,
        accelerator=accelerator,
    )

    # Create a batch of experiences
    states = torch.randint(0, discrete_space.n, (batch_size, 1))
    actions = torch.randint(0, action_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randint(0, discrete_space.n, (batch_size, 1))
    dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [states, actions, rewards, next_states, dones]

    # Copy state dict before learning - should be different to after updating weights
    actor = cqn.actor
    actor_target = cqn.actor_target
    actor_pre_learn_sd = copy.deepcopy(cqn.actor.state_dict())
    actor_target_pre_learn_sd = copy.deepcopy(cqn.actor_target.state_dict())

    # Call the learn method
    cqn.learn(experiences)

    assert actor == cqn.actor
    assert actor_target == cqn.actor_target
    assert_not_equal_state_dict(actor_pre_learn_sd, cqn.actor.state_dict())
    assert_not_equal_state_dict(
        actor_target_pre_learn_sd, cqn.actor_target.state_dict()
    )


# Updates target network parameters with soft update
def test_soft_update(vector_space):
    action_space = spaces.Discrete(2)
    net_config = {"encoder_config": {"hidden_size": [64, 64]}}
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
        vector_space,
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
@pytest.mark.parametrize("observation_space", ["vector_space", "image_space"])
@pytest.mark.parametrize("num_envs", [1, 3])
def test_algorithm_test_loop(observation_space, num_envs, request):
    observation_space = request.getfixturevalue(observation_space)
    action_space = spaces.Discrete(2)

    vect = num_envs > 1
    env = DummyEnv(state_size=observation_space.shape, vect=vect, num_envs=num_envs)
    agent = CQN(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
@pytest.mark.parametrize("observation_space", ["vector_space"])
def test_clone_returns_identical_agent(observation_space, request):
    action_space = spaces.Discrete(2)
    observation_space = request.getfixturevalue(observation_space)

    cqn = DummyCQN(observation_space, action_space)
    cqn.tensor_attribute = torch.randn(1)
    clone_agent = cqn.clone()

    assert clone_agent.observation_space == cqn.observation_space
    assert clone_agent.action_space == cqn.action_space
    assert clone_agent.batch_size == cqn.batch_size
    assert clone_agent.lr == cqn.lr
    assert clone_agent.learn_step == cqn.learn_step
    assert clone_agent.gamma == cqn.gamma
    assert clone_agent.tau == cqn.tau
    assert clone_agent.mut == cqn.mut
    assert clone_agent.device == cqn.device
    assert clone_agent.accelerator == cqn.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), cqn.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), cqn.actor_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), cqn.optimizer.state_dict()
    )
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
    # assert clone_agent.actor_network == cqn.actor_network
    assert clone_agent.batch_size == cqn.batch_size
    assert clone_agent.lr == cqn.lr
    assert clone_agent.learn_step == cqn.learn_step
    assert clone_agent.gamma == cqn.gamma
    assert clone_agent.tau == cqn.tau
    assert clone_agent.mut == cqn.mut
    assert clone_agent.device == cqn.device
    assert clone_agent.accelerator == cqn.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), cqn.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), cqn.actor_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), cqn.optimizer.state_dict()
    )
    assert clone_agent.fitness == cqn.fitness
    assert clone_agent.steps == cqn.steps
    assert clone_agent.scores == cqn.scores

    accelerator = Accelerator()
    cqn = CQN(observation_space, action_space, accelerator=accelerator, wrap=False)
    clone_agent = cqn.clone(wrap=False)

    assert clone_agent.observation_space == cqn.observation_space
    assert clone_agent.action_space == cqn.action_space
    # assert clone_agent.actor_network == cqn.actor_network
    assert clone_agent.batch_size == cqn.batch_size
    assert clone_agent.lr == cqn.lr
    assert clone_agent.learn_step == cqn.learn_step
    assert clone_agent.gamma == cqn.gamma
    assert clone_agent.tau == cqn.tau
    assert clone_agent.mut == cqn.mut
    assert clone_agent.device == cqn.device
    assert clone_agent.accelerator == cqn.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), cqn.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), cqn.actor_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), cqn.optimizer.state_dict()
    )
    assert clone_agent.fitness == cqn.fitness
    assert clone_agent.steps == cqn.steps
    assert clone_agent.scores == cqn.scores


def test_clone_new_index(vector_space):
    action_space = spaces.Discrete(2)

    cqn = CQN(vector_space, action_space)
    clone_agent = cqn.clone(index=100)

    assert clone_agent.index == 100
