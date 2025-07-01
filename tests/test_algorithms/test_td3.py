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
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.networks.actors import DeterministicActor
from agilerl.networks.q_networks import ContinuousQNetwork
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import (
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multidiscrete_space,
    generate_random_box_space,
    get_experiences_batch,
    get_sample_from_space,
)


@pytest.fixture(autouse=True)
def cleanup():
    yield  # Run the test first
    torch.cuda.empty_cache()  # Free up GPU memory


class DummyTD3(TD3):
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
# Initializes all necessary attributes with default values
@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        (generate_random_box_space(shape=(4,)), EvolvableMLP),
        (generate_random_box_space(shape=(3, 32, 32), low=0, high=255), EvolvableCNN),
        (generate_dict_or_tuple_space(2, 2, dict_space=True), EvolvableMultiInput),
        (generate_dict_or_tuple_space(2, 2, dict_space=False), EvolvableMultiInput),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_initialize_td3(observation_space, encoder_cls, accelerator):
    action_space = generate_random_box_space(shape=(2,), low=-1, high=1)

    # Initialize TD3 with default parameters
    td3 = TD3(observation_space, action_space, accelerator=accelerator)

    expected_opt_cls = AcceleratedOptimizer if accelerator else optim.Adam
    expected_device = accelerator.device if accelerator else "cpu"
    assert td3.observation_space == observation_space
    assert td3.action_space == action_space
    assert np.all(td3.max_action == 1)
    assert td3.batch_size == 64
    assert td3.lr_actor == 0.0001
    assert td3.lr_critic == 0.001
    assert td3.learn_step == 5
    assert td3.gamma == 0.99
    assert td3.tau == 0.005
    assert td3.mut is None
    assert td3.device == expected_device
    assert td3.accelerator == accelerator
    assert td3.index == 0
    assert td3.scores == []
    assert td3.fitness == []
    assert td3.steps == [0]
    assert isinstance(td3.actor.encoder, encoder_cls)
    assert isinstance(td3.actor_target.encoder, encoder_cls)
    assert isinstance(td3.actor_optimizer.optimizer, expected_opt_cls)
    assert isinstance(td3.critic_1.encoder, encoder_cls)
    assert isinstance(td3.critic_target_1.encoder, encoder_cls)
    assert isinstance(td3.critic_1_optimizer.optimizer, expected_opt_cls)
    assert isinstance(td3.critic_2.encoder, encoder_cls)
    assert isinstance(td3.critic_target_2.encoder, encoder_cls)
    assert isinstance(td3.critic_2_optimizer.optimizer, expected_opt_cls)
    assert isinstance(td3.criterion, nn.MSELoss)


# Can initialize td3 with an actor network
# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "observation_space, actor_network, critic_1_network, critic_2_network, input_tensor, input_tensor_critic",
    [
        (
            generate_random_box_space(shape=(4,), low=0, high=1),
            "simple_mlp",
            "simple_mlp_critic",
            "simple_mlp_critic",
            torch.randn(1, 4),
            torch.randn(1, 6),
        ),
    ],
)
def test_initialize_td3_with_actor_network(
    observation_space,
    actor_network,
    critic_1_network,
    critic_2_network,
    input_tensor,
    input_tensor_critic,
    request,
):
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    max_action = 1
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_1_network = request.getfixturevalue(critic_1_network)
    critic_1_network = MakeEvolvable(critic_1_network, input_tensor_critic)
    critic_2_network = request.getfixturevalue(critic_2_network)
    critic_2_network = MakeEvolvable(critic_2_network, input_tensor_critic)

    td3 = TD3(
        observation_space,
        action_space,
        max_action,
        expl_noise=np.zeros((1, action_space.shape[0])),
        actor_network=actor_network,
        critic_networks=[critic_1_network, critic_2_network],
    )

    assert td3.observation_space == observation_space
    assert td3.action_space == action_space
    assert np.all(td3.max_action == max_action)
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
    assert isinstance(td3.actor_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.critic_1_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.critic_2_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.criterion, nn.MSELoss)


# Can initialize td3 with an actor network
# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "observation_space, actor_network, critic_1_network, critic_2_network, input_tensor, input_tensor_critic",
    [
        (
            generate_random_box_space(shape=(4,), low=0, high=1),
            "simple_mlp",
            "simple_mlp_critic",
            "simple_mlp_critic",
            torch.randn(1, 4),
            torch.randn(1, 6),
        ),
    ],
)

# Can initialize td3 with an actor network but no critics - should trigger warning
# TODO: This will be deprecated in the future
def test_initialize_td3_with_actor_network_no_critics(
    observation_space,
    actor_network,
    critic_1_network,
    critic_2_network,
    input_tensor,
    input_tensor_critic,
    request,
):
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    max_action = 1
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    td3 = TD3(
        observation_space,
        action_space,
        max_action,
        actor_network=actor_network,
        critic_networks=None,
    )

    assert td3.observation_space == observation_space
    assert td3.action_space == action_space
    assert np.all(td3.max_action == max_action)
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
    assert isinstance(td3.actor_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.critic_1_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.critic_2_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.criterion, nn.MSELoss)


# Can initialize td3 with an actor network
# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (
            generate_random_box_space(shape=(3, 64, 64), low=0, high=255),
            "simple_cnn",
            torch.randn(1, 3, 64, 64),
        ),
    ],
)
def test_initialize_td3_with_actor_network_cnn(
    observation_space, actor_network, input_tensor, request
):
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    max_action = 1
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_1_network = SimpleCNN()
    critic_1_network = MakeEvolvable(
        critic_1_network,
        input_tensor,
        torch.randn(1, action_space.shape[0]),
    )
    critic_2_network = SimpleCNN()
    critic_2_network = MakeEvolvable(
        critic_2_network,
        input_tensor,
        torch.randn(1, action_space.shape[0]),
    )

    td3 = TD3(
        observation_space,
        action_space,
        max_action,
        actor_network=actor_network,
        critic_networks=[critic_1_network, critic_2_network],
    )

    assert td3.observation_space == observation_space
    assert td3.action_space == action_space
    assert np.all(td3.max_action == max_action)
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
    assert isinstance(td3.actor_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.critic_1_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.critic_2_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_space",
    [
        generate_discrete_space(4),
        generate_random_box_space(shape=(4,)),
        generate_multidiscrete_space(2, 2),
        generate_dict_or_tuple_space(2, 2, dict_space=True),
        generate_dict_or_tuple_space(2, 2, dict_space=False),
    ],
)
def test_returns_expected_action_training(observation_space):
    accelerator = Accelerator()
    action_space = generate_random_box_space(shape=(2,), low=-1, high=1)

    td3 = TD3(observation_space, action_space)
    state = get_sample_from_space(observation_space)
    training = False
    action = td3.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1

    td3 = TD3(
        observation_space,
        action_space,
        accelerator=accelerator,
    )
    state = get_sample_from_space(observation_space)
    training = True
    action = td3.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1

    td3 = TD3(
        observation_space,
        action_space,
        O_U_noise=False,
        accelerator=accelerator,
    )
    state = get_sample_from_space(observation_space)
    training = True
    action = td3.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1


# Returns the expected action from float64 input
def test_returns_expected_action_float64():
    observation_space = generate_discrete_space(4)
    action_space = generate_random_box_space(shape=(2,), low=-1, high=1)

    td3 = TD3(observation_space, action_space)
    state = np.array([0, 1, 2, 3]).astype(np.float64)
    training = False
    action = td3.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1

    td3 = TD3(
        observation_space,
        action_space,
    )
    state = np.array([1]).astype(np.float64)
    training = True
    action = td3.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1

    td3 = TD3(
        observation_space,
        action_space,
        O_U_noise=False,
    )
    state = np.array([1]).astype(np.float64)
    training = True
    action = td3.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1


# learns from experiences and updates network parameters
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_discrete_space(4),
        generate_random_box_space(shape=(4,)),
        generate_random_box_space(shape=(3, 32, 32), low=0, high=255),
        generate_multidiscrete_space(2, 2),
        generate_dict_or_tuple_space(2, 2, dict_space=True),
        generate_dict_or_tuple_space(2, 2, dict_space=False),
    ],
)
@pytest.mark.parametrize(
    "min_action, max_action",
    [(-1, 1), ([-1, 0], [1, 1]), ([-1, -1], [0, 1]), ([-1, -2], [1, 0])],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_learns_from_experiences(
    observation_space, min_action, max_action, accelerator
):
    # Continuous action space
    low = np.array(min_action) if isinstance(min_action, list) else min_action
    high = np.array(max_action) if isinstance(max_action, list) else max_action
    action_space = generate_random_box_space(shape=(2,), low=low, high=high)

    # Create an instance of the td3 class
    batch_size = 64
    policy_freq = 2
    td3 = TD3(
        observation_space,
        action_space,
        batch_size=batch_size,
        policy_freq=policy_freq,
        accelerator=accelerator,
    )

    # Create a batch of experiences
    device = accelerator.device if accelerator else "cpu"
    experiences = get_experiences_batch(
        observation_space, action_space, batch_size, device
    )

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


# Updates target network parameters with soft update
def test_soft_update():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    max_action = 1
    net_config = {"encoder_config": {"hidden_size": [64, 64]}}
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
        observation_space,
        action_space,
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
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_random_box_space(shape=(4,)),
        generate_random_box_space(shape=(3, 32, 32), low=0, high=255),
    ],
)
@pytest.mark.parametrize("num_envs", [1, 3])
def test_algorithm_test_loop(observation_space, num_envs):
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    vect = num_envs > 1
    env = DummyEnv(state_size=observation_space.shape, vect=vect, num_envs=num_envs)
    agent = TD3(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_random_box_space(shape=(4,), low=0, high=1),
        generate_random_box_space(shape=(3, 32, 32), low=0, high=255),
        generate_dict_or_tuple_space(2, 2, dict_space=True),
        generate_dict_or_tuple_space(2, 2, dict_space=False),
    ],
)
def test_clone_returns_identical_agent(observation_space):
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)

    td3 = DummyTD3(observation_space, action_space)
    td3.fitness = [200, 200, 200]
    td3.scores = [94, 94, 94]
    td3.steps = [2500]
    td3.tensor_attribute = torch.randn(1)
    clone_agent = td3.clone()

    assert clone_agent.observation_space == td3.observation_space
    assert clone_agent.action_space == td3.action_space
    assert np.all(clone_agent.max_action == td3.max_action)
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
    td3 = TD3(observation_space, action_space, accelerator=accelerator)
    clone_agent = td3.clone()

    assert clone_agent.observation_space == td3.observation_space
    assert clone_agent.action_space == td3.action_space
    assert np.all(clone_agent.max_action == td3.max_action)
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
    td3 = TD3(observation_space, action_space, accelerator=accelerator, wrap=False)
    clone_agent = td3.clone(wrap=False)

    assert clone_agent.observation_space == td3.observation_space
    assert clone_agent.action_space == td3.action_space
    assert np.all(clone_agent.max_action == td3.max_action)
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
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)

    td3 = TD3(observation_space, action_space)
    clone_agent = td3.clone(index=100)

    assert clone_agent.index == 100


def test_clone_after_learning():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    batch_size = 8
    td3 = TD3(observation_space, action_space)

    # Get experiences and learn
    experiences = get_experiences_batch(
        observation_space, action_space, batch_size, device=td3.device
    )
    td3.learn(experiences)

    # Clone the agent
    clone_agent = td3.clone()

    assert clone_agent.observation_space == td3.observation_space
    assert clone_agent.action_space == td3.action_space
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
        observation_space=generate_random_box_space(shape=(4,), low=0, high=1),
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
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
@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        (generate_random_box_space(shape=(4,), low=0, high=1), EvolvableMLP),
        (generate_random_box_space(shape=(3, 32, 32), low=0, high=255), EvolvableCNN),
        (generate_dict_or_tuple_space(2, 2, dict_space=True), EvolvableMultiInput),
        (generate_dict_or_tuple_space(2, 2, dict_space=False), EvolvableMultiInput),
    ],
)
def test_save_load_checkpoint_correct_data_and_format(
    tmpdir, observation_space, encoder_cls
):
    # Initialize the td3 agent
    td3 = TD3(
        observation_space=observation_space,
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    td3.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill, weights_only=False)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critic_1_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_1_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_1_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_1_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_1_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critic_2_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_2_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_2_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_2_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_2_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "max_action" in checkpoint
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

    td3 = TD3(
        observation_space=observation_space,
        action_space=generate_random_box_space(shape=(2,), low=-1, high=1),
    )
    # Load checkpoint
    td3.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(td3.actor.encoder, encoder_cls)
    assert isinstance(td3.actor_target.encoder, encoder_cls)
    assert isinstance(td3.critic_1.encoder, encoder_cls)
    assert isinstance(td3.critic_target_1.encoder, encoder_cls)
    assert isinstance(td3.critic_2.encoder, encoder_cls)
    assert isinstance(td3.critic_target_2.encoder, encoder_cls)
    assert td3.lr_actor == 1e-4
    assert td3.lr_critic == 1e-3
    assert str(td3.actor.state_dict()) == str(td3.actor_target.state_dict())
    assert str(td3.critic_1.state_dict()) == str(td3.critic_target_1.state_dict())
    assert str(td3.critic_2.state_dict()) == str(td3.critic_target_2.state_dict())
    assert np.all(td3.max_action == 1)
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
# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "actor_network, input_tensor",
    [
        ("simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
def test_save_load_checkpoint_correct_data_and_format_cnn_network(
    actor_network, input_tensor, request, tmpdir
):
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    observation_space = generate_random_box_space(
        shape=input_tensor.shape[1:], low=0, high=1
    )

    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_1_network = SimpleCNN()
    critic_1_network = MakeEvolvable(
        critic_1_network,
        input_tensor,
        torch.randn(1, action_space.shape[0]),
    )
    critic_2_network = SimpleCNN()
    critic_2_network = MakeEvolvable(
        critic_2_network,
        input_tensor,
        torch.randn(1, action_space.shape[0]),
    )

    td3 = TD3(
        observation_space,
        action_space,
        actor_network=actor_network,
        critic_networks=[critic_1_network, critic_2_network],
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    td3.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill, weights_only=False)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critic_1_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_1_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_1_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_1_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_1_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critic_2_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_2_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_2_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_2_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_2_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "max_action" in checkpoint
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

    td3 = TD3(
        observation_space=generate_random_box_space(shape=(4,), low=0, high=1),
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
        actor_network=actor_network,
        critic_networks=[critic_1_network, critic_2_network],
    )
    # Load checkpoint
    td3.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
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
    assert np.all(td3.max_action == 1)
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
    "observation_space, net_type",
    [
        (generate_random_box_space(shape=(4,), low=0, high=1), "mlp"),
        (generate_random_box_space(shape=(3, 64, 64), low=0, high=255), "cnn"),
    ],
)
def test_initialize_td3_with_actor_network_evo_net(observation_space, net_type):
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    max_action = 1

    actor_network = DeterministicActor(observation_space, action_space)
    critic_networks = [
        ContinuousQNetwork(observation_space, action_space) for _ in range(2)
    ]

    td3 = TD3(
        observation_space,
        action_space,
        actor_network=actor_network,
        critic_networks=critic_networks,
    )

    assert td3.observation_space == observation_space
    assert td3.action_space == action_space
    assert np.all(td3.max_action == max_action)
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
    assert isinstance(td3.actor_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.critic_1_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.critic_2_optimizer.optimizer, optim.Adam)
    assert isinstance(td3.criterion, nn.MSELoss)


def test_initialize_td3_with_incorrect_actor_net():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    actor_network = "dummy"
    critic_networks = "dummy"
    with pytest.raises(AssertionError):
        td3 = TD3(
            observation_space,
            action_space,
            actor_network=actor_network,
            critic_networks=critic_networks,
        )
        assert td3


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
    td3 = TD3(
        observation_space=generate_random_box_space(shape=(4,), low=0, high=1),
        action_space=generate_random_box_space(shape=(1,), low=0, high=1),
        device=device,
    )
    input = torch.tensor(action, dtype=torch.float32).to(device)
    clamped_actions = td3.multi_dim_clamp(min, max, input).type(torch.float32)
    expected_result = torch.tensor(expected_result)
    assert clamped_actions.dtype == expected_result.dtype


# The saved checkpoint file contains the correct data and format.
@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        (generate_random_box_space(shape=(4,), low=0, high=1), EvolvableMLP),
        (generate_random_box_space(shape=(3, 32, 32), low=0, high=255), EvolvableCNN),
        (generate_dict_or_tuple_space(2, 2, dict_space=True), EvolvableMultiInput),
        (generate_dict_or_tuple_space(2, 2, dict_space=False), EvolvableMultiInput),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_load_from_pretrained(observation_space, encoder_cls, accelerator, tmpdir):
    # Initialize the td3 agent
    td3 = TD3(
        observation_space=observation_space,
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    td3.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_td3 = TD3.load(checkpoint_path, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_td3.observation_space == td3.observation_space
    assert new_td3.action_space == td3.action_space
    assert np.all(new_td3.min_action == td3.min_action)
    assert np.all(new_td3.max_action == td3.max_action)
    assert isinstance(new_td3.actor.encoder, encoder_cls)
    assert isinstance(new_td3.actor_target.encoder, encoder_cls)
    assert isinstance(new_td3.critic_1.encoder, encoder_cls)
    assert isinstance(new_td3.critic_target_1.encoder, encoder_cls)
    assert isinstance(new_td3.critic_2.encoder, encoder_cls)
    assert isinstance(new_td3.critic_target_2.encoder, encoder_cls)
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


# The saved checkpoint file contains the correct data and format.
# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (
            generate_random_box_space(shape=(4,), low=0, high=255),
            "simple_mlp",
            torch.randn(1, 4),
        ),
        (
            generate_random_box_space(shape=(3, 64, 64), low=0, high=255),
            "simple_cnn",
            torch.randn(1, 3, 64, 64),
        ),
    ],
)
def test_load_from_pretrained_networks(
    observation_space, actor_network, input_tensor, request, tmpdir
):
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    # Initialize the td3 agent
    td3 = TD3(
        observation_space=observation_space,
        action_space=action_space,
        actor_network=actor_network,
        critic_networks=[copy.deepcopy(actor_network), copy.deepcopy(actor_network)],
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    td3.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_td3 = TD3.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_td3.observation_space == td3.observation_space
    assert new_td3.action_space == td3.action_space
    assert np.all(new_td3.min_action == td3.min_action)
    assert np.all(new_td3.max_action == td3.max_action)
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
