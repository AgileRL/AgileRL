import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces

from agilerl.algorithms.td3 import TD3
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.networks.actors import DeterministicActor
from agilerl.networks.q_networks import ContinuousQNetwork
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import (
    assert_not_equal_state_dict,
    assert_state_dicts_equal,
    get_experiences_batch,
    get_sample_from_space,
)


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
            32 * 8 * 8, 128
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
        ("vector_space", EvolvableMLP),
        ("image_space", EvolvableCNN),
        ("dict_space", EvolvableMultiInput),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_initialize_td3(
    observation_space, vector_space, encoder_cls, accelerator, request
):
    observation_space = request.getfixturevalue(observation_space)

    # Initialize TD3 with default parameters
    td3 = TD3(observation_space, vector_space, accelerator=accelerator)

    expected_opt_cls = AcceleratedOptimizer if accelerator else optim.Adam
    expected_device = accelerator.device if accelerator else "cpu"
    assert td3.observation_space == observation_space
    assert td3.action_space == vector_space
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
            "vector_space",
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
    vector_space,
    actor_network,
    critic_1_network,
    critic_2_network,
    input_tensor,
    input_tensor_critic,
    request,
):
    observation_space = request.getfixturevalue(observation_space)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_1_network = request.getfixturevalue(critic_1_network)
    critic_1_network = MakeEvolvable(critic_1_network, input_tensor_critic)
    critic_2_network = request.getfixturevalue(critic_2_network)
    critic_2_network = MakeEvolvable(critic_2_network, input_tensor_critic)

    td3 = TD3(
        observation_space,
        vector_space,
        expl_noise=np.zeros((1, vector_space.shape[0])),
        actor_network=actor_network,
        critic_networks=[critic_1_network, critic_2_network],
    )

    assert td3.observation_space == observation_space
    assert td3.action_space == vector_space
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
            "vector_space",
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
    vector_space,
    actor_network,
    critic_1_network,
    critic_2_network,
    input_tensor,
    input_tensor_critic,
    request,
):
    observation_space = request.getfixturevalue(observation_space)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    td3 = TD3(
        observation_space,
        vector_space,
        actor_network=actor_network,
        critic_networks=None,
    )

    assert td3.observation_space == observation_space
    assert td3.action_space == vector_space
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
            "image_space",
            "simple_cnn",
            torch.randn(1, 3, 32, 32),
        ),
    ],
)
def test_initialize_td3_with_actor_network_cnn(
    observation_space, vector_space, actor_network, input_tensor, request
):
    observation_space = request.getfixturevalue(observation_space)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_1_network = SimpleCNN()
    critic_1_network = MakeEvolvable(
        critic_1_network,
        input_tensor,
        torch.randn(1, vector_space.shape[0]),
    )
    critic_2_network = SimpleCNN()
    critic_2_network = MakeEvolvable(
        critic_2_network,
        input_tensor,
        torch.randn(1, vector_space.shape[0]),
    )

    td3 = TD3(
        observation_space,
        vector_space,
        actor_network=actor_network,
        critic_networks=[critic_1_network, critic_2_network],
    )

    assert td3.observation_space == observation_space
    assert td3.action_space == vector_space
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
    "observation_space", ["vector_space", "image_space", "dict_space"]
)
def test_returns_expected_action_training(observation_space, request):
    observation_space = request.getfixturevalue(observation_space)
    action_space = spaces.Box(low=-1, high=1, shape=(2,))
    accelerator = Accelerator()

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
def test_returns_expected_action_float64(discrete_space):
    action_space = spaces.Box(low=-1, high=1, shape=(2,))
    td3 = TD3(discrete_space, action_space)
    state = np.array([0, 1, 0, 1]).astype(np.float64)
    training = False
    action = td3.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1

    td3 = TD3(discrete_space, action_space)
    state = np.array([1]).astype(np.float64)
    training = True
    action = td3.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1

    td3 = TD3(discrete_space, action_space, O_U_noise=False)
    state = np.array([1]).astype(np.float64)
    training = True
    action = td3.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1


# learns from experiences and updates network parameters
@pytest.mark.parametrize(
    "observation_space", ["vector_space", "image_space", "dict_space"]
)
@pytest.mark.parametrize(
    "min_action, max_action",
    [(-1, 1), ([-1, 0], [1, 1]), ([-1, -1], [0, 1]), ([-1, -2], [1, 0])],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_learns_from_experiences(
    observation_space, min_action, max_action, accelerator, request
):
    observation_space = request.getfixturevalue(observation_space)
    # Continuous action space
    low = np.array(min_action) if isinstance(min_action, list) else min_action
    high = np.array(max_action) if isinstance(max_action, list) else max_action
    action_space = spaces.Box(low=low, high=high, shape=(2,))

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
    actor_pre_learn_sd = copy.deepcopy(td3.actor.state_dict())
    critic_1 = td3.critic_1
    critic_target_1 = td3.critic_target_1
    critic_1_pre_learn_sd = copy.deepcopy(td3.critic_1.state_dict())
    critic_2 = td3.critic_2
    critic_target_2 = td3.critic_target_2
    critic_2_pre_learn_sd = copy.deepcopy(td3.critic_2.state_dict())

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
    assert_not_equal_state_dict(actor_pre_learn_sd, td3.actor.state_dict())
    assert critic_1 == td3.critic_1
    assert critic_target_1 == td3.critic_target_1
    assert_not_equal_state_dict(critic_1_pre_learn_sd, td3.critic_1.state_dict())
    assert critic_2 == td3.critic_2
    assert critic_target_2 == td3.critic_target_2
    assert_not_equal_state_dict(critic_2_pre_learn_sd, td3.critic_2.state_dict())


# Updates target network parameters with soft update
def test_soft_update(vector_space):
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
        vector_space,
        copy.deepcopy(vector_space),
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
@pytest.mark.parametrize("observation_space", ["vector_space", "image_space"])
@pytest.mark.parametrize("num_envs", [1, 3])
def test_algorithm_test_loop(observation_space, vector_space, num_envs, request):
    observation_space = request.getfixturevalue(observation_space)
    vect = num_envs > 1
    env = DummyEnv(state_size=observation_space.shape, vect=vect, num_envs=num_envs)
    agent = TD3(observation_space, vector_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
@pytest.mark.parametrize("observation_space", ["vector_space"])
def test_clone_returns_identical_agent(observation_space, vector_space, request):
    observation_space = request.getfixturevalue(observation_space)

    td3 = DummyTD3(observation_space, vector_space)
    td3.fitness = [200, 200, 200]
    td3.scores = [94, 94, 94]
    td3.steps = [2500]
    td3.tensor_attribute = torch.randn(1)
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), td3.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), td3.actor_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_1.state_dict(), td3.critic_1.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_target_1.state_dict(), td3.critic_target_1.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_2.state_dict(), td3.critic_2.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_target_2.state_dict(), td3.critic_target_2.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_optimizer.state_dict(), td3.actor_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_1_optimizer.state_dict(), td3.critic_1_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_2_optimizer.state_dict(), td3.critic_2_optimizer.state_dict()
    )
    assert clone_agent.fitness == td3.fitness
    assert clone_agent.steps == td3.steps
    assert clone_agent.scores == td3.scores
    assert clone_agent.tensor_attribute == td3.tensor_attribute
    assert clone_agent.tensor_test == td3.tensor_test

    accelerator = Accelerator()
    td3 = TD3(observation_space, vector_space, accelerator=accelerator)
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), td3.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), td3.actor_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_1.state_dict(), td3.critic_1.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_target_1.state_dict(), td3.critic_target_1.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_2.state_dict(), td3.critic_2.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_target_2.state_dict(), td3.critic_target_2.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_optimizer.state_dict(), td3.actor_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_1_optimizer.state_dict(), td3.critic_1_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_2_optimizer.state_dict(), td3.critic_2_optimizer.state_dict()
    )
    assert clone_agent.fitness == td3.fitness
    assert clone_agent.steps == td3.steps
    assert clone_agent.scores == td3.scores

    accelerator = Accelerator()
    td3 = TD3(observation_space, vector_space, accelerator=accelerator, wrap=False)
    clone_agent = td3.clone(wrap=False)

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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), td3.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), td3.actor_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_1.state_dict(), td3.critic_1.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_target_1.state_dict(), td3.critic_target_1.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_2.state_dict(), td3.critic_2.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_target_2.state_dict(), td3.critic_target_2.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_optimizer.state_dict(), td3.actor_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_1_optimizer.state_dict(), td3.critic_1_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_2_optimizer.state_dict(), td3.critic_2_optimizer.state_dict()
    )
    assert clone_agent.fitness == td3.fitness
    assert clone_agent.steps == td3.steps
    assert clone_agent.scores == td3.scores


def test_clone_new_index(vector_space):
    td3 = TD3(vector_space, copy.deepcopy(vector_space))
    clone_agent = td3.clone(index=100)

    assert clone_agent.index == 100


def test_clone_after_learning(vector_space):
    batch_size = 8
    td3 = TD3(vector_space, copy.deepcopy(vector_space))

    # Get experiences and learn
    experiences = get_experiences_batch(
        vector_space, vector_space, batch_size, device=td3.device
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), td3.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), td3.actor_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_1.state_dict(), td3.critic_1.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_target_1.state_dict(), td3.critic_target_1.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_2.state_dict(), td3.critic_2.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_target_2.state_dict(), td3.critic_target_2.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_optimizer.state_dict(), td3.actor_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_1_optimizer.state_dict(), td3.critic_1_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_2_optimizer.state_dict(), td3.critic_2_optimizer.state_dict()
    )
    assert clone_agent.fitness == td3.fitness
    assert clone_agent.steps == td3.steps
    assert clone_agent.scores == td3.scores


@pytest.mark.parametrize("observation_space", ["vector_space"])
def test_initialize_td3_with_actor_network_evo_net(
    observation_space, vector_space, request
):
    observation_space = request.getfixturevalue(observation_space)

    actor_network = DeterministicActor(observation_space, vector_space)
    critic_networks = [
        ContinuousQNetwork(observation_space, vector_space) for _ in range(2)
    ]

    td3 = TD3(
        observation_space,
        vector_space,
        actor_network=actor_network,
        critic_networks=critic_networks,
    )

    assert td3.observation_space == observation_space
    assert td3.action_space == vector_space
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


def test_initialize_td3_with_incorrect_actor_net(vector_space):
    actor_network = "dummy"
    critic_networks = "dummy"
    with pytest.raises(AssertionError):
        td3 = TD3(
            vector_space,
            copy.deepcopy(vector_space),
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
def test_multi_dim_clamp(vector_space, min, max, action, expected_result, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if isinstance(min, list):
        min = np.array(min)
    if isinstance(max, list):
        max = np.array(max)
    td3 = TD3(
        vector_space,
        copy.deepcopy(vector_space),
        device=device,
    )
    input = torch.tensor(action, dtype=torch.float32).to(device)
    clamped_actions = td3.multi_dim_clamp(min, max, input).type(torch.float32)
    expected_result = torch.tensor(expected_result)
    assert clamped_actions.dtype == expected_result.dtype
