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

from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import generate_discrete_space, generate_random_box_space


class DummyRainbowDQN(RainbowDQN):
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


# initialize DQN with valid parameters
def test_initialize_dqn_with_minimum_parameters():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)

    dqn = RainbowDQN(observation_space, action_space)

    assert dqn.observation_space == observation_space
    assert dqn.action_space == action_space
    assert dqn.net_config == {
        "arch": "mlp",
        "hidden_size": [64, 64],
        "mlp_output_activation": "ReLU",
        "min_hidden_layers": 2,
    }, dqn.net_config
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
    # assert dqn.actor_network is None
    assert isinstance(dqn.actor, EvolvableMLP)
    assert isinstance(dqn.actor_target, EvolvableMLP)
    assert isinstance(dqn.optimizer.optimizer, optim.Adam)
    assert dqn.arch == "mlp"


@pytest.mark.parametrize(
    "observation_space, net_type",
    [
        (generate_random_box_space(shape=(4,)), "mlp"),
        (generate_random_box_space(shape=(3, 64, 64), low=0, high=255), "cnn"),
    ],
)
def test_initialize_dqn_with_actor_network_evo_net(observation_space, net_type):
    action_space = generate_discrete_space(2)
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

    dqn = RainbowDQN(observation_space, action_space, actor_network=actor_network)

    assert dqn.observation_space == observation_space
    assert dqn.action_space == action_space
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
    # assert dqn.actor_network is None
    # assert dqn.actor == actor_network
    assert isinstance(dqn.optimizer.optimizer, optim.Adam)
    assert dqn.arch == actor_network.arch


def test_initialize_dqn_with_incorrect_actor_net_type():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)
    actor_network = "dummy"

    with pytest.raises(AssertionError) as a:
        dqn = RainbowDQN(observation_space, action_space, actor_network=actor_network)
        assert dqn
        assert (
            str(a.value)
            == f"'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableMLP, EvolvableCNN or MakeEvolvable"
        )


# Initializes actor network with EvolvableCNN based on net_config and Accelerator.
def test_initialize_dqn_with_cnn_accelerator():
    observation_space = spaces.Box(0, 1, shape=(3, 32, 32))
    action_space = generate_discrete_space(2)
    index = 0
    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8, 8],
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
    actor_network = None
    accelerator = Accelerator()
    wrap = True

    dqn = RainbowDQN(
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
        actor_network=actor_network,
        accelerator=accelerator,
        wrap=wrap,
    )

    assert dqn.observation_space == observation_space
    assert dqn.action_space == action_space
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
    # assert dqn.actor_network is None
    assert isinstance(dqn.actor, EvolvableCNN)
    assert isinstance(dqn.actor_target, EvolvableCNN)
    assert dqn.arch == "cnn"
    assert isinstance(dqn.optimizer.optimizer, AcceleratedOptimizer)


# Can initialize DQN with an actor network
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (generate_random_box_space(shape=(4,)), "simple_mlp", torch.randn(1, 4)),
        (generate_random_box_space(shape=(3, 64, 64), low=0, high=255), "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
def test_initialize_dqn_with_actor_network(
    observation_space, actor_network, input_tensor, request
):
    action_space = generate_discrete_space(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    dqn = RainbowDQN(observation_space, action_space, actor_network=actor_network)

    assert dqn.observation_space == observation_space
    assert dqn.action_space == action_space
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
    # assert dqn.actor_network == actor_network
    assert isinstance(dqn.optimizer.optimizer, optim.Adam)
    assert dqn.arch == actor_network.arch


@pytest.mark.parametrize(
    "accelerator",
    [
        None,
        Accelerator(),
    ],
)
# Returns the expected action when given a state observation and action mask
def test_returns_expected_action(accelerator):
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)

    dqn = RainbowDQN(observation_space, action_space, accelerator=accelerator)
    state = np.array([1, 2, 3, 4])

    action_mask = None

    action = dqn.get_action(state, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_space.n

    action_mask = np.array([0, 1])

    action = dqn.get_action(state, action_mask)[0]

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
    observation_space = spaces.Discrete(4)
    action_space = generate_discrete_space(2)

    dqn = RainbowDQN(observation_space, action_space, accelerator=accelerator)
    state = np.array([1])

    action_mask = None

    action = dqn.get_action(state, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_space.n

    action_mask = np.array([0, 1])

    action = dqn.get_action(state, action_mask)[0]

    assert action.is_integer()
    assert action == 1


def test_returns_expected_action_mask_vectorized():
    accelerator = Accelerator()
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)

    dqn = RainbowDQN(observation_space, action_space, accelerator=accelerator)
    state = np.array([[1, 2, 4, 5], [2, 3, 5, 1]])

    action_mask = np.array([[0, 1], [1, 0]])

    action = dqn.get_action(state, action_mask)

    assert np.array_equal(action, [1, 0])


@pytest.mark.parametrize(
    "accelerator, net_config, observation_space",
    [
        (
            None,
            {
                "arch": "mlp",
                "hidden_size": [64, 64],
            },
            generate_random_box_space(shape=(4,)),
        ),
        (
            None,
            {
                "arch": "cnn",
                "channel_size": [16, 32, 32],
                "kernel_size": [8, 4, 3],
                "stride_size": [4, 2, 1],
                "hidden_size": [64, 64],
            },
            spaces.Box(0, 1, shape=(4, 84, 84)),
        ),
        (
            Accelerator(),
            {
                "arch": "mlp",
                "hidden_size": [64, 64],
            },
            generate_random_box_space(shape=(4,)),
        ),
    ],
)
# learns from experiences and updates network parameters
def test_learns_from_experiences(accelerator, net_config, observation_space):
    torch.autograd.set_detect_anomaly(True)
    action_space = generate_discrete_space(2)
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        observation_space,
        action_space,
        net_config=net_config,
        batch_size=batch_size,
        accelerator=accelerator,
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, *observation_space.shape)
    actions = torch.randint(0, action_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, *observation_space.shape)
    dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [states, actions, rewards, next_states, dones]

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(dqn.actor.state_dict()))
    actor_target_pre_learn_sd = str(copy.deepcopy(dqn.actor_target.state_dict()))

    # Call the learn method
    loss, new_idxs, new_priorities = dqn.learn(experiences, n_step=False, per=False)

    assert loss > 0.0
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
    observation_space = generate_discrete_space(4)
    action_space = generate_discrete_space(2)
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        observation_space, action_space, batch_size=batch_size, accelerator=accelerator
    )

    # Create a batch of experiences
    states = torch.randint(0, observation_space.n, (batch_size, 1))
    actions = torch.randint(0, action_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randint(0, observation_space.n, (batch_size, 1))
    dones = torch.randint(0, 2, (batch_size, 1))

    experiences = [states, actions, rewards, next_states, dones]

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(dqn.actor.state_dict()))
    actor_target_pre_learn_sd = str(copy.deepcopy(dqn.actor_target.state_dict()))

    # Call the learn method
    loss, new_idxs, new_priorities = dqn.learn(experiences, n_step=False, per=False)

    assert loss > 0.0
    assert new_idxs is None
    assert new_priorities is None
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert actor_pre_learn_sd != str(dqn.actor.state_dict())
    assert actor_target_pre_learn_sd != str(dqn.actor_target.state_dict())


@pytest.mark.parametrize(
    "accelerator, combined",
    [
        (None, True),
        (Accelerator(), True),
        (None, False),
        (Accelerator(), False),
    ],
)
# learns from experiences and updates network parameters
def test_learns_from_experiences_n_step(accelerator, combined):
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        observation_space,
        action_space,
        batch_size=batch_size,
        accelerator=accelerator,
        combined_reward=combined,
    )

    # Create a batch of experiences
    # Create a batch of experiences
    states = torch.randn(batch_size, observation_space.shape[0])
    actions = torch.randint(0, action_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, observation_space.shape[0])
    dones = torch.randint(0, 2, (batch_size, 1))
    idxs = np.arange(batch_size)
    n_states = torch.randn(batch_size, observation_space.shape[0])
    n_actions = torch.randint(0, action_space.n, (batch_size, 1))
    n_rewards = torch.randn((batch_size, 1))
    n_next_states = torch.randn(batch_size, observation_space.shape[0])
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
    loss, new_idxs, new_priorities = dqn.learn(experiences, n_step=True, per=False)

    assert loss > 0.0
    assert new_idxs is not None
    assert new_priorities is None
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert actor_pre_learn_sd != str(dqn.actor.state_dict())
    assert actor_target_pre_learn_sd != str(dqn.actor_target.state_dict())


@pytest.mark.parametrize(
    "accelerator, combined",
    [
        (None, True),
        (Accelerator(), True),
        (None, False),
        (Accelerator(), False),
    ],
)
# learns from experiences and updates network parameters
def test_learns_from_experiences_per(accelerator, combined):
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        observation_space,
        action_space,
        batch_size=batch_size,
        accelerator=accelerator,
        combined_reward=combined,
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, observation_space.shape[0])
    actions = torch.randint(0, action_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, observation_space.shape[0])
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
    loss, new_idxs, new_priorities = dqn.learn(experiences, n_step=False, per=True)

    assert loss > 0.0
    assert isinstance(new_idxs, np.ndarray)
    assert isinstance(new_priorities, np.ndarray)
    assert np.array_equal(new_idxs, idxs)
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert actor_pre_learn_sd != str(dqn.actor.state_dict())
    assert actor_target_pre_learn_sd != str(dqn.actor_target.state_dict())


@pytest.mark.parametrize(
    "accelerator, combined",
    [
        (None, True),
        (Accelerator(), True),
        (None, False),
        (Accelerator(), False),
    ],
)
# learns from experiences and updates network parameters
def test_learns_from_experiences_per_n_step(accelerator, combined):
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        observation_space,
        action_space,
        batch_size=batch_size,
        accelerator=accelerator,
        combined_reward=combined,
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, observation_space.shape[0])
    actions = torch.randint(0, action_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, observation_space.shape[0])
    dones = torch.randint(0, 2, (batch_size, 1))
    weights = torch.rand(batch_size)
    idxs = np.arange(batch_size)
    n_states = torch.randn(batch_size, observation_space.shape[0])
    n_actions = torch.randint(0, action_space.n, (batch_size, 1))
    n_rewards = torch.randn((batch_size, 1))
    n_next_states = torch.randn(batch_size, observation_space.shape[0])
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
    loss, new_idxs, new_priorities = dqn.learn(experiences, n_step=True, per=True)

    assert loss > 0.0
    assert isinstance(new_idxs, np.ndarray)
    assert isinstance(new_priorities, np.ndarray)
    assert np.array_equal(new_idxs, idxs)
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert actor_pre_learn_sd != str(dqn.actor.state_dict())
    assert actor_target_pre_learn_sd != str(dqn.actor_target.state_dict())


# Updates target network parameters with soft update
def test_soft_update():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)
    net_config = {"arch": "mlp", "hidden_size": [64, 64]}
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
        observation_space,
        action_space,
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

    dqn.soft_update()

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
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)
    num_envs = 3

    env = DummyEnv(state_size=observation_space.shape, vect=True, num_envs=num_envs)

    # env = make_vect_envs("CartPole-v1", num_envs=num_envs)
    agent = RainbowDQN(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorised env
def test_algorithm_test_loop_unvectorized():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)

    env = DummyEnv(state_size=observation_space.shape, vect=False)

    agent = RainbowDQN(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with images
def test_algorithm_test_loop_images():
    observation_space = spaces.Box(0, 1, shape=(3, 32, 32))
    action_space = generate_discrete_space(2)

    env = DummyEnv(state_size=observation_space.shape, vect=True)

    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8, 8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
    }

    agent = RainbowDQN(
        observation_space=observation_space,
        action_space=action_space,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorized images
def test_algorithm_test_loop_images_unvectorized():
    observation_space = spaces.Box(0, 1, shape=(32, 32, 3))
    action_space = generate_discrete_space(2)

    env = DummyEnv(state_size=observation_space.shape, vect=False)

    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8, 8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
    }

    agent = RainbowDQN(
        observation_space=spaces.Box(0, 1, shape=(3, 32, 32)),
        action_space=action_space,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10, swap_channels=True)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
def test_clone_returns_identical_agent():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)

    dqn = DummyRainbowDQN(observation_space, action_space)
    dqn.fitness = [200, 200, 200]
    dqn.scores = [94, 94, 94]
    dqn.steps = [2500]
    dqn.tensor_attribute = torch.randn(1)
    clone_agent = dqn.clone()

    assert clone_agent.observation_space == dqn.observation_space
    assert clone_agent.action_space == dqn.action_space
    assert clone_agent.net_config == dqn.net_config
    # assert clone_agent.actor_network == dqn.actor_network
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
    assert clone_agent.tensor_attribute == dqn.tensor_attribute
    assert clone_agent.tensor_test == dqn.tensor_test

    accelerator = Accelerator()
    dqn = RainbowDQN(observation_space, action_space, accelerator=accelerator)
    clone_agent = dqn.clone()

    assert clone_agent.observation_space == dqn.observation_space
    assert clone_agent.action_space == dqn.action_space
    assert clone_agent.net_config == dqn.net_config
    # assert clone_agent.actor_network == dqn.actor_network
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
        observation_space, action_space, accelerator=accelerator, wrap=False
    )
    clone_agent = dqn.clone(wrap=False)

    assert clone_agent.observation_space == dqn.observation_space
    assert clone_agent.action_space == dqn.action_space
    assert clone_agent.net_config == dqn.net_config
    # assert clone_agent.actor_network == dqn.actor_network
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


def test_clone_new_index():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)

    dqn = RainbowDQN(observation_space, action_space)
    clone_agent = dqn.clone(index=100)

    assert clone_agent.index == 100


def test_clone_after_learning():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_discrete_space(2)
    batch_size = 8
    rainbow_dqn = RainbowDQN(observation_space, action_space, batch_size=batch_size)

    states = torch.randn(batch_size, observation_space.shape[0])
    actions = torch.randint(0, 2, (batch_size, 1))
    rewards = torch.rand(batch_size, 1)
    next_states = torch.randn(batch_size, observation_space.shape[0])
    dones = torch.zeros(batch_size, 1)

    experiences = states, actions, rewards, next_states, dones
    rainbow_dqn.learn(experiences)
    clone_agent = rainbow_dqn.clone()

    assert clone_agent.observation_space == rainbow_dqn.observation_space
    assert clone_agent.action_space == rainbow_dqn.action_space
    assert clone_agent.net_config == rainbow_dqn.net_config
    # assert clone_agent.actor_network == rainbow_dqn.actor_network
    assert clone_agent.batch_size == rainbow_dqn.batch_size
    assert clone_agent.lr == rainbow_dqn.lr
    assert clone_agent.learn_step == rainbow_dqn.learn_step
    assert clone_agent.gamma == rainbow_dqn.gamma
    assert clone_agent.tau == rainbow_dqn.tau
    assert clone_agent.mut == rainbow_dqn.mut
    assert clone_agent.device == rainbow_dqn.device
    assert clone_agent.accelerator == rainbow_dqn.accelerator
    assert str(clone_agent.actor.state_dict()) == str(rainbow_dqn.actor.state_dict())
    assert str(clone_agent.actor_target.state_dict()) == str(
        rainbow_dqn.actor_target.state_dict()
    )
    assert str(clone_agent.optimizer.state_dict()) == str(
        rainbow_dqn.optimizer.state_dict()
    )
    assert clone_agent.fitness == rainbow_dqn.fitness
    assert clone_agent.steps == rainbow_dqn.steps
    assert clone_agent.scores == rainbow_dqn.scores


# The method successfully unwraps the actor and actor_target models when an accelerator is present.
def test_unwrap_models():
    dqn = RainbowDQN(
        observation_space=generate_random_box_space(shape=(4,)), action_space=generate_discrete_space(2), accelerator=Accelerator()
    )
    dqn.unwrap_models()
    assert isinstance(dqn.actor, nn.Module)
    assert isinstance(dqn.actor_target, nn.Module)


# The saved checkpoint file contains the correct data and format.
def test_save_load_checkpoint_correct_data_and_format(tmpdir):
    # Initialize the DQN agent
    dqn = RainbowDQN(observation_space=generate_random_box_space(shape=(4,)), action_space=generate_discrete_space(2))
    initial_actor_state_dict = dqn.actor.state_dict()
    init_optim_state_dict = dqn.optimizer.state_dict()

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.save_checkpoint(checkpoint_path)

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
    dqn.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert dqn.net_config == {
        "arch": "mlp",
        "hidden_size": [64, 64],
        "mlp_output_activation": "ReLU",
        "min_hidden_layers": 2,
    }
    assert isinstance(dqn.actor, EvolvableMLP)
    assert isinstance(dqn.actor_target, EvolvableMLP)
    assert dqn.lr == 1e-4
    assert str(dqn.actor.state_dict()) == str(dqn.actor_target.state_dict())
    assert str(initial_actor_state_dict) == str(dqn.actor.state_dict())
    assert str(init_optim_state_dict) == str(dqn.optimizer.state_dict())
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
        "hidden_size": [8, 8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
    }

    # Initialize the DQN agent
    dqn = RainbowDQN(
        observation_space=spaces.Box(0, 1, shape=(3, 32, 32)), action_space=generate_discrete_space(2), net_config=net_config_cnn
    )
    initial_actor_state_dict = dqn.actor.state_dict()
    init_optim_state_dict = dqn.optimizer.state_dict()

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.save_checkpoint(checkpoint_path)

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
    dqn.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert dqn.net_config == net_config_cnn
    assert isinstance(dqn.actor, EvolvableCNN)
    assert isinstance(dqn.actor_target, EvolvableCNN)
    assert dqn.lr == 1e-4
    assert str(dqn.actor.state_dict()) == str(dqn.actor_target.state_dict())
    assert str(initial_actor_state_dict) == str(dqn.actor.state_dict())
    assert str(init_optim_state_dict) == str(dqn.optimizer.state_dict())
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
        observation_space=generate_random_box_space(shape=(3, 64, 64), low=0, high=255), action_space=generate_discrete_space(2), actor_network=actor_network
    )

    initial_actor_state_dict = dqn.actor.state_dict()
    init_optim_state_dict = dqn.optimizer.state_dict()

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.save_checkpoint(checkpoint_path)

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
    dqn.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert dqn.net_config is None
    assert isinstance(dqn.actor, nn.Module)
    assert isinstance(dqn.actor_target, nn.Module)
    assert dqn.lr == 1e-4
    assert str(dqn.actor.state_dict()) == str(dqn.actor_target.state_dict())
    assert str(initial_actor_state_dict) == str(dqn.actor.state_dict())
    assert str(init_optim_state_dict) == str(dqn.optimizer.state_dict())
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
    dqn = RainbowDQN(observation_space=generate_random_box_space(shape=(4,)), action_space=generate_discrete_space(2))

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_dqn = RainbowDQN.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_dqn.observation_space == dqn.observation_space
    assert new_dqn.action_space == dqn.action_space
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
        observation_space=spaces.Box(0, 1, shape=(3, 32, 32)),
        action_space=generate_discrete_space(2),
        net_config={
            "arch": "cnn",
            "hidden_size": [8, 8],
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1]
        },
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_dqn = RainbowDQN.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_dqn.observation_space == dqn.observation_space
    assert new_dqn.action_space == dqn.action_space
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
    "observation_space, actor_network, input_tensor",
    [
        (generate_random_box_space(shape=(4,)), "simple_mlp", torch.randn(1, 4)),
        (generate_random_box_space(shape=(3, 64, 64), low=0, high=255), "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_networks(
    observation_space, actor_network, input_tensor, request, tmpdir
):
    action_space = generate_discrete_space(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    # Initialize the DQN agent
    dqn = RainbowDQN(
        observation_space=observation_space,
        action_space=action_space,
        actor_network=actor_network,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    dqn.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_dqn = RainbowDQN.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_dqn.observation_space == dqn.observation_space
    assert new_dqn.action_space == dqn.action_space
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
