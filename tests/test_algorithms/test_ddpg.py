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

from agilerl.algorithms.ddpg import DDPG
from agilerl.components.data import Transition
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.networks.actors import DeterministicActor
from agilerl.networks.q_networks import ContinuousQNetwork
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import (
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multidiscrete_space,
    generate_random_box_space,
    get_sample_from_space,
)


@pytest.fixture(autouse=True)
def cleanup():
    yield  # Run the test first
    torch.cuda.empty_cache()  # Free up GPU memory


class DummyDDPG(DDPG):
    def __init__(self, observation_space, action_space, one_hot, *args, **kwargs):
        super().__init__(observation_space, action_space, one_hot, *args, **kwargs)

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
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_random_box_space(shape=(2,), low=-1, high=1)

    ddpg = DDPG(observation_space, action_space)

    assert ddpg.observation_space == observation_space
    assert ddpg.action_space == action_space
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
    # assert ddpg.actor_network is None
    assert isinstance(ddpg.actor.encoder, EvolvableMLP)
    assert isinstance(ddpg.actor_target.encoder, EvolvableMLP)
    assert isinstance(ddpg.actor_optimizer.optimizer, optim.Adam)
    assert isinstance(ddpg.critic.encoder, EvolvableMLP)
    assert isinstance(ddpg.critic_target.encoder, EvolvableMLP)
    assert isinstance(ddpg.critic_optimizer.optimizer, optim.Adam)
    assert isinstance(ddpg.criterion, nn.MSELoss)


# Initializes actor network with EvolvableCNN based on net_config and Accelerator.
def test_initialize_ddpg_with_cnn_accelerator():
    observation_space = generate_random_box_space(shape=(3, 32, 32), low=0, high=255)
    action_space = generate_random_box_space(shape=(2,))
    index = 0
    net_config_cnn = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
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
        observation_space=observation_space,
        action_space=action_space,
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

    assert ddpg.observation_space == observation_space
    assert ddpg.action_space == action_space
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
    # assert ddpg.actor_network is None
    assert isinstance(ddpg.actor.encoder, EvolvableCNN)
    assert isinstance(ddpg.actor_target.encoder, EvolvableCNN)
    assert isinstance(ddpg.critic.encoder, EvolvableCNN)
    assert isinstance(ddpg.critic_target.encoder, EvolvableCNN)
    assert isinstance(ddpg.actor_optimizer.optimizer, AcceleratedOptimizer)
    assert isinstance(ddpg.critic_optimizer.optimizer, AcceleratedOptimizer)
    assert isinstance(ddpg.criterion, nn.MSELoss)


# Can initialize ddpg with an actor network
@pytest.mark.parametrize(
    "observation_space, actor_network, critic_network, input_tensor, input_tensor_critic",
    [
        (
            generate_random_box_space(shape=(4,)),
            "simple_mlp",
            "simple_mlp_critic",
            torch.randn(1, 4),
            torch.randn(1, 6),
        ),
    ],
)
def test_initialize_ddpg_with_actor_network(
    observation_space,
    actor_network,
    critic_network,
    input_tensor,
    input_tensor_critic,
    request,
):
    action_space = generate_random_box_space(shape=(2,))
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_network = request.getfixturevalue(critic_network)
    critic_network = MakeEvolvable(critic_network, input_tensor_critic)

    ddpg = DDPG(
        observation_space,
        action_space,
        actor_network=actor_network,
        critic_network=critic_network,
    )

    assert ddpg.observation_space == observation_space
    assert ddpg.action_space == action_space
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
    assert isinstance(ddpg.actor_optimizer.optimizer, optim.Adam)
    assert isinstance(ddpg.critic_optimizer.optimizer, optim.Adam)
    assert isinstance(ddpg.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_space",
    [
        (generate_random_box_space(shape=(4,))),
        (spaces.Box(0, 1, shape=(3, 64, 64))),
    ],
)
def test_initialize_ddpg_with_actor_network_evo_net(observation_space):
    action_space = generate_random_box_space(shape=(2,))

    actor_network = DeterministicActor(observation_space, action_space)
    critic_network = ContinuousQNetwork(observation_space, action_space)

    ddpg = DDPG(
        observation_space,
        action_space,
        actor_network=actor_network,
        critic_network=critic_network,
    )

    assert ddpg.observation_space == observation_space
    assert ddpg.action_space == action_space
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
    # assert ddpg.actor == actor_network
    # assert ddpg.critic == critic_network
    assert isinstance(ddpg.actor_optimizer.optimizer, optim.Adam)
    assert isinstance(ddpg.critic_optimizer.optimizer, optim.Adam)
    assert isinstance(ddpg.criterion, nn.MSELoss)


def test_initialize_ddpg_with_incorrect_actor_net():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_random_box_space(shape=(2,))
    actor_network = "dummy"
    critic_network = "dummy"
    with pytest.raises(TypeError):
        ddpg = DDPG(
            observation_space,
            action_space,
            expl_noise=np.zeros((1, action_space.shape[0])),
            actor_network=actor_network,
            critic_network=critic_network,
        )
        assert ddpg


# Can initialize ddpg with an actor network but no critic - should trigger warning
@pytest.mark.parametrize(
    "observation_space, actor_network, critic_network, input_tensor, input_tensor_critic",
    [
        (
            generate_random_box_space(shape=(4,)),
            "simple_mlp",
            "simple_mlp_critic",
            torch.randn(1, 4),
            torch.randn(1, 6),
        ),
    ],
)
def test_initialize_ddpg_with_actor_network_no_critic(
    observation_space,
    actor_network,
    critic_network,
    input_tensor,
    input_tensor_critic,
    request,
):
    action_space = generate_random_box_space(shape=(2,))
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    ddpg = DDPG(
        observation_space,
        action_space,
        actor_network=actor_network,
        critic_network=None,
    )

    assert ddpg.observation_space == observation_space
    assert ddpg.action_space == action_space
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
    # assert ddpg.critic_network is None
    assert isinstance(ddpg.actor_optimizer.optimizer, optim.Adam)
    assert isinstance(ddpg.critic_optimizer.optimizer, optim.Adam)
    assert isinstance(ddpg.criterion, nn.MSELoss)


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

    ddpg = DDPG(observation_space, action_space)
    state = get_sample_from_space(observation_space)

    print(state)
    training = False
    action = ddpg.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1

    ddpg = DDPG(observation_space, action_space, accelerator=accelerator)
    state = get_sample_from_space(observation_space)
    training = True
    action = ddpg.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1

    ddpg = DDPG(
        observation_space, action_space, O_U_noise=False, accelerator=accelerator
    )
    state = get_sample_from_space(observation_space)
    training = True
    action = ddpg.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1


# learns from experiences and updates network parameters
@pytest.mark.parametrize(
    "min_action, max_action",
    [(-1, 1), ([-1, 0], [1, 1]), ([-1, -1], [0, 1]), ([-1, -2], [1, 0])],
)
def test_learns_from_experiences(min_action, max_action):
    observation_space = generate_random_box_space(shape=(3, 32, 32), low=0, high=255)
    action_space = generate_random_box_space(shape=(2,))
    batch_size = 4
    policy_freq = 4
    net_config_cnn = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }

    # Create an instance of the ddpg class
    ddpg = DDPG(
        observation_space,
        action_space,
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
        states = torch.rand(batch_size, *observation_space.shape)
        actions = torch.randint(0, 2, (batch_size, action_space.shape[0])).float()
        rewards = torch.randn((batch_size, 1))
        next_states = torch.rand(batch_size, *observation_space.shape)
        dones = torch.randint(0, 2, (batch_size, 1))

        experiences = Transition(
            obs=states,
            action=actions,
            reward=rewards,
            next_obs=next_states,
            done=dones,
            batch_size=[batch_size],
        ).to_tensordict()

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
    observation_space = generate_discrete_space(4)
    action_space = generate_random_box_space(shape=(2,))
    batch_size = 64

    # Create an instance of the ddpg class
    ddpg = DDPG(
        observation_space,
        action_space,
        batch_size=batch_size,
        policy_freq=2,
        accelerator=accelerator,
    )

    # Create a batch of experiences
    states = torch.randint(0, observation_space.n, (batch_size, 1)).float()
    actions = torch.randint(0, 2, (batch_size, action_space.shape[0])).float()
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randint(0, observation_space.n, (batch_size, 1)).float()
    dones = torch.randint(0, 2, (batch_size, 1))

    experiences = Transition(
        obs=states,
        action=actions,
        reward=rewards,
        next_obs=next_states,
        done=dones,
        batch_size=[batch_size],
        device=accelerator.device,
    ).to_tensordict()

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
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_random_box_space(shape=(2,))
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

    ddpg = DDPG(
        observation_space,
        action_space,
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

    ddpg.soft_update(ddpg.actor, ddpg.actor_target)

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

    ddpg.soft_update(ddpg.critic, ddpg.critic_target)

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
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_random_box_space(shape=(2,))
    num_envs = 3

    env = DummyEnv(state_size=observation_space.shape, vect=True, num_envs=num_envs)

    # env = make_vect_envs("CartPole-v1", num_envs=num_envs)
    agent = DDPG(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorised env
def test_algorithm_test_loop_unvectorized():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_random_box_space(shape=(2,))

    env = DummyEnv(state_size=observation_space.shape, vect=False)

    agent = DDPG(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with images
def test_algorithm_test_loop_images():
    observation_space = generate_random_box_space(shape=(3, 32, 32), low=0, high=255)
    action_space = generate_random_box_space(shape=(2,))

    env = DummyEnv(state_size=observation_space.shape, vect=True)

    net_config_cnn = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }

    agent = DDPG(
        observation_space=observation_space,
        action_space=action_space,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorized images
def test_algorithm_test_loop_images_unvectorized():
    observation_space = generate_random_box_space(shape=(32, 32, 3), low=0, high=255)
    action_space = generate_random_box_space(shape=(2,))

    env = DummyEnv(state_size=observation_space.shape, vect=False)

    net_config_cnn = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }

    agent = DDPG(
        observation_space=generate_random_box_space(shape=(3, 32, 32), low=0, high=255),
        action_space=action_space,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10, swap_channels=True)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
def test_clone_returns_identical_agent():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_random_box_space(shape=(2,))

    ddpg = DDPG(observation_space, action_space)
    ddpg.fitness = [200, 200, 200]
    ddpg.scores = [94, 94, 94]
    ddpg.steps = [2500]
    ddpg.tensor_attribute = torch.randn(1)
    clone_agent = ddpg.clone()

    assert clone_agent.observation_space == ddpg.observation_space
    assert clone_agent.action_space == ddpg.action_space
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
    ddpg = DDPG(observation_space, action_space, accelerator=accelerator)
    clone_agent = ddpg.clone()

    assert clone_agent.observation_space == ddpg.observation_space
    assert clone_agent.action_space == ddpg.action_space
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
    ddpg = DDPG(observation_space, action_space, accelerator=accelerator, wrap=False)
    clone_agent = ddpg.clone(wrap=False)

    assert clone_agent.observation_space == ddpg.observation_space
    assert clone_agent.action_space == ddpg.action_space
    assert clone_agent.batch_size == ddpg.batch_size
    assert clone_agent.lr_actor == ddpg.lr_actor
    assert clone_agent.lr_critic == ddpg.lr_critic
    assert clone_agent.learn_step == ddpg.learn_step
    assert clone_agent.gamma == ddpg.gamma
    assert clone_agent.tau == ddpg.tau
    assert clone_agent.mut == ddpg.mut
    assert clone_agent.device == ddpg.device
    assert clone_agent.accelerator == ddpg.accelerator
    print(clone_agent.wrap, ddpg.wrap)
    print("1 = ", clone_agent.actor.state_dict())
    print("\n\n2 = ", ddpg.actor.state_dict())
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


def test_clone_new_index():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_random_box_space(shape=(2,))

    ddpg = DDPG(observation_space, action_space)
    clone_agent = ddpg.clone(index=100)

    assert clone_agent.index == 100


def test_clone_after_learning():
    observation_space = generate_random_box_space(shape=(4,))
    action_space = generate_random_box_space(shape=(2,))

    batch_size = 8
    ddpg = DDPG(observation_space, action_space)

    states = torch.randn(batch_size, observation_space.shape[0])
    actions = torch.randn(batch_size, action_space.shape[0])
    rewards = torch.rand(batch_size, 1)
    next_states = torch.randn(batch_size, observation_space.shape[0])
    dones = torch.zeros(batch_size, 1)

    experiences = Transition(
        obs=states,
        action=actions,
        reward=rewards,
        next_obs=next_states,
        done=dones,
        batch_size=[batch_size],
    ).to_tensordict()

    ddpg.learn(experiences)
    clone_agent = ddpg.clone()

    assert clone_agent.observation_space == ddpg.observation_space
    assert clone_agent.action_space == ddpg.action_space
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
    ddpg = DDPG(
        observation_space=generate_random_box_space(shape=(4,)),
        action_space=generate_random_box_space(shape=(2,)),
        accelerator=Accelerator(),
    )
    ddpg.unwrap_models()
    assert isinstance(ddpg.actor.encoder, nn.Module)
    assert isinstance(ddpg.actor_target.encoder, nn.Module)
    assert isinstance(ddpg.critic.encoder, nn.Module)
    assert isinstance(ddpg.critic_target.encoder, nn.Module)


# The saved checkpoint file contains the correct data and format.
def test_save_load_checkpoint_correct_data_and_format(tmpdir):
    # Initialize the ddpg agent
    ddpg = DDPG(
        observation_space=generate_random_box_space(shape=(4,)),
        action_space=generate_random_box_space(shape=(2,)),
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ddpg.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critic_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
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
        observation_space=generate_random_box_space(shape=(3, 32, 32), low=0, high=255),
        action_space=generate_random_box_space(shape=(2,)),
    )
    # Load checkpoint
    ddpg.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(ddpg.actor.encoder, EvolvableMLP)
    assert isinstance(ddpg.actor_target.encoder, EvolvableMLP)
    assert isinstance(ddpg.critic.encoder, EvolvableMLP)
    assert isinstance(ddpg.critic_target.encoder, EvolvableMLP)
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
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }

    # Initialize the ddpg agent
    ddpg = DDPG(
        observation_space=generate_random_box_space(shape=(3, 32, 32), low=0, high=255),
        action_space=generate_random_box_space(shape=(2,)),
        net_config=net_config_cnn,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ddpg.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critic_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
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

    # Load checkpoint
    ddpg = DDPG(
        observation_space=generate_random_box_space(shape=(3, 32, 32), low=0, high=255),
        action_space=generate_random_box_space(shape=(2,)),
    )
    ddpg.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(ddpg.actor.encoder, EvolvableCNN)
    assert isinstance(ddpg.actor_target.encoder, EvolvableCNN)
    assert isinstance(ddpg.critic.encoder, EvolvableCNN)
    assert isinstance(ddpg.critic_target.encoder, EvolvableCNN)
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
    observation_space = spaces.Box(0, 1, shape=(3, 64, 64))
    action_space = generate_random_box_space(shape=(2,))

    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_network = SimpleCNN()
    critic_network = MakeEvolvable(
        critic_network,
        input_tensor,
        torch.randn(1, action_space.shape[0]),
    )

    # Initialize the ddpg agent
    ddpg = DDPG(
        observation_space=observation_space,
        action_space=action_space,
        actor_network=actor_network,
        critic_network=critic_network,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ddpg.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "actor_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "critic_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_target_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
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
        observation_space=generate_random_box_space(shape=(3, 32, 32), low=0, high=255),
        action_space=generate_random_box_space(shape=(2,)),
        actor_network=actor_network,
        critic_network=critic_network,
    )
    # Load checkpoint
    ddpg.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
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
    ddpg = DDPG(
        observation_space=generate_random_box_space(shape=(4,)),
        action_space=spaces.Box(0, 1, shape=(1,)),
        device=device,
    )
    input = torch.tensor(action, dtype=torch.float32).to(device)
    clamped_actions = ddpg.multi_dim_clamp(min, max, input).type(torch.float32)
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
    # Initialize the ddpg agent
    ddpg = DDPG(
        observation_space=generate_random_box_space(shape=(4,)),
        action_space=generate_random_box_space(shape=(2,)),
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ddpg.save_checkpoint(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=device, pickle_module=dill)
    assert "agilerl_version" in checkpoint

    # Create new agent object
    new_ddpg = DDPG.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_ddpg.observation_space == ddpg.observation_space
    assert new_ddpg.action_space == ddpg.action_space
    assert np.all(new_ddpg.min_action == ddpg.min_action)
    assert np.all(new_ddpg.max_action == ddpg.max_action)
    assert isinstance(new_ddpg.actor.encoder, EvolvableMLP)
    assert isinstance(new_ddpg.actor_target.encoder, EvolvableMLP)
    assert isinstance(new_ddpg.critic.encoder, EvolvableMLP)
    assert isinstance(new_ddpg.critic_target.encoder, EvolvableMLP)
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
        observation_space=generate_random_box_space(shape=(3, 32, 32), low=0, high=255),
        action_space=generate_random_box_space(shape=(2,)),
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
    ddpg.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_ddpg = DDPG.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_ddpg.observation_space == ddpg.observation_space
    assert new_ddpg.action_space == ddpg.action_space
    assert np.all(new_ddpg.min_action == ddpg.min_action)
    assert np.all(new_ddpg.max_action == ddpg.max_action)
    assert isinstance(new_ddpg.actor.encoder, EvolvableCNN)
    assert isinstance(new_ddpg.actor_target.encoder, EvolvableCNN)
    assert isinstance(new_ddpg.critic.encoder, EvolvableCNN)
    assert isinstance(new_ddpg.critic_target.encoder, EvolvableCNN)
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
    "observation_space, actor_network, input_tensor",
    [
        (generate_random_box_space(shape=(4,)), "simple_mlp", torch.randn(1, 4)),
        (spaces.Box(0, 1, shape=(3, 64, 64)), "simple_cnn", torch.randn(1, 3, 64, 64)),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_networks(
    observation_space, actor_network, input_tensor, request, tmpdir
):
    action_space = generate_random_box_space(shape=(2,))
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    # Initialize the ddpg agent
    ddpg = DDPG(
        observation_space=observation_space,
        action_space=action_space,
        actor_network=actor_network,
        critic_network=copy.deepcopy(actor_network),
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ddpg.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_ddpg = DDPG.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_ddpg.observation_space == ddpg.observation_space
    assert new_ddpg.action_space == ddpg.action_space
    assert np.all(new_ddpg.min_action == ddpg.min_action)
    assert np.all(new_ddpg.max_action == ddpg.max_action)
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
