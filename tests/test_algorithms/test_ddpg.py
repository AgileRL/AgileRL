import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces

from agilerl.algorithms.ddpg import DDPG
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


class DummyDDPG(DDPG):
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


# initialize ddpg with valid parameters
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
def test_initialize_ddpg(observation_space, encoder_cls, accelerator, request):
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    observation_space = request.getfixturevalue(observation_space)
    ddpg = DDPG(observation_space, action_space, accelerator=accelerator)

    expected_opt_cls = AcceleratedOptimizer if accelerator else optim.Adam
    expected_device = accelerator.device if accelerator else "cpu"
    assert ddpg.observation_space == observation_space
    assert ddpg.action_space == action_space
    assert ddpg.batch_size == 64
    assert ddpg.lr_actor == 0.0001
    assert ddpg.lr_critic == 0.001
    assert ddpg.learn_step == 5
    assert ddpg.gamma == 0.99
    assert ddpg.tau == 0.001
    assert ddpg.mut is None
    assert ddpg.device == expected_device
    assert ddpg.accelerator == accelerator
    assert ddpg.index == 0
    assert ddpg.scores == []
    assert ddpg.fitness == []
    assert ddpg.steps == [0]
    assert isinstance(ddpg.actor.encoder, encoder_cls)
    assert isinstance(ddpg.actor_target.encoder, encoder_cls)
    assert isinstance(ddpg.actor_optimizer.optimizer, expected_opt_cls)
    assert isinstance(ddpg.critic.encoder, encoder_cls)
    assert isinstance(ddpg.critic_target.encoder, encoder_cls)
    assert isinstance(ddpg.critic_optimizer.optimizer, expected_opt_cls)
    assert isinstance(ddpg.criterion, nn.MSELoss)


# Can initialize ddpg with an actor network
# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "observation_space, actor_network, critic_network, input_tensor, input_tensor_critic",
    [
        (
            "vector_space",
            "simple_mlp",
            "simple_mlp",
            torch.randn(1, 4),
            torch.randn(1, 4),
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
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    observation_space = request.getfixturevalue(observation_space)
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


def test_initialize_ddpg_with_actor_network_evo_net(vector_space):
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    actor_network = DeterministicActor(vector_space, action_space)
    critic_network = ContinuousQNetwork(vector_space, action_space)

    ddpg = DDPG(
        vector_space,
        action_space,
        actor_network=actor_network,
        critic_network=critic_network,
    )

    assert ddpg.observation_space == vector_space
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


def test_initialize_ddpg_with_incorrect_actor_net(vector_space):
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    actor_network = "dummy"
    critic_network = "dummy"
    with pytest.raises(TypeError):
        ddpg = DDPG(
            vector_space,
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
            "vector_space",
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
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    observation_space = request.getfixturevalue(observation_space)
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
        "vector_space",
        "discrete_space",
        "image_space",
        "dict_space",
        "multidiscrete_space",
    ],
)
def test_returns_expected_action_training(observation_space, request):
    accelerator = Accelerator()
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    observation_space = request.getfixturevalue(observation_space)

    # Test without accelerator
    ddpg = DDPG(observation_space, action_space)
    state = get_sample_from_space(observation_space)

    training = False
    action = ddpg.get_action(state, training)[0]
    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1

    # Test with accelerator
    ddpg = DDPG(observation_space, action_space, accelerator=accelerator)
    state = get_sample_from_space(observation_space)
    training = True
    action = ddpg.get_action(state, training)[0]

    assert len(action) == action_space.shape[0]
    for act in action:
        assert isinstance(act, np.float32)
        assert -1 <= act <= 1

    # Test without OU noise
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
    "observation_space",
    [
        "vector_space",
        "discrete_space",
        "image_space",
        "dict_space",
        "multidiscrete_space",
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_learns_from_experiences(observation_space, accelerator, request):
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    observation_space = request.getfixturevalue(observation_space)
    batch_size = 4
    policy_freq = 4

    # Create an instance of the ddpg class
    ddpg = DDPG(
        observation_space,
        action_space,
        batch_size=batch_size,
        policy_freq=policy_freq,
        accelerator=accelerator,
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = ddpg.actor
    actor_target = ddpg.actor_target
    actor_pre_learn_sd = copy.deepcopy(ddpg.actor.state_dict())
    critic = ddpg.critic
    critic_target = ddpg.critic_target
    critic_pre_learn_sd = copy.deepcopy(ddpg.critic.state_dict())

    for i in range(policy_freq * 2):
        # Create a batch of experiences & learn
        device = accelerator.device if accelerator else "cpu"
        experiences = get_experiences_batch(
            observation_space, action_space, batch_size, device
        )
        ddpg.scores.append(0)
        actor_loss, critic_loss = ddpg.learn(experiences)

    assert isinstance(actor_loss, float)
    assert isinstance(critic_loss, float)
    assert critic_loss >= 0.0
    assert actor == ddpg.actor
    assert actor_target == ddpg.actor_target
    assert_not_equal_state_dict(actor_pre_learn_sd, ddpg.actor.state_dict())
    assert critic == ddpg.critic
    assert critic_target == ddpg.critic_target
    assert_not_equal_state_dict(critic_pre_learn_sd, ddpg.critic.state_dict())


# Updates target network parameters with soft update
def test_soft_update():
    observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
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
@pytest.mark.parametrize("observation_space", ["vector_space", "image_space"])
@pytest.mark.parametrize("num_envs", [1, 3])
def test_algorithm_test_loop(observation_space, num_envs, request):
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    observation_space = request.getfixturevalue(observation_space)

    vect = num_envs > 1
    env = DummyEnv(state_size=observation_space.shape, vect=vect, num_envs=num_envs)

    # env = make_vect_envs("CartPole-v1", num_envs=num_envs)
    agent = DDPG(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
@pytest.mark.parametrize("observation_space", ["vector_space"])
def test_clone_returns_identical_agent(observation_space, request):
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    observation_space = request.getfixturevalue(observation_space)

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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ddpg.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), ddpg.actor_target.state_dict()
    )
    assert_state_dicts_equal(clone_agent.critic.state_dict(), ddpg.critic.state_dict())
    assert_state_dicts_equal(
        clone_agent.critic_target.state_dict(), ddpg.critic_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_optimizer.state_dict(), ddpg.actor_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_optimizer.state_dict(), ddpg.critic_optimizer.state_dict()
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ddpg.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), ddpg.actor_target.state_dict()
    )
    assert_state_dicts_equal(clone_agent.critic.state_dict(), ddpg.critic.state_dict())
    assert_state_dicts_equal(
        clone_agent.critic_target.state_dict(), ddpg.critic_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_optimizer.state_dict(), ddpg.actor_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_optimizer.state_dict(), ddpg.critic_optimizer.state_dict()
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ddpg.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), ddpg.actor_target.state_dict()
    )
    assert_state_dicts_equal(clone_agent.critic.state_dict(), ddpg.critic.state_dict())
    assert_state_dicts_equal(
        clone_agent.critic_target.state_dict(), ddpg.critic_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_optimizer.state_dict(), ddpg.actor_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_optimizer.state_dict(), ddpg.critic_optimizer.state_dict()
    )
    assert clone_agent.fitness == ddpg.fitness
    assert clone_agent.steps == ddpg.steps
    assert clone_agent.scores == ddpg.scores


def test_clone_new_index():
    observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    ddpg = DDPG(observation_space, action_space)
    clone_agent = ddpg.clone(index=100)

    assert clone_agent.index == 100


def test_clone_after_learning():
    observation_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

    batch_size = 8
    ddpg = DDPG(observation_space, action_space)

    experiences = get_experiences_batch(observation_space, action_space, batch_size)

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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ddpg.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), ddpg.actor_target.state_dict()
    )
    assert_state_dicts_equal(clone_agent.critic.state_dict(), ddpg.critic.state_dict())
    assert_state_dicts_equal(
        clone_agent.critic_target.state_dict(), ddpg.critic_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_optimizer.state_dict(), ddpg.actor_optimizer.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.critic_optimizer.state_dict(), ddpg.critic_optimizer.state_dict()
    )
    assert clone_agent.fitness == ddpg.fitness
    assert clone_agent.steps == ddpg.steps
    assert clone_agent.scores == ddpg.scores


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
def test_multi_dim_clamp(min, max, action, expected_result, vector_space, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    if isinstance(min, list):
        min = np.array(min)
    if isinstance(max, list):
        max = np.array(max)

    ddpg = DDPG(
        observation_space=vector_space,
        action_space=spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        device=device,
    )
    input = torch.tensor(action, dtype=torch.float32).to(device)
    clamped_actions = ddpg.multi_dim_clamp(min, max, input).type(torch.float32)
    expected_result = torch.tensor(expected_result)
    assert clamped_actions.dtype == expected_result.dtype
