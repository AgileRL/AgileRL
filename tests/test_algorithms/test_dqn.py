import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer

from agilerl.algorithms.dqn import DQN
from agilerl.components.data import Transition
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import (
    assert_state_dicts_equal,
    get_experiences_batch,
    get_sample_from_space,
)

# Cleanup fixture moved to conftest.py for better performance


class DummyDQN(DQN):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs)

        self.tensor_test = torch.randn(1)


class DummyEnv:
    def __init__(self, observation_space, vect=True, num_envs=2):
        self.observation_space = observation_space.shape
        self.vect = vect
        if self.vect:
            self.observation_space = (num_envs,) + self.observation_space
            self.n_envs = num_envs
            self.num_envs = num_envs
        else:
            self.n_envs = 1

    def reset(self):
        return np.random.rand(*self.observation_space), {}

    def step(self, action):
        return (
            np.random.rand(*self.observation_space),
            np.random.randint(0, 5, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            np.random.randint(0, 2, self.n_envs),
            {},
        )


# initialize DQN with valid parameters
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
def test_initialize_dqn(
    observation_space, encoder_cls, accelerator, discrete_space, request
):
    action_space = discrete_space
    observation_space = request.getfixturevalue(observation_space)
    dqn = DQN(observation_space, action_space, accelerator=accelerator)

    expected_device = accelerator.device if accelerator else "cpu"
    assert dqn.observation_space == observation_space
    assert dqn.action_space == action_space
    assert dqn.batch_size == 64
    assert dqn.lr == 0.0001
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 0.001
    assert dqn.mut is None
    assert dqn.device == expected_device
    assert dqn.accelerator == accelerator
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]
    assert dqn.double is False
    # assert dqn.actor_network is None
    assert isinstance(dqn.actor.encoder, encoder_cls)
    assert isinstance(dqn.actor_target.encoder, encoder_cls)
    expected_opt_cls = AcceleratedOptimizer if accelerator else optim.Adam
    assert isinstance(dqn.optimizer.optimizer, expected_opt_cls)
    assert isinstance(dqn.criterion, nn.MSELoss)


# Can initialize DQN with an actor network
# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        ("vector_space", "simple_mlp", torch.randn(1, 4)),
        (
            "image_space",
            "simple_cnn",
            torch.randn(1, 3, 32, 32),
        ),
    ],
)
def test_initialize_dqn_with_actor_network_make_evo(
    observation_space, actor_network, input_tensor, request, discrete_space
):
    action_space = discrete_space
    observation_space = request.getfixturevalue(observation_space)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    dqn = DQN(observation_space, action_space, actor_network=actor_network)

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
    assert dqn.double is False
    assert isinstance(dqn.optimizer.optimizer, optim.Adam)
    assert isinstance(dqn.criterion, nn.MSELoss)


@pytest.mark.parametrize(
    "observation_space, net_type",
    [
        ("vector_space", "mlp"),
        ("image_space", "cnn"),
    ],
)
def test_initialize_dqn_with_actor_network_evo_net(
    observation_space, net_type, discrete_space, request
):
    action_space = discrete_space
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

    dqn = DQN(observation_space, action_space, actor_network=actor_network)

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
    assert dqn.double is False
    assert isinstance(dqn.optimizer.optimizer, optim.Adam)
    assert isinstance(dqn.criterion, nn.MSELoss)


def test_initialize_dqn_with_incorrect_actor_net_type(vector_space, discrete_space):
    actor_network = "dummy"

    with pytest.raises(TypeError) as a:
        dqn = DQN(vector_space, discrete_space, actor_network=actor_network)

        assert dqn
        assert (
            str(a.value)
            == f"'actor_network' argument is of type {type(actor_network)}, but must be of type nn.Module."
        )


# Returns the expected action when given a state observation and epsilon=0 or 1.
@pytest.mark.parametrize(
    "observation_space",
    [
        "vector_space",
        "discrete_space",
        "image_space",
        "multidiscrete_space",
        "dict_space",
    ],
)
def test_returns_expected_action_epsilon_greedy(
    observation_space, discrete_space, request
):
    action_space = discrete_space
    observation_space = request.getfixturevalue(observation_space)

    dqn = DQN(observation_space, action_space)
    state = get_sample_from_space(observation_space)

    action_mask = None

    epsilon = 0
    action = dqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_space.n

    epsilon = 1
    action = dqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < action_space.n


# Returns the expected action when given a state observation and action mask.
def test_returns_expected_action_mask(vector_space, discrete_space):
    accelerator = Accelerator()

    dqn = DQN(vector_space, discrete_space, accelerator=accelerator)
    state = get_sample_from_space(vector_space)

    action_mask = np.array([0, 1])

    epsilon = 0
    action = dqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action == 1

    epsilon = 1
    action = dqn.get_action(state, epsilon, action_mask)[0]

    assert action.is_integer()
    assert action == 1


def test_returns_expected_action_mask_vectorized(vector_space, discrete_space):
    accelerator = Accelerator()
    dqn = DQN(vector_space, discrete_space, accelerator=accelerator)
    state = get_sample_from_space(vector_space, batch_size=2)

    action_mask = np.array([[0, 1], [1, 0]])

    epsilon = 0
    action = dqn.get_action(state, epsilon, action_mask)

    assert np.array_equal(action, [1, 0])

    epsilon = 1
    action = dqn.get_action(state, epsilon, action_mask)

    assert np.array_equal(action, [1, 0])


def test_dqn_optimizer_parameters(vector_space, discrete_space):
    dqn = DQN(vector_space, discrete_space)

    # Store initial parameters
    initial_params = {
        name: param.clone() for name, param in dqn.actor.named_parameters()
    }

    # Perform a dummy optimization step
    dummy_input = torch.randn(1, 4)
    dummy_return = torch.tensor([1.0])

    q_eval = dqn.actor(dummy_input)
    loss = (dummy_return - q_eval) ** 2
    loss = loss.mean()
    dqn.optimizer.zero_grad()
    loss.backward()
    dqn.optimizer.step()

    # Check if parameters have changed
    not_updated = []
    for name, param in dqn.actor.named_parameters():
        if torch.equal(initial_params[name], param):
            not_updated.append(name)

    assert not not_updated, f"The following parameters weren't updated:\n{not_updated}"


# learns from experiences and updates network parameters
@pytest.mark.parametrize(
    "observation_space",
    [
        "vector_space",
        "discrete_space",
        "image_space",
        "multidiscrete_space",
        "dict_space",
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("double", [False, True])
def test_learns_from_experiences(
    observation_space, accelerator, double, discrete_space, request
):
    action_space = discrete_space
    observation_space = request.getfixturevalue(observation_space)
    batch_size = 64

    # Create an instance of the DQN class
    dqn = DQN(
        observation_space,
        action_space,
        batch_size=batch_size,
        accelerator=accelerator,
        double=double,
    )

    # Create a batch of experiences
    device = accelerator.device if accelerator else "cpu"
    experiences = get_experiences_batch(
        observation_space, action_space, batch_size, device
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = str(copy.deepcopy(dqn.actor.state_dict()))

    # Call the learn method
    loss = dqn.learn(experiences)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert actor_pre_learn_sd != str(dqn.actor.state_dict())


# Updates target network parameters with soft update
def test_soft_update(vector_space, discrete_space):
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

    dqn = DQN(
        vector_space,
        discrete_space,
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
@pytest.mark.parametrize("observation_space", ["vector_space", "image_space"])
@pytest.mark.parametrize("num_envs", [1, 3])
def test_algorithm_test_loop(observation_space, discrete_space, num_envs, request):
    observation_space = request.getfixturevalue(observation_space)
    vect = num_envs > 1
    env = DummyEnv(observation_space=observation_space, vect=vect, num_envs=num_envs)
    agent = DQN(observation_space=observation_space, action_space=discrete_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
def test_clone_returns_identical_agent(vector_space, discrete_space):
    dqn = DummyDQN(vector_space, discrete_space)
    dqn.tensor_attribute = torch.randn(1)
    clone_agent = dqn.clone()

    assert clone_agent.observation_space == dqn.observation_space
    assert clone_agent.action_space == dqn.action_space
    # assert clone_agent.actor_network == dqn.actor_network
    assert clone_agent.batch_size == dqn.batch_size
    assert clone_agent.lr == dqn.lr
    assert clone_agent.learn_step == dqn.learn_step
    assert clone_agent.gamma == dqn.gamma
    assert clone_agent.tau == dqn.tau
    assert clone_agent.mut == dqn.mut
    assert clone_agent.device == dqn.device
    assert clone_agent.accelerator == dqn.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), dqn.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), dqn.optimizer.state_dict()
    )
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores
    assert clone_agent.tensor_attribute == dqn.tensor_attribute
    assert clone_agent.tensor_test == dqn.tensor_test

    accelerator = Accelerator()
    dqn = DQN(vector_space, discrete_space, accelerator=accelerator)
    clone_agent = dqn.clone()

    assert clone_agent.observation_space == dqn.observation_space
    assert clone_agent.action_space == dqn.action_space
    assert clone_agent.batch_size == dqn.batch_size
    assert clone_agent.lr == dqn.lr
    assert clone_agent.learn_step == dqn.learn_step
    assert clone_agent.gamma == dqn.gamma
    assert clone_agent.tau == dqn.tau
    assert clone_agent.mut == dqn.mut
    assert clone_agent.device == dqn.device
    assert clone_agent.accelerator == dqn.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), dqn.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), dqn.optimizer.state_dict()
    )
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores

    accelerator = Accelerator()
    dqn = DQN(vector_space, discrete_space, accelerator=accelerator, wrap=False)
    clone_agent = dqn.clone(wrap=False)

    assert clone_agent.observation_space == dqn.observation_space
    assert clone_agent.action_space == dqn.action_space
    assert clone_agent.batch_size == dqn.batch_size
    assert clone_agent.lr == dqn.lr
    assert clone_agent.learn_step == dqn.learn_step
    assert clone_agent.gamma == dqn.gamma
    assert clone_agent.tau == dqn.tau
    assert clone_agent.mut == dqn.mut
    assert clone_agent.device == dqn.device
    assert clone_agent.accelerator == dqn.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), dqn.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), dqn.optimizer.state_dict()
    )
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores


def test_clone_new_index(vector_space, discrete_space):
    dqn = DummyDQN(vector_space, discrete_space)
    clone_agent = dqn.clone(index=100)

    assert clone_agent.index == 100


def test_clone_after_learning(vector_space, discrete_space):
    batch_size = 8
    dqn = DQN(vector_space, discrete_space)

    states = torch.randn(batch_size, vector_space.shape[0])
    actions = torch.randint(0, 2, (batch_size, 1))
    rewards = torch.rand(batch_size, 1)
    next_states = torch.randn(batch_size, vector_space.shape[0])
    dones = torch.zeros(batch_size, 1)

    experiences = Transition(
        obs=states,
        action=actions,
        reward=rewards,
        next_obs=next_states,
        done=dones,
    ).to_tensordict()

    dqn.learn(experiences)
    clone_agent = dqn.clone()

    assert clone_agent.observation_space == dqn.observation_space
    assert clone_agent.action_space == dqn.action_space
    assert clone_agent.batch_size == dqn.batch_size
    assert clone_agent.lr == dqn.lr
    assert clone_agent.learn_step == dqn.learn_step
    assert clone_agent.gamma == dqn.gamma
    assert clone_agent.tau == dqn.tau
    assert clone_agent.mut == dqn.mut
    assert clone_agent.device == dqn.device
    assert clone_agent.accelerator == dqn.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), dqn.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), dqn.optimizer.state_dict()
    )
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores
