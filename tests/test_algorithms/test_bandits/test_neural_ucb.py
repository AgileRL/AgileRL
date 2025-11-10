import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces
from tensordict import TensorDict

from agilerl.algorithms.neural_ucb_bandit import NeuralUCB
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import assert_state_dicts_equal


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


@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        ("vector_space", EvolvableMLP),
        ("image_space", EvolvableCNN),
        ("dict_space", EvolvableMultiInput),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_initialize_bandit(observation_space, encoder_cls, accelerator, request):
    action_space = spaces.Discrete(2)
    observation_space = request.getfixturevalue(observation_space)
    device = accelerator.device if accelerator else "cpu"
    bandit = NeuralUCB(observation_space, action_space, accelerator=accelerator)

    assert bandit.observation_space == observation_space
    assert bandit.action_space == action_space
    assert bandit.batch_size == 64
    assert bandit.lr == 0.001
    assert bandit.learn_step == 2
    assert bandit.gamma == 1.0
    assert bandit.lamb == 1.0
    assert bandit.reg == 0.000625
    assert bandit.mut is None
    assert bandit.device == device
    assert bandit.accelerator == accelerator
    assert bandit.index == 0
    assert bandit.scores == []
    assert bandit.fitness == []
    assert bandit.steps == [0]
    assert isinstance(bandit.actor.encoder, encoder_cls)
    expected_optimizer = AcceleratedOptimizer if accelerator else optim.Adam
    assert isinstance(bandit.optimizer.optimizer, expected_optimizer)
    assert isinstance(bandit.criterion, nn.MSELoss)


# Can initialize NeuralUCB with an actor network
# TODO: Will be deprecated in the future
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        ("vector_space", "simple_mlp", torch.randn(1, 4)),
        ("image_space", "simple_cnn", torch.randn(1, 3, 32, 32)),
    ],
)
def test_initialize_bandit_with_actor_network(
    observation_space, discrete_space, actor_network, input_tensor, request
):
    observation_space = request.getfixturevalue(observation_space)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    bandit = NeuralUCB(observation_space, discrete_space, actor_network=actor_network)

    assert bandit.observation_space == observation_space
    assert bandit.action_space == discrete_space
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


def test_initialize_bandit_with_incorrect_actor_network(vector_space, discrete_space):
    actor_network = nn.Sequential(nn.Linear(vector_space.shape[0], discrete_space.n))

    with pytest.raises(TypeError) as e:
        bandit = NeuralUCB(vector_space, discrete_space, actor_network=actor_network)

        assert bandit
        assert (
            e
            == "'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableModule."
        )


def test_initialize_bandit_with_evo_nets(vector_space, discrete_space):
    actor_network = EvolvableMLP(
        num_inputs=vector_space.shape[0],
        num_outputs=1,
        hidden_size=[64, 64],
        layer_norm=False,
    )

    bandit = NeuralUCB(vector_space, discrete_space, actor_network=actor_network)
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


def test_initialize_neuralucb_with_incorrect_actor_net_type(
    vector_space, discrete_space
):
    actor_network = "dummy"
    with pytest.raises(TypeError) as a:
        bandit = NeuralUCB(vector_space, discrete_space, actor_network=actor_network)
        assert bandit
        assert (
            str(a.value)
            == f"'actor_network' argument is of type {type(actor_network)}, but must be of type EvolvableMLP, EvolvableCNN or MakeEvolvable"
        )


# Returns the expected action when given a state observation and epsilon=0 or 1.
def test_returns_expected_action(vector_space, discrete_space):
    bandit = NeuralUCB(vector_space, discrete_space)
    state = np.array([1, 2, 3, 4])

    action_mask = None

    action = bandit.get_action(state, action_mask)

    assert action.is_integer()
    assert action >= 0 and action < discrete_space.n


# Returns the expected action when given a state observation and action mask.
def test_returns_expected_action_mask(vector_space, discrete_space):
    accelerator = Accelerator()

    bandit = NeuralUCB(vector_space, discrete_space, accelerator=accelerator)
    state = np.array([1, 2, 3, 4])

    action_mask = np.array([0, 1])

    action = bandit.get_action(state, action_mask)

    assert action.is_integer()
    assert action == 1


# learns from experiences and updates network parameters
@pytest.mark.parametrize("observation_space", ["vector_space", "image_space"])
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_learns_from_experiences(
    observation_space, discrete_space, accelerator, request
):
    observation_space = request.getfixturevalue(observation_space)
    batch_size = 64

    # Create an instance of the NeuralTS class
    bandit = NeuralUCB(
        observation_space,
        discrete_space,
        batch_size=batch_size,
        accelerator=accelerator,
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, *observation_space.shape)
    rewards = torch.randn((batch_size, 1))

    experiences = TensorDict(
        {"obs": states, "reward": rewards},
        batch_size=[batch_size],
        device=bandit.device,
    )

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
@pytest.mark.parametrize("observation_space", ["vector_space", "image_space"])
def test_algorithm_test_loop(observation_space, discrete_space, request):
    observation_space = request.getfixturevalue(observation_space)

    env = DummyBanditEnv(state_size=observation_space.shape, arms=discrete_space.n)

    agent = NeuralUCB(observation_space=observation_space, action_space=discrete_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
@pytest.mark.parametrize("observation_space", ["vector_space"])
def test_clone_returns_identical_agent(observation_space, discrete_space, request):
    observation_space = request.getfixturevalue(observation_space)
    bandit = DummyNeuralUCB(observation_space, discrete_space)
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), bandit.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), bandit.optimizer.state_dict()
    )
    assert clone_agent.fitness == bandit.fitness
    assert clone_agent.steps == bandit.steps
    assert clone_agent.scores == bandit.scores
    assert clone_agent.tensor_attribute == bandit.tensor_attribute
    assert clone_agent.tensor_test == bandit.tensor_test
    assert clone_agent.numpy_attribute == bandit.numpy_attribute
    assert clone_agent.numpy_test == bandit.numpy_test

    accelerator = Accelerator()
    bandit = NeuralUCB(observation_space, discrete_space, accelerator=accelerator)
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), bandit.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), bandit.optimizer.state_dict()
    )
    assert clone_agent.fitness == bandit.fitness
    assert clone_agent.steps == bandit.steps
    assert clone_agent.scores == bandit.scores

    accelerator = Accelerator()
    bandit = NeuralUCB(
        observation_space, discrete_space, accelerator=accelerator, wrap=False
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), bandit.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), bandit.optimizer.state_dict()
    )
    assert clone_agent.fitness == bandit.fitness
    assert clone_agent.steps == bandit.steps
    assert clone_agent.scores == bandit.scores


def test_clone_new_index(vector_space, discrete_space):
    bandit = NeuralUCB(vector_space, discrete_space)
    clone_agent = bandit.clone(index=100)
    assert clone_agent.index == 100


def test_clone_after_learning(vector_space, discrete_space):
    batch_size = 4
    states = torch.randn(batch_size, vector_space.shape[0])
    rewards = torch.rand(batch_size, 1)
    experiences = TensorDict(
        {"obs": states, "reward": rewards},
        batch_size=[batch_size],
    )
    bandit = NeuralUCB(vector_space, discrete_space, batch_size=batch_size)
    bandit.learn(experiences)
    clone_agent = bandit.clone()

    assert clone_agent.observation_space == bandit.observation_space
    assert clone_agent.action_space == discrete_space
    assert clone_agent.batch_size == bandit.batch_size
    assert clone_agent.lr == bandit.lr
    assert clone_agent.learn_step == bandit.learn_step
    assert clone_agent.gamma == bandit.gamma
    assert clone_agent.mut == bandit.mut
    assert clone_agent.device == bandit.device
    assert clone_agent.accelerator == bandit.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), bandit.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), bandit.optimizer.state_dict()
    )
    assert clone_agent.fitness == bandit.fitness
    assert clone_agent.steps == bandit.steps
    assert clone_agent.scores == bandit.scores


# TODO: Will be deprecated in the future
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        ("vector_space", "simple_mlp", torch.randn(1, 4)),
    ],
)
def test_clone_with_make_evo(
    observation_space, discrete_space, actor_network, input_tensor, request
):
    observation_space = request.getfixturevalue(observation_space)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    bandit = NeuralUCB(observation_space, discrete_space, actor_network=actor_network)
    clone_agent = bandit.clone()

    assert clone_agent.observation_space == bandit.observation_space
    assert clone_agent.action_space == discrete_space
    assert clone_agent.batch_size == bandit.batch_size
    assert clone_agent.lr == bandit.lr
    assert clone_agent.learn_step == bandit.learn_step
    assert clone_agent.gamma == bandit.gamma
    assert clone_agent.mut == bandit.mut
    assert clone_agent.device == bandit.device
    assert clone_agent.accelerator == bandit.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), bandit.actor.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), bandit.optimizer.state_dict()
    )
    assert clone_agent.fitness == bandit.fitness
    assert clone_agent.steps == bandit.steps
    assert clone_agent.scores == bandit.scores
