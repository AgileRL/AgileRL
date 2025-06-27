import copy

import numpy as np
import pytest
import torch
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from tensordict import TensorDict

from agilerl.algorithms.dqn_rainbow import RainbowDQN
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.networks.q_networks import RainbowQNetwork
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import (
    assert_not_equal_state_dict,
    assert_state_dicts_equal,
    get_experiences_batch,
    get_sample_from_space,
)


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
    observation_space, discrete_space, encoder_cls, accelerator, request
):
    observation_space = request.getfixturevalue(observation_space)
    dqn = RainbowDQN(observation_space, discrete_space, accelerator=accelerator)

    expected_device = accelerator.device if accelerator else "cpu"
    assert dqn.observation_space == observation_space
    assert dqn.action_space == discrete_space
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
    # assert dqn.actor_network is None
    assert isinstance(dqn.actor.encoder, encoder_cls)
    assert isinstance(dqn.actor_target.encoder, encoder_cls)
    expected_opt_cls = AcceleratedOptimizer if accelerator else optim.Adam
    assert isinstance(dqn.optimizer.optimizer, expected_opt_cls)


@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        ("vector_space", EvolvableMLP),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_initialize_dqn_with_actor_network_evo_net(
    observation_space, discrete_space, encoder_cls, accelerator, request
):
    observation_space = request.getfixturevalue(observation_space)
    support = torch.linspace(0, 1, 51)
    device = accelerator.device if accelerator else "cpu"
    actor_network = RainbowQNetwork(
        observation_space=observation_space,
        action_space=discrete_space,
        support=support,
        device=device,
    )

    # Create an instance of the RainbowDQN class
    dqn = RainbowDQN(
        observation_space,
        discrete_space,
        actor_network=actor_network,
        accelerator=accelerator,
    )

    assert dqn.observation_space == observation_space
    assert dqn.action_space == discrete_space
    assert dqn.batch_size == 64
    assert dqn.lr == 0.0001
    assert dqn.learn_step == 5
    assert dqn.gamma == 0.99
    assert dqn.tau == 0.001
    assert dqn.mut is None
    assert dqn.device == device
    assert dqn.accelerator == accelerator
    assert dqn.index == 0
    assert dqn.scores == []
    assert dqn.fitness == []
    assert dqn.steps == [0]

    assert isinstance(dqn.actor.encoder, encoder_cls)
    assert isinstance(dqn.actor_target.encoder, encoder_cls)
    if accelerator is not None:
        assert isinstance(dqn.optimizer.optimizer, AcceleratedOptimizer)
    else:
        assert isinstance(dqn.optimizer.optimizer, optim.Adam)


def test_initialize_dqn_with_incorrect_actor_net_type(
    vector_space, discrete_space, request
):
    actor_network = "dummy"

    with pytest.raises(TypeError) as a:
        dqn = RainbowDQN(vector_space, discrete_space, actor_network=actor_network)
        assert dqn
        assert (
            str(a.value)
            == f"'actor_network' argument is of type {type(actor_network)}, but must be of type nn.Module."
        )


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
def test_initialize_dqn_with_make_evolvable(
    observation_space, discrete_space, actor_network, input_tensor, request
):
    observation_space = request.getfixturevalue(observation_space)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    dqn = RainbowDQN(observation_space, discrete_space, actor_network=actor_network)

    assert dqn.observation_space == observation_space
    assert dqn.action_space == discrete_space
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
def test_returns_expected_action(
    accelerator, observation_space, discrete_space, request
):
    observation_space = request.getfixturevalue(observation_space)
    dqn = RainbowDQN(observation_space, discrete_space, accelerator=accelerator)
    state = get_sample_from_space(observation_space)

    action_mask = None

    action = dqn.get_action(state, action_mask)[0]

    assert action.is_integer()
    assert action >= 0 and action < discrete_space.n

    action_mask = np.array([0, 1])

    action = dqn.get_action(state, action_mask)[0]

    assert action.is_integer()
    assert action == 1


def test_returns_expected_action_mask_vectorized(vector_space, discrete_space):
    accelerator = Accelerator()

    dqn = RainbowDQN(vector_space, discrete_space, accelerator=accelerator)
    state = np.array([[1, 2, 4, 5], [2, 3, 5, 1]])

    action_mask = np.array([[0, 1], [1, 0]])

    action = dqn.get_action(state, action_mask)

    assert np.array_equal(action, [1, 0])


@pytest.mark.parametrize(
    "observation_space",
    [
        "discrete_space",
        "vector_space",
        "dict_space",
        "multidiscrete_space",
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
# learns from experiences and updates network parameters
def test_learns_from_experiences(
    accelerator, observation_space, discrete_space, request
):
    observation_space = request.getfixturevalue(observation_space)
    torch.autograd.set_detect_anomaly(True)
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        observation_space,
        discrete_space,
        batch_size=batch_size,
        accelerator=accelerator,
    )

    # Create a batch of experiences
    device = accelerator.device if accelerator else "cpu"
    experiences = get_experiences_batch(
        observation_space, discrete_space, batch_size, device
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = copy.deepcopy(dqn.actor.state_dict())
    actor_target_pre_learn_sd = copy.deepcopy(dqn.actor_target.state_dict())

    # Call the learn method
    loss, new_idxs, new_priorities = dqn.learn(experiences, per=False)

    assert loss > 0.0
    assert new_idxs is None
    assert new_priorities is None
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert_not_equal_state_dict(actor_pre_learn_sd, dqn.actor.state_dict())
    assert_not_equal_state_dict(
        actor_target_pre_learn_sd, dqn.actor_target.state_dict()
    )


@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("combined", [True, False])
# learns from experiences and updates network parameters
def test_learns_from_experiences_n_step(
    accelerator, combined, vector_space, discrete_space
):
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        vector_space,
        discrete_space,
        batch_size=batch_size,
        accelerator=accelerator,
        combined_reward=combined,
    )

    # Create a batch of experiences
    # Create a batch of experiences
    states = torch.randn(batch_size, vector_space.shape[0])
    actions = torch.randint(0, discrete_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, vector_space.shape[0])
    dones = torch.randint(0, 2, (batch_size, 1))
    idxs = np.arange(batch_size)
    n_states = torch.randn(batch_size, vector_space.shape[0])
    n_actions = torch.randint(0, discrete_space.n, (batch_size, 1))
    n_rewards = torch.randn((batch_size, 1))
    n_next_states = torch.randn(batch_size, vector_space.shape[0])
    n_dones = torch.randint(0, 2, (batch_size, 1))

    experiences = TensorDict(
        {
            "obs": states,
            "action": actions,
            "reward": rewards,
            "next_obs": next_states,
            "done": dones,
            "idxs": idxs,
        },
        batch_size=[batch_size],
        device=accelerator.device if accelerator else "cpu",
    )

    n_experiences = TensorDict(
        {
            "obs": n_states,
            "action": n_actions,
            "reward": n_rewards,
            "next_obs": n_next_states,
            "done": n_dones,
        },
        batch_size=[batch_size],
        device=accelerator.device if accelerator else "cpu",
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = copy.deepcopy(dqn.actor.state_dict())
    actor_target_pre_learn_sd = copy.deepcopy(dqn.actor_target.state_dict())

    # Call the learn method
    loss, new_idxs, new_priorities = dqn.learn(experiences, n_experiences, per=False)

    assert loss > 0.0
    assert new_idxs is not None
    assert new_priorities is None
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert_not_equal_state_dict(actor_pre_learn_sd, dqn.actor.state_dict())
    assert_not_equal_state_dict(
        actor_target_pre_learn_sd, dqn.actor_target.state_dict()
    )


# learns from experiences and updates network parameters
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("combined", [True, False])
def test_learns_from_experiences_per(
    accelerator, combined, vector_space, discrete_space
):
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        vector_space,
        discrete_space,
        batch_size=batch_size,
        accelerator=accelerator,
        combined_reward=combined,
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, vector_space.shape[0])
    actions = torch.randint(0, discrete_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, vector_space.shape[0])
    dones = torch.randint(0, 2, (batch_size, 1))
    weights = torch.rand(batch_size)
    idxs = torch.from_numpy(np.arange(batch_size))

    experiences = TensorDict(
        {
            "obs": states,
            "action": actions,
            "reward": rewards,
            "next_obs": next_states,
            "done": dones,
            "idxs": idxs,
            "weights": weights,
        },
        batch_size=[batch_size],
        device=accelerator.device if accelerator else "cpu",
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = copy.deepcopy(dqn.actor.state_dict())
    actor_target_pre_learn_sd = copy.deepcopy(dqn.actor_target.state_dict())

    # Call the learn method
    loss, new_idxs, new_priorities = dqn.learn(experiences, per=True)

    assert loss > 0.0
    assert isinstance(new_idxs, torch.Tensor)
    assert isinstance(new_priorities, np.ndarray)
    assert torch.equal(new_idxs.cpu(), idxs)
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert_not_equal_state_dict(actor_pre_learn_sd, dqn.actor.state_dict())
    assert_not_equal_state_dict(
        actor_target_pre_learn_sd, dqn.actor_target.state_dict()
    )


# learns from experiences and updates network parameters
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("combined", [True, False])
def test_learns_from_experiences_per_n_step(
    accelerator, combined, vector_space, discrete_space
):
    batch_size = 64

    # Create an instance of the DQN class
    dqn = RainbowDQN(
        vector_space,
        discrete_space,
        batch_size=batch_size,
        accelerator=accelerator,
        combined_reward=combined,
    )

    # Create a batch of experiences
    states = torch.randn(batch_size, vector_space.shape[0])
    actions = torch.randint(0, discrete_space.n, (batch_size, 1))
    rewards = torch.randn((batch_size, 1))
    next_states = torch.randn(batch_size, vector_space.shape[0])
    dones = torch.randint(0, 2, (batch_size, 1))
    weights = torch.rand(batch_size)
    idxs = torch.from_numpy(np.arange(batch_size))
    n_states = torch.randn(batch_size, vector_space.shape[0])
    n_actions = torch.randint(0, discrete_space.n, (batch_size, 1))
    n_rewards = torch.randn((batch_size, 1))
    n_next_states = torch.randn(batch_size, vector_space.shape[0])
    n_dones = torch.randint(0, 2, (batch_size, 1))

    experiences = TensorDict(
        {
            "obs": states,
            "action": actions,
            "reward": rewards,
            "next_obs": next_states,
            "done": dones,
            "idxs": idxs,
            "weights": weights,
        },
        batch_size=[batch_size],
        device=accelerator.device if accelerator else "cpu",
    )

    n_experiences = TensorDict(
        {
            "obs": n_states,
            "action": n_actions,
            "reward": n_rewards,
            "next_obs": n_next_states,
            "done": n_dones,
        },
        batch_size=[batch_size],
        device=accelerator.device if accelerator else "cpu",
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = dqn.actor
    actor_target = dqn.actor_target
    actor_pre_learn_sd = copy.deepcopy(dqn.actor.state_dict())
    actor_target_pre_learn_sd = copy.deepcopy(dqn.actor_target.state_dict())

    # Call the learn method
    loss, new_idxs, new_priorities = dqn.learn(experiences, n_experiences, per=True)

    assert loss > 0.0
    assert isinstance(new_idxs, torch.Tensor)
    assert isinstance(new_priorities, np.ndarray)
    assert torch.equal(new_idxs.cpu(), idxs)
    assert actor == dqn.actor
    assert actor_target == dqn.actor_target
    assert_not_equal_state_dict(actor_pre_learn_sd, dqn.actor.state_dict())
    assert_not_equal_state_dict(
        actor_target_pre_learn_sd, dqn.actor_target.state_dict()
    )


# Updates target network parameters with soft update
def test_soft_update(vector_space, discrete_space):
    net_config = {"encoder_config": {"hidden_size": [64, 64]}}
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
        vector_space,
        discrete_space,
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
@pytest.mark.parametrize("num_envs", [1, 3])
@pytest.mark.parametrize("observation_space", ["vector_space", "image_space"])
def test_algorithm_test_loop(num_envs, observation_space, discrete_space, request):
    observation_space = request.getfixturevalue(observation_space)
    vect = num_envs > 1
    env = DummyEnv(state_size=observation_space.shape, vect=vect, num_envs=num_envs)
    agent = RainbowDQN(observation_space=observation_space, action_space=discrete_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
@pytest.mark.parametrize("observation_space", ["vector_space"])
def test_clone_returns_identical_agent(observation_space, discrete_space, request):
    observation_space = request.getfixturevalue(observation_space)
    dqn = DummyRainbowDQN(observation_space, discrete_space)
    dqn.fitness = [200, 200, 200]
    dqn.scores = [94, 94, 94]
    dqn.steps = [2500]
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
        clone_agent.actor_target.state_dict(), dqn.actor_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), dqn.optimizer.state_dict()
    )
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores
    assert clone_agent.tensor_attribute == dqn.tensor_attribute
    assert clone_agent.tensor_test == dqn.tensor_test

    accelerator = Accelerator()
    dqn = RainbowDQN(observation_space, discrete_space, accelerator=accelerator)
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
        clone_agent.actor_target.state_dict(), dqn.actor_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), dqn.optimizer.state_dict()
    )
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores

    accelerator = Accelerator()
    dqn = RainbowDQN(
        observation_space, discrete_space, accelerator=accelerator, wrap=False
    )
    clone_agent = dqn.clone(wrap=False)

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
        clone_agent.actor_target.state_dict(), dqn.actor_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), dqn.optimizer.state_dict()
    )
    assert clone_agent.fitness == dqn.fitness
    assert clone_agent.steps == dqn.steps
    assert clone_agent.scores == dqn.scores


def test_clone_new_index(vector_space, discrete_space):
    dqn = RainbowDQN(vector_space, discrete_space)
    clone_agent = dqn.clone(index=100)

    assert clone_agent.index == 100


def test_clone_after_learning(vector_space, discrete_space):
    batch_size = 8
    rainbow_dqn = RainbowDQN(vector_space, discrete_space, batch_size=batch_size)

    experiences = get_experiences_batch(vector_space, discrete_space, batch_size)
    rainbow_dqn.learn(experiences)
    clone_agent = rainbow_dqn.clone()

    assert clone_agent.observation_space == rainbow_dqn.observation_space
    assert clone_agent.action_space == rainbow_dqn.action_space
    # assert clone_agent.actor_network == rainbow_dqn.actor_network
    assert clone_agent.batch_size == rainbow_dqn.batch_size
    assert clone_agent.lr == rainbow_dqn.lr
    assert clone_agent.learn_step == rainbow_dqn.learn_step
    assert clone_agent.gamma == rainbow_dqn.gamma
    assert clone_agent.tau == rainbow_dqn.tau
    assert clone_agent.mut == rainbow_dqn.mut
    assert clone_agent.device == rainbow_dqn.device
    assert clone_agent.accelerator == rainbow_dqn.accelerator
    assert_state_dicts_equal(
        clone_agent.actor.state_dict(), rainbow_dqn.actor.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.actor_target.state_dict(), rainbow_dqn.actor_target.state_dict()
    )
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), rainbow_dqn.optimizer.state_dict()
    )
    assert clone_agent.fitness == rainbow_dqn.fitness
    assert clone_agent.steps == rainbow_dqn.steps
    assert clone_agent.scores == rainbow_dqn.scores
