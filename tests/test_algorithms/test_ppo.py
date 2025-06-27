import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces

from agilerl.algorithms.ppo import PPO
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import assert_not_equal_state_dict, assert_state_dicts_equal

# Cleanup fixture moved to conftest.py for better performance


class DummyPPO(PPO):
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


@pytest.fixture(scope="function")
def build_ppo(observation_space, action_space, accelerator, request):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)
    return PPO(observation_space, action_space, accelerator=accelerator)


# Initializes all necessary attributes with default values
@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        ("vector_space", EvolvableMLP),
        ("image_space", EvolvableCNN),
        ("dict_space", EvolvableMultiInput),
    ],
)
@pytest.mark.parametrize(
    "action_space",
    ["vector_space", "discrete_space", "multidiscrete_space", "multibinary_space"],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_initialize_ppo(
    observation_space, action_space, encoder_cls, accelerator, request
):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)
    ppo = PPO(observation_space, action_space, accelerator=accelerator)
    assert ppo.algo == "PPO"
    assert ppo.observation_space == observation_space
    assert ppo.action_space == action_space
    assert ppo.batch_size == 64
    assert ppo.lr == 1e-4
    assert ppo.gamma == 0.99
    assert ppo.gae_lambda == 0.95
    assert ppo.mut is None
    assert ppo.action_std_init == 0.0
    assert ppo.clip_coef == 0.2
    assert ppo.ent_coef == 0.01
    assert ppo.vf_coef == 0.5
    assert ppo.max_grad_norm == 0.5
    assert ppo.target_kl is None
    assert ppo.update_epochs == 4
    assert ppo.device == accelerator.device if accelerator else "cpu"
    assert ppo.accelerator == accelerator
    assert ppo.index == 0
    assert ppo.scores == []
    assert ppo.fitness == []
    assert ppo.steps == [0]
    assert isinstance(ppo.actor.encoder, encoder_cls)
    assert isinstance(ppo.critic.encoder, encoder_cls)
    expected_optimizer = AcceleratedOptimizer if accelerator else optim.Adam
    assert isinstance(ppo.optimizer.optimizer, expected_optimizer)


# Can initialize ppo with an actor network
# TODO: Will be deprecated in the future
@pytest.mark.parametrize(
    "obs_space, action_space, actor_network, critic_network, input_tensor, input_tensor_critic",
    [
        (
            "vector_space",
            "discrete_space",
            "simple_mlp",
            "simple_mlp_critic",
            torch.randn(1, 4),
            torch.randn(1, 6),
        ),
    ],
)
def test_initialize_ppo_with_make_evo(
    obs_space,
    action_space,
    actor_network,
    critic_network,
    input_tensor,
    input_tensor_critic,
    request,
):
    obs_space = request.getfixturevalue(obs_space)
    action_space = request.getfixturevalue(action_space)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_network = request.getfixturevalue(critic_network)
    critic_network = MakeEvolvable(critic_network, input_tensor_critic)

    ppo = PPO(
        obs_space,
        action_space,
        actor_network=actor_network,
        critic_network=critic_network,
    )

    assert ppo.observation_space == obs_space
    assert ppo.action_space == action_space
    assert ppo.batch_size == 64
    assert ppo.lr == 1e-4
    assert ppo.gamma == 0.99
    assert ppo.gae_lambda == 0.95
    assert ppo.mut is None
    assert ppo.action_std_init == 0.0
    assert ppo.clip_coef == 0.2
    assert ppo.ent_coef == 0.01
    assert ppo.vf_coef == 0.5
    assert ppo.max_grad_norm == 0.5
    assert ppo.target_kl is None
    assert ppo.update_epochs == 4
    assert ppo.device == "cpu"
    assert ppo.accelerator is None
    assert ppo.index == 0
    assert ppo.scores == []
    assert ppo.fitness == []
    assert ppo.steps == [0]
    assert isinstance(ppo.optimizer.optimizer, optim.Adam)


def test_initialize_ppo_with_incorrect_actor_net(vector_space, discrete_space):
    actor_network = "dummy"
    critic_network = "dummy"
    with pytest.raises(TypeError):
        ppo = PPO(
            vector_space,
            discrete_space,
            actor_network=actor_network,
            critic_network=critic_network,
        )
        assert ppo


# Can initialize ppo with an actor network but no critic - should trigger warning
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
def test_initialize_ppo_with_actor_network_no_critic(
    observation_space,
    discrete_space,
    actor_network,
    critic_network,
    input_tensor,
    input_tensor_critic,
    request,
):
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    observation_space = request.getfixturevalue(observation_space)
    with pytest.raises(TypeError):
        ppo = PPO(
            observation_space,
            discrete_space,
            actor_network=actor_network,
            critic_network=critic_network,
        )
        assert ppo


@pytest.mark.parametrize(
    "observation_space", ["vector_space", "image_space", "dict_space"]
)
@pytest.mark.parametrize(
    "action_space",
    ["vector_space", "discrete_space", "multidiscrete_space", "multibinary_space"],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
# Returns the expected action when given a state observation.
def test_returns_expected_action(observation_space, action_space, build_ppo, request):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)
    state = observation_space.sample()

    # First with grad=False
    action, action_logprob, dist_entropy, state_values = build_ppo.get_action(state)

    assert isinstance(action, np.ndarray)
    assert isinstance(action_logprob, np.ndarray)
    assert isinstance(dist_entropy, np.ndarray)
    assert isinstance(state_values, np.ndarray)

    if isinstance(action_space, spaces.Discrete):
        for act in action:
            assert act.is_integer()
            assert act >= 0 and act < action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        assert len(action[0]) == len(action_space.nvec)
        for i, act in enumerate(action[0]):
            assert act.is_integer()
            assert act >= 0 and act < action_space.nvec[i]
    elif isinstance(action_space, spaces.MultiBinary):
        assert len(action[0]) == action_space.n
        for act in action[0]:
            assert isinstance(act, np.float32)
    else:
        assert isinstance(action, np.ndarray)
        assert action.shape == (1, *action_space.shape)

    # Now with grad=True, and eval_action
    eval_action = torch.Tensor([[0, 1, 0, 1]]).to(build_ppo.device)
    action_logprob, dist_entropy, state_values = build_ppo.evaluate_actions(
        state, actions=eval_action
    )

    assert isinstance(action_logprob, torch.Tensor)
    assert isinstance(dist_entropy, torch.Tensor)
    assert isinstance(state_values, torch.Tensor)


def test_ppo_optimizer_parameters(vector_space, discrete_space):
    ppo = PPO(vector_space, discrete_space)

    # Store initial parameters
    initial_params = {
        name: param.clone() for name, param in ppo.actor.named_parameters()
    }

    # Perform a dummy optimization step
    dummy_input = torch.randn(1, 4)
    dummy_action = torch.tensor([0])
    dummy_log_prob = torch.tensor([1.0])

    _, _, _ = ppo.actor(dummy_input)
    loss = (dummy_log_prob - ppo.actor.action_log_prob(dummy_action)) ** 2
    loss = loss.mean()
    ppo.optimizer.zero_grad()
    loss.backward()
    ppo.optimizer.step()

    # Check if parameters have changed
    not_updated = []
    for name, param in ppo.actor.named_parameters():
        if torch.equal(initial_params[name], param):
            not_updated.append(name)

    assert not not_updated, f"The following parameters weren't updated:\n{not_updated}"


@pytest.mark.parametrize("observation_space", ["vector_space"])
@pytest.mark.parametrize("action_space", ["discrete_space"])
@pytest.mark.parametrize("accelerator", [None])
def test_returns_expected_action_mask_vectorized(
    build_ppo, observation_space, action_space, request
):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)
    state = np.stack([observation_space.sample(), observation_space.sample()])
    action_mask = np.stack([np.array([0, 1]), np.array([1, 0])])
    action, _, _, _ = build_ppo.get_action(state, action_mask=action_mask)
    assert np.array_equal(action, [1, 0]), action


@pytest.mark.parametrize(
    "observation_space", ["vector_space", "image_space", "dict_space"]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_learns_from_experiences(
    observation_space, discrete_space, accelerator, request
):
    batch_size = 45
    observation_space = request.getfixturevalue(observation_space)
    ppo = PPO(
        observation_space=observation_space,
        action_space=discrete_space,
        batch_size=batch_size,
        accelerator=accelerator,
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = ppo.actor
    actor_pre_learn_sd = copy.deepcopy(ppo.actor.state_dict())

    # Create batch size + 1 samples to ensure we can handle this
    num_steps = batch_size + 1

    # Create a batch of experiences
    if isinstance(observation_space, spaces.Discrete):
        states = torch.randint(0, observation_space.n, (num_steps,)).float()
        next_states = torch.randint(0, observation_space.n, (1,)).float()
    elif isinstance(observation_space, spaces.MultiDiscrete):
        states = torch.randint(0, observation_space.nvec, (num_steps,)).float()
        next_states = torch.randint(0, observation_space.nvec, (1,)).float()
    elif isinstance(observation_space, spaces.MultiBinary):
        states = torch.randint(0, 2, (num_steps,)).float()
        next_states = torch.randint(0, 2, (1,)).float()
    elif isinstance(observation_space, spaces.Box):
        states = torch.rand(num_steps, *observation_space.shape)
        next_states = torch.rand(1, *observation_space.shape)
    elif isinstance(observation_space, spaces.Dict):
        states = {
            key: torch.rand(num_steps, *space.shape)
            for key, space in observation_space.spaces.items()
        }
        next_states = {
            key: torch.rand(1, *space.shape)
            for key, space in observation_space.spaces.items()
        }
    elif isinstance(observation_space, spaces.Tuple):
        states = tuple(
            torch.rand(num_steps, *space.shape) for space in observation_space.spaces
        )
        next_states = tuple(
            torch.rand(1, *space.shape) for space in observation_space.spaces
        )

    # Create a batch of experiences
    actions = torch.randint(0, discrete_space.n, (num_steps,)).float()
    log_probs = torch.randn(num_steps)
    rewards = torch.randn(num_steps)
    dones = torch.randint(0, 2, (num_steps,))
    values = torch.randn(num_steps)
    next_done = torch.zeros(1)
    experiences = [
        [states],
        [actions],
        [log_probs],
        [rewards],
        [dones],
        [values],
        [next_states],
        [next_done],
    ]

    # Call the learn method
    loss = ppo.learn(experiences)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert actor == ppo.actor
    assert_not_equal_state_dict(actor_pre_learn_sd, ppo.actor.state_dict())


# Runs algorithm test loop
@pytest.mark.parametrize("observation_space", ["vector_space", "image_space"])
@pytest.mark.parametrize("num_envs", [1, 3])
def test_algorithm_test_loop(observation_space, discrete_space, num_envs, request):
    observation_space = request.getfixturevalue(observation_space)

    # Create a vectorised environment & test loop
    vect = num_envs > 1
    env = DummyEnv(state_size=observation_space.shape, vect=vect, num_envs=num_envs)
    agent = PPO(observation_space=observation_space, action_space=discrete_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
@pytest.mark.parametrize("observation_space", ["vector_space"])
def test_clone_returns_identical_agent(observation_space, discrete_space, request):
    observation_space = request.getfixturevalue(observation_space)

    ppo = DummyPPO(observation_space, discrete_space)
    ppo.fitness = [200, 200, 200]
    ppo.scores = [94, 94, 94]
    ppo.steps = [2500]
    ppo.tensor_attribute = torch.randn(1)
    clone_agent = ppo.clone()

    assert clone_agent.observation_space == ppo.observation_space
    assert clone_agent.action_space == ppo.action_space
    assert clone_agent.batch_size == ppo.batch_size
    assert clone_agent.lr == ppo.lr
    assert clone_agent.gamma == ppo.gamma
    assert clone_agent.gae_lambda == ppo.gae_lambda
    assert clone_agent.mut == ppo.mut
    assert clone_agent.action_std_init == ppo.action_std_init
    assert clone_agent.clip_coef == ppo.clip_coef
    assert clone_agent.ent_coef == ppo.ent_coef
    assert clone_agent.vf_coef == ppo.vf_coef
    assert clone_agent.max_grad_norm == ppo.max_grad_norm
    assert clone_agent.target_kl == ppo.target_kl
    assert clone_agent.update_epochs == ppo.update_epochs
    assert clone_agent.device == ppo.device
    assert clone_agent.accelerator == ppo.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ppo.actor.state_dict())
    assert_state_dicts_equal(clone_agent.critic.state_dict(), ppo.critic.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), ppo.optimizer.state_dict()
    )
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores
    assert clone_agent.tensor_attribute == ppo.tensor_attribute
    assert clone_agent.tensor_test == ppo.tensor_test

    accelerator = Accelerator()
    ppo = PPO(observation_space, discrete_space, accelerator=accelerator)
    clone_agent = ppo.clone()

    assert clone_agent.observation_space == ppo.observation_space
    assert clone_agent.action_space == ppo.action_space
    assert clone_agent.batch_size == ppo.batch_size
    assert clone_agent.lr == ppo.lr
    assert clone_agent.gamma == ppo.gamma
    assert clone_agent.gae_lambda == ppo.gae_lambda
    assert clone_agent.mut == ppo.mut
    assert clone_agent.action_std_init == ppo.action_std_init
    assert clone_agent.clip_coef == ppo.clip_coef
    assert clone_agent.ent_coef == ppo.ent_coef
    assert clone_agent.vf_coef == ppo.vf_coef
    assert clone_agent.max_grad_norm == ppo.max_grad_norm
    assert clone_agent.target_kl == ppo.target_kl
    assert clone_agent.update_epochs == ppo.update_epochs
    assert clone_agent.device == ppo.device
    assert clone_agent.accelerator == ppo.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ppo.actor.state_dict())
    assert_state_dicts_equal(clone_agent.critic.state_dict(), ppo.critic.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), ppo.optimizer.state_dict()
    )
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores

    accelerator = Accelerator()
    ppo = PPO(
        observation_space,
        discrete_space,
        accelerator=accelerator,
        wrap=False,
    )
    clone_agent = ppo.clone(wrap=False)

    assert clone_agent.observation_space == ppo.observation_space
    assert clone_agent.action_space == ppo.action_space
    assert clone_agent.batch_size == ppo.batch_size
    assert clone_agent.lr == ppo.lr
    assert clone_agent.gamma == ppo.gamma
    assert clone_agent.gae_lambda == ppo.gae_lambda
    assert clone_agent.mut == ppo.mut
    assert clone_agent.action_std_init == ppo.action_std_init
    assert clone_agent.clip_coef == ppo.clip_coef
    assert clone_agent.ent_coef == ppo.ent_coef
    assert clone_agent.vf_coef == ppo.vf_coef
    assert clone_agent.max_grad_norm == ppo.max_grad_norm
    assert clone_agent.target_kl == ppo.target_kl
    assert clone_agent.update_epochs == ppo.update_epochs
    assert clone_agent.device == ppo.device
    assert clone_agent.accelerator == ppo.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ppo.actor.state_dict())
    assert_state_dicts_equal(clone_agent.critic.state_dict(), ppo.critic.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), ppo.optimizer.state_dict()
    )
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores


def test_clone_new_index(vector_space, discrete_space):
    ppo = PPO(vector_space, discrete_space)
    clone_agent = ppo.clone(index=100)
    assert clone_agent.index == 100


def test_clone_after_learning(vector_space):
    max_env_steps = 20
    num_vec_envs = 2
    ppo = PPO(vector_space, copy.deepcopy(vector_space))

    states = np.random.randn(max_env_steps, num_vec_envs, vector_space.shape[0])
    next_states = np.random.randn(num_vec_envs, vector_space.shape[0])
    actions = np.random.rand(max_env_steps, num_vec_envs, vector_space.shape[0])
    log_probs = -np.random.rand(max_env_steps, num_vec_envs)
    rewards = np.random.randint(0, 100, (max_env_steps, num_vec_envs))
    dones = np.zeros((max_env_steps, num_vec_envs))
    values = np.random.randn(max_env_steps, num_vec_envs)
    next_done = np.zeros((1, num_vec_envs))
    experiences = (
        states,
        actions,
        log_probs,
        rewards,
        dones,
        values,
        next_states,
        next_done,
    )
    ppo.learn(experiences)
    clone_agent = ppo.clone()
    assert clone_agent.observation_space == ppo.observation_space
    assert clone_agent.action_space == ppo.action_space
    assert clone_agent.batch_size == ppo.batch_size
    assert clone_agent.lr == ppo.lr
    assert clone_agent.gamma == ppo.gamma
    assert clone_agent.gae_lambda == ppo.gae_lambda
    assert clone_agent.mut == ppo.mut
    assert clone_agent.action_std_init == ppo.action_std_init
    assert clone_agent.clip_coef == ppo.clip_coef
    assert clone_agent.ent_coef == ppo.ent_coef
    assert clone_agent.vf_coef == ppo.vf_coef
    assert clone_agent.max_grad_norm == ppo.max_grad_norm
    assert clone_agent.target_kl == ppo.target_kl
    assert clone_agent.update_epochs == ppo.update_epochs
    assert clone_agent.device == ppo.device
    assert clone_agent.accelerator == ppo.accelerator
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ppo.actor.state_dict())
    assert_state_dicts_equal(clone_agent.critic.state_dict(), ppo.critic.state_dict())
    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), ppo.optimizer.state_dict()
    )
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores
