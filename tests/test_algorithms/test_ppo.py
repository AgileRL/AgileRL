import copy
from pathlib import Path

import dill
import gymnasium
from gymnasium.vector import SyncVectorEnv
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
from tests.helper_functions import (
    generate_dict_or_tuple_space,
    generate_discrete_space,
    generate_multidiscrete_space,
    generate_random_box_space,
)


@pytest.fixture(autouse=True)
def cleanup():
    yield  # Run the test first
    torch.cuda.empty_cache()  # Free up GPU memory


class DummyPPO(PPO):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super().__init__(observation_space, action_space, *args, **kwargs)

        self.tensor_test = torch.randn(1)


class DummyEnv:
    def __init__(self, state_size, vect=True, num_envs=2):
        self.state_size = state_size
        self.vect = vect
        self.num_envs = num_envs
        if self.vect:
            self.state_size = (num_envs,) + self.state_size
            self.n_envs = num_envs
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


@pytest.fixture
def vector_space():
    return generate_random_box_space(shape=(4,), low=0, high=1)


@pytest.fixture
def image_space():
    return generate_random_box_space(shape=(3, 32, 32), low=0, high=255)


@pytest.fixture
def action_space():
    return generate_random_box_space(shape=(2,), low=0, high=1)


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
@pytest.mark.parametrize(
    "action_space",
    [
        generate_random_box_space(shape=(2,), low=-1, high=1),
        generate_discrete_space(2),
        generate_multidiscrete_space(2, 3),
        spaces.MultiBinary(2),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_initialize_ppo(observation_space, action_space, encoder_cls, accelerator):
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
            "action_space",
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
    action_space = generate_discrete_space(2)
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
    assert ppo.num_envs == 1


def test_initialize_ppo_with_incorrect_actor_net():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)
    actor_network = "dummy"
    critic_network = "dummy"
    with pytest.raises(TypeError):
        ppo = PPO(
            observation_space,
            action_space,
            actor_network=actor_network,
            critic_network=critic_network,
        )
        assert ppo


# Can initialize ppo with an actor network but no critic - should trigger warning
@pytest.mark.parametrize(
    "observation_space, actor_network, critic_network, input_tensor, input_tensor_critic",
    [
        (
            generate_random_box_space(shape=(4,), low=0, high=1),
            "simple_mlp",
            "simple_mlp_critic",
            torch.randn(1, 4),
            torch.randn(1, 6),
        ),
    ],
)
def test_initialize_ppo_with_actor_network_no_critic(
    observation_space,
    actor_network,
    critic_network,
    input_tensor,
    input_tensor_critic,
    request,
):
    action_space = generate_discrete_space(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    with pytest.raises(TypeError):
        ppo = PPO(
            observation_space,
            action_space,
            actor_network=actor_network,
            critic_network=critic_network,
        )
        assert ppo


@pytest.fixture
def build_ppo(observation_space, action_space, accelerator):
    ppo = PPO(observation_space, action_space, accelerator=accelerator)
    yield ppo
    del ppo


@pytest.mark.parametrize(
    "observation_space",
    [
        generate_discrete_space(4),
        generate_random_box_space(shape=(4,), low=0, high=1),
        generate_random_box_space(shape=(3, 32, 32), low=0, high=1),
        generate_dict_or_tuple_space(2, 3, dict_space=False),
        generate_dict_or_tuple_space(2, 3, dict_space=True),
    ],
)
@pytest.mark.parametrize(
    "action_space",
    [
        generate_random_box_space(shape=(2,), low=0, high=1),
        generate_discrete_space(2),
        spaces.MultiDiscrete([2, 3]),
        spaces.MultiBinary(2),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
# Returns the expected action when given a state observation.
def test_returns_expected_action(observation_space, action_space, build_ppo):
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
        print(action, action_space)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1, *action_space.shape)

    # Now with grad=True, and eval_action
    eval_action = torch.Tensor([[0, 1]]).to(build_ppo.device)
    action_logprob, dist_entropy, state_values = build_ppo.evaluate_actions(
        state, actions=eval_action
    )

    assert isinstance(action_logprob, torch.Tensor)
    assert isinstance(dist_entropy, torch.Tensor)
    assert isinstance(state_values, torch.Tensor)


def test_ppo_optimizer_parameters():
    observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(2)
    ppo = PPO(observation_space, action_space)

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


@pytest.mark.parametrize(
    "observation_space", [generate_random_box_space(shape=(4,), low=0, high=1)]
)
@pytest.mark.parametrize("action_space", [generate_discrete_space(2)])
@pytest.mark.parametrize("accelerator", [None])
def test_returns_expected_action_mask_vectorized(build_ppo):
    state = np.array([[1, 2, 4, 5], [2, 3, 5, 1]])
    action_mask = np.array([[0, 1], [1, 0]])
    action, _, _, _ = build_ppo.get_action(state, action_mask=action_mask)
    assert np.array_equal(action, [1, 0]), action


@pytest.mark.parametrize(
    "observation_space",
    [
        generate_random_box_space(shape=(4,), low=0, high=1),
        generate_random_box_space(shape=(3, 32, 32), low=0, high=1),
        generate_discrete_space(4),
        generate_dict_or_tuple_space(2, 3, dict_space=False),
        generate_dict_or_tuple_space(2, 3, dict_space=True),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_learns_from_experiences(observation_space, accelerator):
    batch_size = 45
    action_space = spaces.Discrete(2)
    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        batch_size=batch_size,
        accelerator=accelerator,
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = ppo.actor
    actor_pre_learn_sd = str(copy.deepcopy(ppo.actor.state_dict()))

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
    actions = torch.randint(0, action_space.n, (num_steps,)).float()
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
    assert actor_pre_learn_sd != str(ppo.actor.state_dict())


# Runs algorithm test loop
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_random_box_space(shape=(4,), low=0, high=1),
        generate_random_box_space(shape=(3, 32, 32), low=0, high=1),
    ],
)
@pytest.mark.parametrize("num_envs", [1, 3])
def test_algorithm_test_loop(observation_space, num_envs):
    action_space = generate_discrete_space(2)

    # Create a vectorised environment & test loop
    vect = num_envs > 1
    env = DummyEnv(state_size=observation_space.shape, vect=vect, num_envs=num_envs)
    agent = PPO(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
@pytest.mark.parametrize(
    "observation_space",
    [
        generate_random_box_space(shape=(4,), low=0, high=1),
        generate_random_box_space(shape=(3, 32, 32), low=0, high=1),
        generate_dict_or_tuple_space(2, 3, dict_space=False),
        generate_dict_or_tuple_space(2, 3, dict_space=True),
    ],
)
def test_clone_returns_identical_agent(observation_space):
    action_space = generate_discrete_space(2)

    ppo = DummyPPO(observation_space, action_space)
    ppo.num_envs = 1
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
    assert str(clone_agent.actor.state_dict()) == str(ppo.actor.state_dict())
    assert str(clone_agent.critic.state_dict()) == str(ppo.critic.state_dict())
    assert str(clone_agent.optimizer.state_dict()) == str(ppo.optimizer.state_dict())
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores
    assert clone_agent.tensor_attribute == ppo.tensor_attribute
    assert clone_agent.tensor_test == ppo.tensor_test
    assert clone_agent.num_envs == ppo.num_envs
    assert clone_agent.index == ppo.index

    accelerator = Accelerator()
    ppo = PPO(observation_space, action_space, accelerator=accelerator)
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
    assert str(clone_agent.actor.state_dict()) == str(ppo.actor.state_dict())
    assert str(clone_agent.critic.state_dict()) == str(ppo.critic.state_dict())
    assert str(clone_agent.optimizer.state_dict()) == str(ppo.optimizer.state_dict())
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores
    assert clone_agent.num_envs == ppo.num_envs
    assert clone_agent.index == ppo.index

    accelerator = Accelerator()
    ppo = PPO(
        observation_space,
        action_space,
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
    assert str(clone_agent.actor.state_dict()) == str(ppo.actor.state_dict())
    assert str(clone_agent.critic.state_dict()) == str(ppo.critic.state_dict())
    assert str(clone_agent.optimizer.state_dict()) == str(ppo.optimizer.state_dict())
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores
    assert clone_agent.num_envs == ppo.num_envs
    assert clone_agent.index == ppo.index


def test_clone_new_index():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    ppo = PPO(observation_space, action_space)
    clone_agent = ppo.clone(index=100)

    assert clone_agent.index == 100


@pytest.mark.parametrize("device", ["cpu", "cuda"], ids=lambda d: f"device={d}")
@pytest.mark.parametrize(
    "use_rollout_buffer", [True, False], ids=lambda b: f"use_rollout_buffer={b}"
)
@pytest.mark.parametrize("recurrent", [True, False], ids=lambda r: f"recurrent={r}")
@pytest.mark.parametrize(
    "share_encoders", [True, False], ids=lambda s: f"share_encoders={s}"
)
def test_clone_after_learning(device, use_rollout_buffer, recurrent, share_encoders):

    # accept if recurrent and no rollout buffer
    if recurrent and not use_rollout_buffer:
        return

    # check if device is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    max_env_steps = 20
    num_vec_envs = 2

    if recurrent:
        net_config = {
            "encoder_config": {
                "hidden_state_size": 64,
                "max_seq_len": 10,
            }
        }
    else:
        net_config = {}

    ppo = PPO(
        observation_space,
        action_space,
        device=torch.device(device),
        use_rollout_buffer=use_rollout_buffer,
        recurrent=recurrent,
        net_config=net_config,
        num_envs=num_vec_envs,
        share_encoders=share_encoders,
    )

    if use_rollout_buffer:
        dummy_env = DummyEnv(observation_space.shape, vect=True, num_envs=num_vec_envs)
        ppo.collect_rollouts(dummy_env)
        ppo.learn()
    else:
        states = np.random.randn(
            max_env_steps, num_vec_envs, observation_space.shape[0]
        )
        next_states = np.random.randn(num_vec_envs, observation_space.shape[0])
        actions = np.random.rand(max_env_steps, num_vec_envs, action_space.shape[0])
        log_probs = -np.random.rand(max_env_steps, num_vec_envs)
        rewards = np.random.randint(0, 100, (max_env_steps, num_vec_envs))
        dones = np.zeros((max_env_steps, num_vec_envs))
        values = np.random.randn(
            max_env_steps,
            num_vec_envs,
        )
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
    assert str(clone_agent.actor.state_dict()) == str(ppo.actor.state_dict())

    if share_encoders and recurrent:
        # the critic might be different if share_encoders is True
        # (the encoder might be different because of the logic in the share_encoder_parameters)
        # The important thing is that the head_net is the same as the encoder is neither ran during
        # the forward of the exploration, nor the learning step.

        # !IMPORTANT: I'm having trouble understanding why the critic is different here.
        # ! I'm not sure if this is an issue or not. the agent is learning, so it's not like
        # ! the encoder is not being used. we might need to look into this more.
        # ! it could be because of something inside the LSTM, or possibly the optimizer
        # ! however, the parameters are being detached from the encoder.
        # !IMPORTANT
        assert str(clone_agent.critic.head_net.state_dict()) == str(
            ppo.critic.head_net.state_dict()
        )
    else:
        assert str(clone_agent.critic.state_dict()) == str(ppo.critic.state_dict())

    assert str(clone_agent.optimizer.state_dict()) == str(ppo.optimizer.state_dict())
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores
    assert clone_agent.num_envs == ppo.num_envs
    assert clone_agent.index == ppo.index


# The saved checkpoint file contains the correct data and format.
@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        (generate_random_box_space(shape=(4,), low=0, high=1), EvolvableMLP),
        (generate_random_box_space(shape=(3, 32, 32), low=0, high=1), EvolvableCNN),
        (generate_dict_or_tuple_space(2, 3, dict_space=False), EvolvableMultiInput),
        (generate_dict_or_tuple_space(2, 3, dict_space=True), EvolvableMultiInput),
    ],
)
def test_save_load_checkpoint_correct_data_and_format(
    observation_space, encoder_cls, tmpdir
):
    # Initialize the ppo agent
    ppo = PPO(
        observation_space=observation_space,
        action_space=generate_random_box_space(shape=(2,)),
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ppo.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_state_dict" in checkpoint["network_info"]["modules"]
    assert "optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "gamma" in checkpoint
    assert "gae_lambda" in checkpoint
    assert "mut" in checkpoint
    assert "action_std_init" in checkpoint
    assert "clip_coef" in checkpoint
    assert "ent_coef" in checkpoint
    assert "vf_coef" in checkpoint
    assert "max_grad_norm" in checkpoint
    assert "target_kl" in checkpoint
    assert "update_epochs" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint
    assert "num_envs" in checkpoint

    ppo = PPO(
        observation_space=observation_space,
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
    )
    # Load checkpoint
    ppo.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(ppo.actor.encoder, encoder_cls)
    assert isinstance(ppo.critic.encoder, encoder_cls)
    assert ppo.lr == 1e-4
    assert ppo.num_envs == 1
    assert ppo.batch_size == 64
    assert ppo.gamma == 0.99
    assert ppo.mut is None
    assert ppo.action_std_init == 0.0
    assert ppo.clip_coef == 0.2
    assert ppo.ent_coef == 0.01
    assert ppo.vf_coef == 0.5
    assert ppo.max_grad_norm == 0.5
    assert ppo.target_kl is None
    assert ppo.update_epochs == 4
    assert ppo.index == 0
    assert ppo.scores == []
    assert ppo.fitness == []
    assert ppo.steps == [0]


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
    observation_space = generate_random_box_space(
        shape=input_tensor.shape[1:], low=0, high=1
    )
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)

    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    critic_network = SimpleCNN()
    critic_network = MakeEvolvable(
        critic_network,
        input_tensor,
        torch.randn(1, action_space.shape[0]),
    )

    # Initialize the ppo agent
    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        actor_network=actor_network,
        critic_network=critic_network,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ppo.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint["network_info"]["modules"]
    assert "actor_state_dict" in checkpoint["network_info"]["modules"]
    assert "critic_init_dict" in checkpoint["network_info"]["modules"]
    assert "critic_state_dict" in checkpoint["network_info"]["modules"]
    assert "optimizer_state_dict" in checkpoint["network_info"]["optimizers"]
    assert "batch_size" in checkpoint
    assert "lr" in checkpoint
    assert "gamma" in checkpoint
    assert "gae_lambda" in checkpoint
    assert "mut" in checkpoint
    assert "action_std_init" in checkpoint
    assert "clip_coef" in checkpoint
    assert "ent_coef" in checkpoint
    assert "vf_coef" in checkpoint
    assert "max_grad_norm" in checkpoint
    assert "target_kl" in checkpoint
    assert "update_epochs" in checkpoint
    assert "mut" in checkpoint
    assert "index" in checkpoint
    assert "scores" in checkpoint
    assert "fitness" in checkpoint
    assert "steps" in checkpoint
    assert "num_envs" in checkpoint

    ppo = PPO(
        observation_space=generate_random_box_space(shape=(4,), low=0, high=1),
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
        actor_network=actor_network,
        critic_network=critic_network,
    )
    # Load checkpoint
    ppo.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(ppo.actor, nn.Module)
    assert isinstance(ppo.critic, nn.Module)
    assert ppo.lr == 1e-4
    assert ppo.num_envs == 1
    assert ppo.batch_size == 64
    assert ppo.gamma == 0.99
    assert ppo.mut is None
    assert ppo.action_std_init == 0.0
    assert ppo.clip_coef == 0.2
    assert ppo.ent_coef == 0.01
    assert ppo.vf_coef == 0.5
    assert ppo.max_grad_norm == 0.5
    assert ppo.target_kl is None
    assert ppo.update_epochs == 4
    assert ppo.index == 0
    assert ppo.scores == []
    assert ppo.fitness == []
    assert ppo.steps == [0]


# The saved checkpoint file contains the correct data and format.]
@pytest.mark.parametrize(
    "observation_space, encoder_cls",
    [
        (generate_random_box_space(shape=(4,), low=0, high=1), EvolvableMLP),
        (generate_random_box_space(shape=(3, 32, 32), low=0, high=1), EvolvableCNN),
        (generate_dict_or_tuple_space(2, 3, dict_space=False), EvolvableMultiInput),
        (generate_dict_or_tuple_space(2, 3, dict_space=True), EvolvableMultiInput),
    ],
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
def test_load_from_pretrained(observation_space, encoder_cls, accelerator, tmpdir):
    # Initialize the ppo agent
    ppo = PPO(
        observation_space=observation_space,
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
        accelerator=accelerator,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ppo.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_ppo = PPO.load(checkpoint_path, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_ppo.observation_space == ppo.observation_space
    assert new_ppo.action_space == ppo.action_space
    assert new_ppo.discrete_actions == ppo.discrete_actions
    assert isinstance(new_ppo.actor.encoder, encoder_cls)
    assert isinstance(new_ppo.critic.encoder, encoder_cls)
    assert new_ppo.lr == ppo.lr
    assert str(new_ppo.actor.state_dict()) == str(ppo.actor.state_dict())
    assert str(new_ppo.critic.state_dict()) == str(ppo.critic.state_dict())
    assert new_ppo.batch_size == ppo.batch_size
    assert new_ppo.gamma == ppo.gamma
    assert new_ppo.mut == ppo.mut
    assert new_ppo.index == ppo.index
    assert new_ppo.scores == ppo.scores
    assert new_ppo.fitness == ppo.fitness
    assert new_ppo.steps == ppo.steps
    assert new_ppo.num_envs == ppo.num_envs


# TODO: This will be deprecated in the future
@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (
            generate_random_box_space(shape=(4,), low=0, high=1),
            "simple_mlp",
            torch.randn(1, 4),
        ),
        (
            generate_random_box_space(shape=(3, 64, 64), low=0, high=1),
            "simple_cnn",
            torch.randn(1, 3, 64, 64),
        ),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_networks(
    observation_space, actor_network, input_tensor, request, tmpdir
):
    action_space = spaces.Discrete(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)

    # Initialize the ppo agent
    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        actor_network=actor_network,
        critic_network=copy.deepcopy(actor_network),
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ppo.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_ppo = PPO.load(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert new_ppo.observation_space == ppo.observation_space
    assert new_ppo.action_space == ppo.action_space
    assert new_ppo.discrete_actions == ppo.discrete_actions
    assert isinstance(new_ppo.actor, nn.Module)
    assert isinstance(new_ppo.critic, nn.Module)
    assert new_ppo.lr == ppo.lr
    assert str(new_ppo.actor.to("cpu").state_dict()) == str(ppo.actor.state_dict())
    assert str(new_ppo.critic.to("cpu").state_dict()) == str(ppo.critic.state_dict())
    assert new_ppo.batch_size == ppo.batch_size
    assert new_ppo.gamma == ppo.gamma
    assert new_ppo.mut == ppo.mut
    assert new_ppo.index == ppo.index
    assert new_ppo.scores == ppo.scores
    assert new_ppo.fitness == ppo.fitness
    assert new_ppo.steps == ppo.steps
    assert new_ppo.num_envs == ppo.num_envs


# Test the RolloutBuffer implementation
def test_rollout_buffer_initialization():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    buffer = RolloutBuffer(
        capacity=100,
        num_envs=1,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
    )

    assert buffer.capacity == 100
    assert buffer.observation_space == observation_space
    assert buffer.action_space == action_space
    assert buffer.gamma == 0.99
    assert buffer.gae_lambda == 0.95
    assert buffer.recurrent is False
    assert buffer.hidden_state_architecture is None
    assert buffer.device == "cpu"
    assert buffer.pos == 0
    assert buffer.full is False

    # Test with hidden states
    buffer = RolloutBuffer(
        capacity=100,
        num_envs=1,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        recurrent=False,
    )

    assert buffer.recurrent is False
    assert buffer.hidden_state_architecture is None
    assert buffer.hidden_states is None
    assert buffer.next_hidden_states is None


# Test the RolloutBuffer implementation
def test_rollout_buffer_initialization_recurrent():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    buffer = RolloutBuffer(
        capacity=100,
        num_envs=1,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
    )

    assert buffer.capacity == 100
    assert buffer.observation_space == observation_space
    assert buffer.action_space == action_space
    assert buffer.gamma == 0.99
    assert buffer.gae_lambda == 0.95
    assert buffer.recurrent is False
    assert buffer.hidden_state_architecture is None
    assert buffer.device == "cpu"
    assert buffer.pos == 0
    assert buffer.full is False

    # Test with hidden states
    buffer = RolloutBuffer(
        capacity=100,
        num_envs=1,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        recurrent=True,
        # (num_layers * directions, num_envs, hidden_size)
        hidden_state_architecture={
            "shared_encoder_h": (1, 1, 64),
            "shared_encoder_c": (1, 1, 64),
        },
    )

    assert buffer.recurrent is True
    assert buffer.hidden_state_architecture is not None
    assert buffer.hidden_states is not None
    assert buffer.next_hidden_states is not None

    # Test with hidden states
    buffer = RolloutBuffer(
        capacity=100,
        num_envs=8,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        recurrent=True,
        # (num_layers * directions, num_envs, hidden_size)
        hidden_state_architecture={
            "shared_encoder_h": (1, 8, 64),
            "shared_encoder_c": (1, 8, 64),
        },
    )

    assert buffer.recurrent is True
    assert buffer.hidden_state_architecture is not None
    assert buffer.hidden_states is not None
    assert buffer.next_hidden_states is not None


# Test adding samples to the buffer
def test_rollout_buffer_add():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    buffer = RolloutBuffer(
        capacity=100,
        num_envs=1,
        observation_space=observation_space,
        action_space=action_space,
    )

    # Add a single sample
    obs = np.random.rand(*observation_space.shape).astype(observation_space.dtype)
    action = np.array([1])
    reward = 1.0
    done = False
    value = 0.5
    log_prob = -0.5
    next_obs = np.random.rand(*observation_space.shape).astype(observation_space.dtype)

    buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    assert buffer.pos == 1
    assert not buffer.full
    # where X is the environment index, and Y is the position in the buffer
    assert np.array_equal(buffer.observations[0][0], obs)
    assert np.array_equal(
        buffer.actions[0], action
    )  # Check the action for the first env at the first timestep
    assert buffer.rewards[0][0] == reward
    assert buffer.dones[0][0] == done
    assert buffer.values[0][0] == value
    assert buffer.log_probs[0][0] == log_prob
    assert np.array_equal(buffer.next_observations[0][0], next_obs)

    # Add samples until buffer is full
    for i in range(99):
        buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    assert buffer.pos == 100  # Wrapped around
    assert buffer.full is True

    buffer.reset()

    assert buffer.pos == 0
    assert buffer.full is False


# Test computing returns and advantages
def test_rollout_buffer_compute_returns_and_advantages():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    buffer = RolloutBuffer(
        capacity=5,
        num_envs=1,
        observation_space=observation_space,
        action_space=action_space,
        gamma=0.99,
        gae_lambda=0.95,
    )

    # Add samples
    for i in range(5):
        obs = np.random.rand(*observation_space.shape)
        action = np.array([1])
        reward = 1.0
        done = i == 4  # Last step is done
        value = 0.5
        log_prob = -0.5
        next_obs = np.random.rand(*observation_space.shape)

        buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    # Compute returns and advantages
    last_value = 0.0
    last_done = np.zeros(1)
    buffer.compute_returns_and_advantages(
        last_value, last_done
    )  # Already done by the add method at the end of the collection

    # Check that returns and advantages are computed
    assert not np.array_equal(buffer.returns[:, 0], np.zeros(5))
    assert not np.array_equal(buffer.advantages[:, 0], np.zeros(5))

    # Check that returns are higher for earlier steps (due to discounting)
    assert buffer.returns[0, 0] > buffer.returns[4, 0]


# Test getting batch from buffer
def test_rollout_buffer_get_batch():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    buffer = RolloutBuffer(
        capacity=100,
        num_envs=1,
        observation_space=observation_space,
        action_space=action_space,
    )

    # Add samples
    for i in range(10):
        obs = np.random.rand(*observation_space.shape)
        action = np.array([1])
        reward = 1.0
        done = i == 9  # Last step is done
        value = 0.5
        log_prob = -0.5
        next_obs = np.random.rand(*observation_space.shape)

        buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    # Compute returns and advantages
    last_value = np.zeros(1)
    last_done = np.zeros(1)
    buffer.compute_returns_and_advantages(
        last_value, last_done
    )  # Already done by the add method at the end of the collection

    # Get all data
    batch = buffer.get()

    assert len(batch["observations"]) == 10
    assert len(batch["actions"]) == 10
    assert len(batch["rewards"]) == 10
    assert len(batch["dones"]) == 10
    assert len(batch["values"]) == 10
    assert len(batch["log_probs"]) == 10
    assert len(batch["advantages"]) == 10
    assert len(batch["returns"]) == 10

    # Get batch of specific size
    batch = buffer.get(batch_size=5)

    assert len(batch["observations"]) == 5
    assert len(batch["actions"]) == 5

    # Get tensor batch
    tensor_batch = buffer.get_tensor_batch(batch_size=5)

    assert isinstance(tensor_batch["observations"], torch.Tensor)
    assert isinstance(tensor_batch["actions"], torch.Tensor)
    assert isinstance(tensor_batch["advantages"], torch.Tensor)
    assert tensor_batch["observations"].shape[0] == 5


# Test PPO initialization with rollout buffer
def test_ppo_with_rollout_buffer():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        learn_step=100,
    )

    assert ppo.use_rollout_buffer
    assert hasattr(ppo, "rollout_buffer")
    assert isinstance(ppo.rollout_buffer, RolloutBuffer)
    assert ppo.rollout_buffer.capacity == ppo.learn_step
    assert not ppo.rollout_buffer.recurrent

    # Test with hidden states
    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        recurrent=True,
        use_rollout_buffer=True,
        net_config={
            "encoder_config": {
                "hidden_state_size": 64,
                "max_seq_len": 10,
            }
        },
    )

    assert ppo.recurrent
    assert ppo.rollout_buffer.hidden_state_architecture == {
        "shared_encoder_h": (1, 1, 64),
        "shared_encoder_c": (1, 1, 64),
    }
    assert ppo.rollout_buffer.recurrent

    # Test with hidden states
    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        recurrent=True,
        use_rollout_buffer=True,
        share_encoders=False,
        net_config={
            "encoder_config": {
                "hidden_state_size": 64,
                "max_seq_len": 10,
            }
        },
    )

    assert ppo.recurrent
    assert ppo.rollout_buffer.hidden_state_architecture == {
        "actor_encoder_h": (1, 1, 64),
        "actor_encoder_c": (1, 1, 64),
        "critic_encoder_h": (1, 1, 64),
        "critic_encoder_c": (1, 1, 64),
    }
    assert ppo.rollout_buffer.recurrent
    assert not ppo.share_encoders


# Test PPO learning with rollout buffer
def test_ppo_learn_with_rollout_buffer():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)
    batch_size = 32

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        learn_step=64,
        batch_size=batch_size,
    )

    # Fill the buffer manually
    for i in range(64):
        obs = np.random.rand(*observation_space.shape)
        action = np.array([1])
        reward = 1.0
        done = i == 63  # Last step is done
        value = 0.5
        log_prob = -0.5
        next_obs = np.random.rand(*observation_space.shape)

        ppo.rollout_buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    # Compute returns and advantages
    last_value = np.zeros(1)
    last_done = np.zeros(1)
    ppo.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

    # Learn from rollout buffer
    loss = ppo.learn()

    assert isinstance(loss, float)
    assert loss >= 0.0


# Test PPO with hidden states
def test_ppo_with_hidden_states():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        recurrent=True,
        net_config={
            "encoder_config": {
                "hidden_state_size": 64,
                "max_seq_len": 10,
            }
        },
    )

    # Get action with hidden state
    obs = np.random.rand(*observation_space.shape)
    hidden_state = ppo.get_initial_hidden_state()

    action, log_prob, entropy, value, next_hidden = ppo.get_action(
        obs, hidden_state=hidden_state
    )

    assert action.shape[0] == 1
    assert isinstance(log_prob, np.ndarray)
    assert isinstance(entropy, np.ndarray)
    assert isinstance(value, np.ndarray)
    assert next_hidden is not None
    assert next_hidden.get("shared_encoder_h", None).shape == (1, 1, 64)
    assert next_hidden.get("shared_encoder_c", None).shape == (1, 1, 64)


# Test PPO with hidden states
def test_ppo_with_hidden_states_multiple_obs():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        recurrent=True,
        num_envs=2,
        net_config={
            "encoder_config": {
                "hidden_state_size": 64,
                "max_seq_len": 10,
            }
        },
    )

    # Get action with hidden state (multiple observations)
    obs = np.zeros((2, *observation_space.shape))
    hidden_state = ppo.get_initial_hidden_state(num_envs=2)

    action, log_prob, entropy, value, next_hidden = ppo.get_action(
        obs, hidden_state=hidden_state
    )

    assert action.shape[0] == 2
    assert isinstance(log_prob, np.ndarray)
    assert isinstance(entropy, np.ndarray)
    assert isinstance(value, np.ndarray)
    assert next_hidden is not None
    assert next_hidden.get("shared_encoder_h", None).shape == (1, 2, 64)
    assert next_hidden.get("shared_encoder_c", None).shape == (1, 2, 64)


# Test PPO with hidden states
def test_ppo_with_hidden_states_multiple_envs():
    num_envs = 2
    env = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make("CartPole-v1")] * num_envs
    )

    observation_space = env.single_observation_space  # Use single env space
    action_space = env.single_action_space  # Use single env space

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        recurrent=True,
        num_envs=num_envs,
        net_config={
            "encoder_config": {
                "hidden_state_size": 64,
                "max_seq_len": 10,
            }
        },
    )

    # Get action with hidden state (multiple observations)
    obs, _ = env.reset()
    hidden_state = ppo.get_initial_hidden_state(num_envs=num_envs)

    action, log_prob, entropy, value, next_hidden = ppo.get_action(
        obs, hidden_state=hidden_state
    )

    assert action.shape[0] == 2
    assert isinstance(log_prob, np.ndarray)
    assert isinstance(entropy, np.ndarray)
    assert isinstance(value, np.ndarray)
    assert next_hidden is not None
    assert next_hidden.get("shared_encoder_h", None).shape == (1, 2, 64)
    assert next_hidden.get("shared_encoder_c", None).shape == (1, 2, 64)


# Test PPO with hidden states and collect_rollouts
def test_ppo_with_hidden_states_multiple_envs_collect_rollouts():
    num_envs = 2
    env = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make("CartPole-v1")] * num_envs
    )

    observation_space = env.single_observation_space  # Use single env space
    action_space = env.single_action_space  # Use single env space

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        recurrent=True,
        num_envs=num_envs,
        learn_step=5,
        net_config={
            "encoder_config": {
                "hidden_state_size": 64,
                "max_seq_len": 5,
            }
        },
    )

    # Collect rollouts with recurrent network
    ppo.collect_rollouts(env, n_steps=5)

    # Check buffer contents
    assert ppo.rollout_buffer.pos == 5
    assert ppo.rollout_buffer.recurrent is True
    assert not np.array_equal(
        ppo.rollout_buffer.observations[0][0], np.zeros(observation_space.shape)
    )
    assert not np.array_equal(ppo.rollout_buffer.actions[0], np.zeros(1))
    assert ppo.rollout_buffer.hidden_states is not None
    assert ppo.rollout_buffer.next_hidden_states is not None

    # Verify hidden states were properly stored
    assert ppo.rollout_buffer.hidden_states["shared_encoder_h"][0].shape == (
        1,
        num_envs,
        64,
    )
    assert ppo.rollout_buffer.hidden_states["shared_encoder_c"][0].shape == (
        1,
        num_envs,
        64,
    )

    # Learn from collected rollouts
    loss = ppo.learn()

    assert isinstance(loss, float)
    assert loss >= 0.0


# Test PPO with hidden states and collect_rollouts
def test_ppo_with_hidden_states_multiple_envs_collect_rollouts_and_test():
    num_envs = 8
    env = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make("CartPole-v1")] * num_envs
    )
    num_test_envs = 2
    test_env = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make("CartPole-v1")] * num_test_envs
    )

    observation_space = env.single_observation_space  # Use single env space
    action_space = env.single_action_space  # Use single env space

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        recurrent=True,
        num_envs=num_envs,
        learn_step=5,
        net_config={
            "encoder_config": {
                "hidden_state_size": 64,
                "max_seq_len": 5,
            }
        },
    )

    # Collect rollouts with recurrent network
    ppo.collect_rollouts(env, n_steps=5)

    # Check buffer contents
    assert ppo.rollout_buffer.pos == 5
    assert ppo.rollout_buffer.recurrent is True
    assert not np.array_equal(
        ppo.rollout_buffer.observations[0][0], np.zeros(observation_space.shape)
    )
    assert not np.array_equal(ppo.rollout_buffer.actions[0], np.zeros(1))
    assert ppo.rollout_buffer.hidden_states is not None
    assert ppo.rollout_buffer.next_hidden_states is not None

    # Verify hidden states were properly stored
    assert ppo.rollout_buffer.hidden_states["shared_encoder_h"][0].shape == (
        1,
        num_envs,
        64,
    )
    assert ppo.rollout_buffer.hidden_states["shared_encoder_c"][0].shape == (
        1,
        num_envs,
        64,
    )

    # Learn from collected rollouts
    loss = ppo.learn()

    assert isinstance(loss, float)
    assert loss >= 0.0
    
    # Test test loop
    ppo.test(test_env)
    
    


# Test PPO collect_rollouts method
def test_ppo_collect_rollouts():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        learn_step=5,
    )

    env = DummyEnv(state_size=observation_space.shape, vect=True, num_envs=1)

    # Collect rollouts
    ppo.collect_rollouts(env, n_steps=5)

    # Check buffer contents
    print(ppo.rollout_buffer.pos)
    assert ppo.rollout_buffer.pos == 5
    assert not np.array_equal(
        ppo.rollout_buffer.observations[0][0], np.zeros(observation_space.shape)
    )
    # Check shape and dtype of the stored action for the first env/step
    assert ppo.rollout_buffer.actions[0].shape == (
        ppo.num_envs,
    )  # Shape should be (num_envs,)
    assert (
        ppo.rollout_buffer.actions.dtype == np.int64
    )  # Dtype for Discrete action space

    # Compute returns and advantages should have been called
    assert not np.array_equal(ppo.rollout_buffer.returns[:, 0], np.zeros(5))

    # Learn from collected rollouts
    loss = ppo.learn()

    assert isinstance(loss, float)
    assert loss >= 0.0
    


def test_ppo_wrap_at_capacity():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        learn_step=10,
        rollout_buffer_config={"wrap_at_capacity": True},
    )

    env = DummyEnv(state_size=observation_space.shape, vect=True, num_envs=1)

    ppo.collect_rollouts(env, n_steps=10)

    assert ppo.rollout_buffer.pos == 10
    assert ppo.rollout_buffer.full is True

    ppo.collect_rollouts(env, n_steps=7)

    assert ppo.rollout_buffer.pos == 7
    assert ppo.rollout_buffer.full is False

    # Collect rollouts resets the buffer
    ppo.collect_rollouts(env, n_steps=14)

    # Wrapped around, so pos is 4
    assert ppo.rollout_buffer.pos == 4
    assert ppo.rollout_buffer.full is True


# Test compatibility with old format
def test_ppo_backward_compatibility():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    # Create PPO with rollout buffer
    ppo_new = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
    )

    # Create PPO with old implementation
    ppo_old = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=False,
    )

    # Prepare experiences in old format
    num_steps = 5
    states = torch.rand(num_steps, *observation_space.shape)
    actions = torch.randint(0, action_space.n, (num_steps,)).float()
    log_probs = torch.randn(num_steps)
    rewards = torch.randn(num_steps)
    dones = torch.randint(0, 2, (num_steps,))
    values = torch.randn(num_steps)
    next_state = torch.rand(1, *observation_space.shape)
    next_done = np.zeros(1)
    experiences = [
        [states],
        [actions],
        [log_probs],
        [rewards],
        [dones],
        [values],
        [next_state],
        [next_done],
    ]

    # Both should work with old format
    loss_old = ppo_old.learn(experiences)
    loss_new = ppo_new.learn(experiences)

    assert isinstance(loss_old, float)
    assert isinstance(loss_new, float)

    # Fill rollout buffer
    for i in range(ppo_new.learn_step):
        obs = np.random.rand(*observation_space.shape)
        action = np.array([1])
        reward = 1.0
        done = i == ppo_new.learn_step - 1
        value = 0.5
        log_prob = -0.5
        next_obs = np.random.rand(*observation_space.shape)

        ppo_new.rollout_buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    ppo_new.rollout_buffer.compute_returns_and_advantages(
        last_value=0.0, last_done=np.zeros(1)
    )

    # New implementation should work without experiences (from buffer)
    loss_from_buffer = ppo_new.learn()
    assert isinstance(loss_from_buffer, float)

    # Old implementation should fail without experiences (no buffer)
    with pytest.raises(ValueError):
        ppo_old.learn()
