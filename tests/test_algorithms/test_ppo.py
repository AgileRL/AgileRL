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

from agilerl.algorithms.ppo import PPO
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.utils.rollout_buffer import RolloutBuffer
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import (
    generate_dict_or_tuple_space,
    generate_discrete_space,
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
def test_initializes_with_default_values():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_random_box_space(shape=(2,), low=-1, high=1)
    net_config = {"encoder_config": {"hidden_size": [64, 64]}}

    ppo = PPO(observation_space, action_space, net_config=net_config)

    assert ppo.algo == "PPO"
    assert ppo.observation_space == generate_random_box_space(shape=(4,), low=0, high=1)
    assert ppo.action_space == generate_random_box_space(shape=(2,), low=-1, high=1)
    assert not ppo.discrete_actions
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
    assert isinstance(ppo.actor.encoder, EvolvableMLP)
    assert isinstance(ppo.critic.encoder, EvolvableMLP)
    assert isinstance(ppo.optimizer.optimizer, optim.Adam)


# Initializes actor network with EvolvableCNN based on net_config and Accelerator.
def test_initialize_ppo_with_cnn_accelerator():
    observation_space = generate_random_box_space(shape=(3, 32, 32), low=0, high=255)
    action_space = generate_discrete_space(2)
    net_config_cnn = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }
    batch_size = 64
    lr = 1e-4
    gamma = 0.99
    gae_lambda = 0.95
    mut = None
    action_std_init = 0.6
    clip_coef = 0.2
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    target_kl = None
    update_epochs = 4
    actor_network = None
    critic_network = None
    accelerator = Accelerator()
    wrap = True

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        net_config=net_config_cnn,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        mut=mut,
        action_std_init=action_std_init,
        clip_coef=clip_coef,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        target_kl=target_kl,
        update_epochs=update_epochs,
        actor_network=actor_network,
        critic_network=critic_network,
        accelerator=accelerator,
        wrap=wrap,
    )

    net_config_cnn.update({"output_activation": "Softmax"})

    assert ppo.observation_space == observation_space
    assert ppo.action_space == action_space
    assert ppo.discrete_actions
    assert ppo.batch_size == batch_size
    assert ppo.lr == lr
    assert ppo.gamma == gamma
    assert ppo.gae_lambda == gae_lambda
    assert ppo.mut == mut
    assert ppo.action_std_init == action_std_init
    assert ppo.clip_coef == clip_coef
    assert ppo.ent_coef == ent_coef
    assert ppo.vf_coef == vf_coef
    assert ppo.max_grad_norm == max_grad_norm
    assert ppo.target_kl == target_kl
    assert ppo.update_epochs == update_epochs
    assert isinstance(ppo.actor.encoder, EvolvableCNN)
    assert isinstance(ppo.critic.encoder, EvolvableCNN)
    assert isinstance(ppo.optimizer.optimizer, AcceleratedOptimizer)


# Can initialize ppo with an actor network
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
def test_initialize_ppo_with_actor_network(
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


@pytest.mark.parametrize(
    "observation_space, net_type",
    [
        (generate_random_box_space(shape=(4,), low=0, high=1), "mlp"),
        (generate_random_box_space(shape=(3, 64, 64), low=0, high=1), "cnn"),
    ],
)
def test_initialize_ppo_with_actor_network_evo_net(observation_space, net_type):
    action_space = generate_discrete_space(2)
    if net_type == "mlp":
        actor_network = EvolvableMLP(
            num_inputs=observation_space.shape[0],
            num_outputs=action_space.n,
            hidden_size=[64, 64],
            activation="Tanh",
            output_activation="Tanh",
        )
        critic_network = EvolvableMLP(
            num_inputs=observation_space.shape[0] + action_space.n,
            num_outputs=1,
            hidden_size=[64, 64],
            activation="Tanh",
        )
    else:
        actor_network = EvolvableCNN(
            input_shape=observation_space.shape,
            num_outputs=action_space.n,
            channel_size=[8, 8],
            kernel_size=[2, 2],
            stride_size=[1, 1],
            activation="Tanh",
            output_activation="Tanh",
        )

        critic_network = EvolvableCNN(
            input_shape=observation_space.shape,
            num_outputs=action_space.n,
            channel_size=[8, 8],
            kernel_size=[2, 2],
            stride_size=[1, 1],
            activation="Tanh",
        )

    ppo = PPO(
        observation_space,
        action_space,
        actor_network=actor_network,
        critic_network=critic_network,
    )

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
    assert ppo.device == "cpu"
    assert ppo.accelerator is None
    assert ppo.index == 0
    assert ppo.scores == []
    assert ppo.fitness == []
    assert ppo.steps == [0]
    assert isinstance(ppo.optimizer.optimizer, optim.Adam)


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


# Converts numpy array to torch tensor of type float
def test_convert_numpy_array_to_tensor():
    state = np.array([1, 2, 3, 4])
    ppo = PPO(
        observation_space=generate_random_box_space(shape=(5,), low=0, high=1),
        action_space=generate_discrete_space(2),
    )
    prepared_state = ppo.preprocess_observation(state)
    assert isinstance(prepared_state, torch.Tensor)


def test_unsqueeze_prepare():
    state = np.array([1, 2, 3, 4])
    ppo = PPO(
        observation_space=generate_random_box_space(shape=(4,), low=0, high=1),
        action_space=generate_discrete_space(2),
    )
    prepared_state = ppo.preprocess_observation(state)
    assert isinstance(prepared_state, torch.Tensor)


def test_prepare_state_cnn_accelerator():
    accelerator = Accelerator()
    observation_space = generate_random_box_space(shape=(3, 32, 32), low=0, high=1)
    state = torch.rand(*observation_space.shape)
    net_config_cnn = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }
    ppo = PPO(
        observation_space=observation_space,
        action_space=generate_discrete_space(2),
        net_config=net_config_cnn,
        accelerator=accelerator,
    )
    prepared_state = ppo.preprocess_observation(state)
    assert isinstance(prepared_state, torch.Tensor)
    assert prepared_state.dtype == torch.float32


@pytest.fixture
def build_ppo(observation_space, action_space, accelerator):
    return PPO(observation_space, action_space, accelerator=accelerator)


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
    "observation_space, action_space, accelerator",
    [
        (
            generate_random_box_space(shape=(4,), low=0, high=1),
            generate_discrete_space(2),
            None,
        ),
    ],
)
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
def test_learns_from_experiences(observation_space):
    batch_size = 45
    action_space = spaces.Discrete(2)
    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        batch_size=batch_size,
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
    log_probs = torch.randn(
        num_steps,
    )
    rewards = torch.randn(
        num_steps,
    )
    dones = torch.randint(0, 2, (num_steps,))
    values = torch.randn(
        num_steps,
    )
    next_done = np.zeros(1)
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


# learns from experiences and updates network parameters
def test_learns_from_experiences_continuous_accel():
    accelerator = Accelerator()
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    batch_size = 10
    target_kl = 0

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        net_config={
            "encoder_config": {
                "hidden_size": [64, 64],
                "output_activation": "Tanh",
            }
        },
        target_kl=target_kl,
        batch_size=batch_size,
        accelerator=accelerator,
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = ppo.actor
    actor_pre_learn_sd = str(copy.deepcopy(ppo.actor.state_dict()))

    num_steps = 11

    # Create a batch of experiences
    states = torch.rand(num_steps, *observation_space.shape)
    actions = torch.rand(num_steps, action_space.shape[0])
    log_probs = torch.randn(
        num_steps,
    )
    rewards = torch.randn(
        num_steps,
    )
    dones = torch.randint(0, 2, (num_steps,))
    values = torch.randn(
        num_steps,
    )
    next_state = torch.rand(1, *observation_space.shape)
    next_done = np.zeros(1)
    experiences = [
        states,
        actions,
        log_probs,
        rewards,
        dones,
        values,
        next_state,
        next_done,
    ]
    # Call the learn method
    loss = ppo.learn(experiences)

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert actor == ppo.actor
    assert actor_pre_learn_sd != str(ppo.actor.state_dict())


# Runs algorithm test loop
def test_algorithm_test_loop():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)
    num_envs = 3

    env = DummyEnv(state_size=observation_space.shape, vect=True, num_envs=num_envs)

    # env = make_vect_envs("CartPole-v1", num_envs=num_envs)
    agent = PPO(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorised env
def test_algorithm_test_loop_unvectorized():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    env = DummyEnv(state_size=observation_space.shape, vect=False)

    agent = PPO(observation_space=observation_space, action_space=action_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with images
def test_algorithm_test_loop_images():
    observation_space = generate_random_box_space(shape=(3, 32, 32), low=0, high=1)
    action_space = generate_discrete_space(2)

    env = DummyEnv(state_size=observation_space.shape, vect=True)

    net_config_cnn = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }

    agent = PPO(
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
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }

    agent = PPO(
        observation_space=generate_random_box_space(shape=(3, 32, 32), low=0, high=1),
        action_space=action_space,
        net_config=net_config_cnn,
    )
    mean_score = agent.test(env, max_steps=10, swap_channels=True)
    assert isinstance(mean_score, float)


# Clones the agent and returns an identical agent.
def test_clone_returns_identical_agent():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    ppo = DummyPPO(observation_space, action_space)
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


def test_clone_new_index():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    ppo = PPO(observation_space, action_space)
    clone_agent = ppo.clone(index=100)

    assert clone_agent.index == 100


def test_clone_after_learning():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_random_box_space(shape=(2,), low=0, high=1)
    max_env_steps = 20
    num_vec_envs = 2
    ppo = PPO(observation_space, action_space)
    states = np.random.randn(max_env_steps, num_vec_envs, observation_space.shape[0])

    next_states = np.random.randn(num_vec_envs, observation_space.shape[0])
    actions = np.random.rand(max_env_steps, num_vec_envs, action_space.shape[0])
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
    assert str(clone_agent.actor.state_dict()) == str(ppo.actor.state_dict())
    assert str(clone_agent.critic.state_dict()) == str(ppo.critic.state_dict())
    assert str(clone_agent.optimizer.state_dict()) == str(ppo.optimizer.state_dict())
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores


# The saved checkpoint file contains the correct data and format.
def test_save_load_checkpoint_correct_data_and_format(tmpdir):
    # Initialize the ppo agent
    ppo = PPO(
        observation_space=generate_random_box_space(shape=(4,), low=0, high=1),
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

    ppo = PPO(
        observation_space=generate_random_box_space(shape=(4,), low=0, high=1),
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
    )
    # Load checkpoint
    print(
        "Actor state dict: ",
        checkpoint["network_info"]["modules"]["actor_state_dict"].keys(),
    )
    print(
        "Critic state dict: ",
        checkpoint["network_info"]["modules"]["critic_state_dict"].keys(),
    )
    print("Critic new state dict: ", ppo.critic.state_dict().keys())
    ppo.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(ppo.actor.encoder, EvolvableMLP)
    assert isinstance(ppo.critic.encoder, EvolvableMLP)
    assert ppo.lr == 1e-4
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


def test_save_load_checkpoint_correct_data_and_format_cnn(tmpdir):
    net_config_cnn = {
        "encoder_config": {
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1],
        }
    }

    # Initialize the ppo agent
    ppo = PPO(
        observation_space=generate_random_box_space(shape=(3, 32, 32), low=0, high=1),
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
        net_config=net_config_cnn,
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

    ppo = PPO(
        observation_space=generate_random_box_space(shape=(4,), low=0, high=1),
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
    )
    # Load checkpoint
    ppo.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert isinstance(ppo.actor.encoder, EvolvableCNN)
    assert isinstance(ppo.critic.encoder, EvolvableCNN)
    assert ppo.lr == 1e-4
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


@pytest.mark.parametrize(
    "device", ["cpu", torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained(device, accelerator, tmpdir):
    # Initialize the ppo agent
    ppo = PPO(
        observation_space=generate_random_box_space(shape=(4,), low=0, high=1),
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
        device=device,
        accelerator=accelerator,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ppo.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_ppo = PPO.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_ppo.observation_space == ppo.observation_space
    assert new_ppo.action_space == ppo.action_space
    assert new_ppo.discrete_actions == ppo.discrete_actions
    assert isinstance(new_ppo.actor.encoder, EvolvableMLP)
    assert isinstance(new_ppo.critic.encoder, EvolvableMLP)
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


@pytest.mark.parametrize(
    "device", ["cpu", torch.device("cuda" if torch.cuda.is_available() else "cpu")]
)
@pytest.mark.parametrize("accelerator", [None, Accelerator()])
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_cnn(device, accelerator, tmpdir):
    # Initialize the ppo agent
    ppo = PPO(
        observation_space=generate_random_box_space(shape=(3, 32, 32), low=0, high=1),
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
        net_config={
            "encoder_config": {
                "channel_size": [3],
                "kernel_size": [3],
                "stride_size": [1],
            }
        },
        device=device,
        accelerator=accelerator,
    )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ppo.save_checkpoint(checkpoint_path)

    # Create new agent object
    new_ppo = PPO.load(checkpoint_path, device=device, accelerator=accelerator)

    # Check if properties and weights are loaded correctly
    assert new_ppo.observation_space == ppo.observation_space
    assert new_ppo.action_space == ppo.action_space
    assert new_ppo.discrete_actions == ppo.discrete_actions
    assert isinstance(new_ppo.actor.encoder, EvolvableCNN)
    assert isinstance(new_ppo.critic.encoder, EvolvableCNN)
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


# Test the RolloutBuffer implementation
def test_rollout_buffer_initialization():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    buffer = RolloutBuffer(
        capacity=100,
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
    assert buffer.recurrent == False
    assert buffer.hidden_size is None
    assert buffer.device == "cpu"
    assert buffer.pos == 0
    assert buffer.full == False

    # Test with hidden states
    buffer = RolloutBuffer(
        capacity=100,
        observation_space=observation_space,
        action_space=action_space,
        device="cpu",
        gae_lambda=0.95,
        gamma=0.99,
        recurrent=True,
        hidden_size=(64,),
    )

    assert buffer.recurrent == True
    assert buffer.hidden_size == (64,)
    assert buffer.hidden_states is not None
    assert buffer.next_hidden_states is not None


# Test adding samples to the buffer
def test_rollout_buffer_add():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    buffer = RolloutBuffer(
        capacity=100,
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
    print(np.array_equal(buffer.observations[0], obs))
    assert np.array_equal(buffer.observations[0], obs)
    assert np.array_equal(buffer.actions[0], action[0])  # Discrete action space
    assert buffer.rewards[0] == reward
    assert buffer.dones[0] == done
    assert buffer.values[0] == value
    assert buffer.log_probs[0] == log_prob
    assert np.array_equal(buffer.next_observations[0], next_obs)

    # Add samples until buffer is full
    for i in range(99):
        buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    assert buffer.pos == 0  # Wrapped around
    assert buffer.full == True


# Test computing returns and advantages
def test_rollout_buffer_compute_returns_and_advantages():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    buffer = RolloutBuffer(
        capacity=5,
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
    buffer.compute_returns_and_advantages()

    # Check that returns and advantages are computed
    assert not np.array_equal(buffer.returns, np.zeros(5))
    assert not np.array_equal(buffer.advantages, np.zeros(5))

    # Check that returns are higher for earlier steps (due to discounting)
    assert buffer.returns[0] > buffer.returns[4]


# Test getting batch from buffer
def test_rollout_buffer_get_batch():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    buffer = RolloutBuffer(
        capacity=100,
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
    buffer.compute_returns_and_advantages()

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

    assert ppo.use_rollout_buffer == True
    assert hasattr(ppo, "rollout_buffer")
    assert isinstance(ppo.rollout_buffer, RolloutBuffer)
    assert ppo.rollout_buffer.capacity == ppo.learn_step
    assert ppo.rollout_buffer.recurrent == False

    # Test with hidden states
    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        recurrent=True,
        hidden_size=64,
    )

    assert ppo.recurrent == True
    assert ppo.hidden_size == (64,)
    assert ppo.rollout_buffer.recurrent == True
    assert ppo.rollout_buffer.hidden_size == (64,)


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
    ppo.rollout_buffer.compute_returns_and_advantages()

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
        hidden_size=(64,),
    )

    # Get action with hidden state
    obs = np.random.rand(*observation_space.shape)
    hidden_state = np.zeros((1, 64))

    action, log_prob, entropy, value, next_hidden = ppo.get_action(
        obs, hidden_state=hidden_state
    )

    assert action.shape[0] == 1
    assert isinstance(log_prob, np.ndarray)
    assert isinstance(entropy, np.ndarray)
    assert isinstance(value, np.ndarray)
    assert next_hidden is not None
    assert next_hidden.shape == (1, 64)


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
    assert ppo.rollout_buffer.pos == 5
    assert not np.array_equal(
        ppo.rollout_buffer.observations[0], np.zeros(observation_space.shape)
    )
    assert not np.array_equal(ppo.rollout_buffer.actions[0], np.zeros(1))

    # Compute returns and advantages should have been called
    assert not np.array_equal(ppo.rollout_buffer.returns, np.zeros(5))
    assert not np.array_equal(ppo.rollout_buffer.advantages, np.zeros(5))

    # Learn from collected rollouts
    loss = ppo.learn()

    assert isinstance(loss, float)
    assert loss >= 0.0


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

    ppo_new.rollout_buffer.compute_returns_and_advantages()

    # New implementation should work without experiences
    loss_from_buffer = ppo_new.learn()
    assert isinstance(loss_from_buffer, float)

    # Old implementation should fail without experiences
    with pytest.raises(ValueError):
        ppo_old.learn()
