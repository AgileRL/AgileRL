import copy
from pathlib import Path

import dill
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from gymnasium import spaces
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces

from agilerl.algorithms.ppo import PPO
from agilerl.modules.cnn import EvolvableCNN
from agilerl.modules.mlp import EvolvableMLP
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import generate_discrete_space, generate_random_box_space

class DummyPPO(PPO):
    def __init__(
        self, observation_space, action_space, *args, **kwargs
    ):
        super().__init__(
            observation_space, action_space, *args, **kwargs
        )

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
    net_config = {"arch": "mlp", "hidden_size": [64, 64]}

    ppo = PPO(observation_space, action_space, net_config=net_config)

    print("ppo net config", ppo.net_config)
    assert ppo.algo == "PPO"
    assert ppo.observation_space ==generate_random_box_space(shape=(4,), low=0, high=1)
    assert ppo.action_space == generate_random_box_space(shape=(2,), low=-1, high=1)
    assert ppo.discrete_actions == False
    assert ppo.net_config == {
        "arch": "mlp",
        "hidden_size": [64, 64],
        "mlp_activation": "Tanh",
        "mlp_output_activation": "Tanh",
    }, ppo.net_config

    assert ppo.batch_size == 64
    assert ppo.lr == 1e-4
    assert ppo.gamma == 0.99
    assert ppo.gae_lambda == 0.95
    assert ppo.mut is None
    assert ppo.action_std_init == 0.6
    assert ppo.clip_coef == 0.2
    assert ppo.ent_coef == 0.01
    assert ppo.vf_coef == 0.5
    assert ppo.max_grad_norm == 0.5
    assert ppo.target_kl is None
    assert ppo.update_epochs == 4
    assert ppo.actor_network is None
    assert ppo.critic_network is None
    assert ppo.device == "cpu"
    assert ppo.accelerator is None
    assert ppo.index == 0
    assert ppo.scores == []
    assert ppo.fitness == []
    assert ppo.steps == [0]
    assert isinstance(ppo.actor, EvolvableMLP)
    assert isinstance(ppo.critic, EvolvableMLP)
    assert isinstance(ppo.optimizer, optim.Adam)
    assert ppo.arch == "mlp"


# Initializes actor network with EvolvableCNN based on net_config and Accelerator.
def test_initialize_ppo_with_cnn_accelerator():
    observation_space = generate_random_box_space(shape=(3, 32, 32), low=0, high=255)
    action_space = generate_discrete_space(2)
    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
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
        wrap=wrap
    )

    net_config_cnn.update({"mlp_output_activation": "Softmax"})

    assert ppo.observation_space == observation_space
    assert ppo.action_space == action_space
    assert ppo.discrete_actions == True
    assert ppo.net_config == net_config_cnn
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
    assert ppo.actor_network == actor_network
    assert ppo.critic_network == critic_network
    assert isinstance(ppo.actor, EvolvableCNN)
    assert isinstance(ppo.critic, EvolvableCNN)
    assert ppo.arch == "cnn"
    assert isinstance(ppo.optimizer, AcceleratedOptimizer)

# Can initialize ppo with an actor network
@pytest.mark.parametrize(
    "obs_space, action_space, actor_network, critic_network, input_tensor, input_tensor_critic",
    [
        ("vector_space", "action_space", "simple_mlp", "simple_mlp_critic", torch.randn(1, 4), torch.randn(1, 6)),
    ],
)
def test_initialize_ppo_with_actor_network(
    obs_space, action_space, actor_network, critic_network, input_tensor, input_tensor_critic, request
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
    assert ppo.net_config is None
    assert ppo.batch_size == 64
    assert ppo.lr == 1e-4
    assert ppo.gamma == 0.99
    assert ppo.gae_lambda == 0.95
    assert ppo.mut is None
    assert ppo.action_std_init == 0.6
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
    assert ppo.actor_network == actor_network
    assert ppo.actor == actor_network
    assert ppo.critic_network == critic_network
    assert ppo.critic == critic_network
    assert isinstance(ppo.optimizer, optim.Adam)
    assert ppo.arch == actor_network.arch


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
            mlp_activation="Tanh",
            mlp_output_activation="Tanh",
        )
        critic_network = EvolvableMLP(
            num_inputs=observation_space.shape[0] + action_space.n,
            num_outputs=1,
            hidden_size=[64, 64],
            mlp_activation="Tanh",
        )
    else:
        actor_network = EvolvableCNN(
            input_shape=observation_space.shape,
            num_outputs=action_space.n,
            channel_size=[8, 8],
            kernel_size=[2, 2],
            stride_size=[1, 1],
            hidden_size=[64, 64],
            mlp_activation="Tanh",
            mlp_output_activation="Tanh",
        )

        critic_network = EvolvableCNN(
            input_shape=observation_space.shape,
            num_outputs=action_space.n,
            channel_size=[8, 8],
            kernel_size=[2, 2],
            stride_size=[1, 1],
            hidden_size=[64, 64],
            critic=True,
            mlp_activation="Tanh",
        )

    ppo = PPO(
        observation_space,
        action_space,
        actor_network=actor_network,
        critic_network=critic_network
    )

    assert ppo.observation_space == observation_space
    assert ppo.action_space == action_space
    assert ppo.batch_size == 64
    assert ppo.lr == 1e-4
    assert ppo.gamma == 0.99
    assert ppo.gae_lambda == 0.95
    assert ppo.mut is None
    assert ppo.action_std_init == 0.6
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
    assert ppo.actor_network == actor_network
    assert ppo.actor == actor_network
    assert ppo.critic_network == critic_network
    assert ppo.critic == critic_network
    assert isinstance(ppo.optimizer, optim.Adam)
    assert ppo.arch == actor_network.arch


def test_initialize_ppo_with_incorrect_actor_net():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)
    actor_network = "dummy"
    critic_network = "dummy"
    with pytest.raises(AssertionError):
        ppo = PPO(
            observation_space,
            action_space,
            actor_network=actor_network,
            critic_network=critic_network
        )
        assert ppo


# Can initialize ppo with an actor network but no critic - should trigger warning
@pytest.mark.parametrize(
    "observation_space, actor_network, critic_network, input_tensor, input_tensor_critic",
    [
        (generate_random_box_space(shape=(4,), low=0, high=1), "simple_mlp", "simple_mlp_critic", torch.randn(1, 4), torch.randn(1, 6)),
    ],
)
def test_initialize_ppo_with_actor_network_no_critic(
    observation_space, actor_network, critic_network, input_tensor, input_tensor_critic, request
):
    action_space = generate_discrete_space(2)
    actor_network = request.getfixturevalue(actor_network)
    actor_network = MakeEvolvable(actor_network, input_tensor)
    with pytest.raises(AssertionError):
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
    ppo = PPO(observation_space=generate_random_box_space(shape=(5,), low=0, high=1), action_space=generate_discrete_space(2))
    prepared_state = ppo.preprocess_observation(state)
    assert isinstance(prepared_state, torch.Tensor)


def test_unsqueeze_prepare():
    state = np.array([1, 2, 3, 4])
    ppo = PPO(observation_space=generate_random_box_space(shape=(4,), low=0, high=1), action_space=generate_discrete_space(2))
    prepared_state = ppo.preprocess_observation(state)
    assert isinstance(prepared_state, torch.Tensor)


def test_prepare_state_cnn_accelerator():
    accelerator = Accelerator()
    observation_space = generate_random_box_space(shape=(3, 32, 32), low=0, high=1)
    state = torch.rand(*observation_space.shape)
    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
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
    return PPO(
        observation_space, action_space, accelerator=accelerator
    )


@pytest.mark.parametrize(
    "observation_space, action_space, accelerator",
    [
        (generate_random_box_space(shape=(4,), low=0, high=1), generate_random_box_space(shape=(2,), low=0, high=1), None),
        (generate_discrete_space(4), generate_discrete_space(2), Accelerator()),
    ],
)
# Returns the expected action when given a state observation.
def test_returns_expected_action(
    observation_space, action_space, build_ppo
):
    if isinstance(observation_space, spaces.Discrete):
        state = np.array([[0]])
    else:
        state = np.array([[1, 2, 3, 4]])

    # First with grad=False
    grad = False
    action, action_logprob, dist_entropy, state_values = build_ppo.get_action(
        state, grad=grad
    )

    assert isinstance(action, np.ndarray)
    assert isinstance(action_logprob, np.ndarray)
    assert isinstance(dist_entropy, np.ndarray)
    assert isinstance(state_values, np.ndarray)

    if isinstance(action_space, spaces.Discrete):
        for act in action:
            assert act.is_integer()
            assert act >= 0 and act < action_space.n
    else:
        action = action[0]
        assert len(action) == action_space.shape[0]
        for act in action:
            assert isinstance(act, np.float32)

    # Now with grad=True, and eval_action
    grad = True
    eval_action = torch.Tensor([[0, 1]])
    action, action_logprob, dist_entropy, state_values = build_ppo.get_action(
        state, action=eval_action, grad=grad
    )

    assert isinstance(action, torch.Tensor)
    assert isinstance(action_logprob, torch.Tensor)
    assert isinstance(dist_entropy, torch.Tensor)
    assert isinstance(state_values, torch.Tensor)

    if isinstance(action_space, spaces.Discrete):
        action = torch.argmax(action, dim=-1)
        assert act.is_integer()
        assert act >= 0 and act < action_space.n
    else:
        action = action.cpu().data.numpy()
        action = action[0]
        assert len(action) == action_space.shape[0]
        for act in action:
            assert isinstance(act, np.float32)


@pytest.mark.parametrize(
    "observation_space, action_space, accelerator",
    [
        (generate_random_box_space(shape=(4,), low=0, high=1), generate_discrete_space(2), None),
    ],
)
def test_returns_expected_action_mask_vectorized(build_ppo):
    state = np.array([[1, 2, 4, 5], [2, 3, 5, 1]])
    action_mask = np.array([[0, 1], [1, 0]])
    action, _, _, _ = build_ppo.get_action(state, action_mask=action_mask)
    assert np.array_equal(action, [1, 0])

# learns from experiences and updates network parameters
def test_learns_from_experiences():
    observation_space = generate_random_box_space(shape=(3, 32, 32), low=0, high=1)
    action_space = generate_discrete_space(2)
    batch_size = 45
    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
    }

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        net_config=net_config_cnn,
        batch_size=batch_size,
    )

    # Copy state dict before learning - should be different to after updating weights
    actor = ppo.actor
    actor_pre_learn_sd = str(copy.deepcopy(ppo.actor.state_dict()))

    # Create batch size + 1 samples to ensure we can handle this
    num_steps = batch_size + 1

    # Create a batch of experiences
    states = torch.rand(num_steps, *observation_space.shape)
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
    next_states = torch.rand(1, *observation_space.shape)

    experiences = [states, actions, log_probs, rewards, dones, values, next_states]
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
            "arch": "mlp",
            "hidden_size": [64, 64],
            "mlp_output_activation": "Tanh",
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

    experiences = [states, actions, log_probs, rewards, dones, values, next_state]
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
    agent = PPO(
        observation_space=observation_space, action_space=action_space
    )
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with unvectorised env
def test_algorithm_test_loop_unvectorized():
    observation_space = generate_random_box_space(shape=(4,), low=0, high=1)
    action_space = generate_discrete_space(2)

    env = DummyEnv(state_size=observation_space.shape, vect=False)

    agent = PPO(
        observation_space=observation_space, action_space=action_space
    )
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)


# Runs algorithm test loop with images
def test_algorithm_test_loop_images():
    observation_space = generate_random_box_space(shape=(3, 32, 32), low=0, high=1)
    action_space = generate_discrete_space(2)

    env = DummyEnv(state_size=observation_space.shape, vect=True)

    net_config_cnn = {
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
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
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
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
    assert clone_agent.net_config == ppo.net_config
    assert clone_agent.actor_network == ppo.actor_network
    assert clone_agent.critic_network == ppo.critic_network
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
    assert clone_agent.net_config == ppo.net_config
    assert clone_agent.actor_network == ppo.actor_network
    assert clone_agent.critic_network == ppo.critic_network
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
    assert clone_agent.net_config == ppo.net_config
    assert clone_agent.actor_network == ppo.actor_network
    assert clone_agent.critic_network == ppo.critic_network
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
    experiences = states, actions, log_probs, rewards, dones, values, next_states
    ppo.learn(experiences)
    clone_agent = ppo.clone()
    assert clone_agent.observation_space == ppo.observation_space
    assert clone_agent.action_space == ppo.action_space
    assert clone_agent.net_config == ppo.net_config
    assert clone_agent.actor_network == ppo.actor_network
    assert clone_agent.critic_network == ppo.critic_network
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
        action_space=generate_random_box_space(shape=(2,))
        )

    # Save the checkpoint to a file
    checkpoint_path = Path(tmpdir) / "checkpoint.pth"
    ppo.save_checkpoint(checkpoint_path)

    # Load the saved checkpoint file
    checkpoint = torch.load(checkpoint_path, pickle_module=dill)

    print("netty c ", ppo.net_config)

    # Check if the loaded checkpoint has the correct keys
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "critic_init_dict" in checkpoint
    assert "critic_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "net_config" in checkpoint
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
        action_space=generate_random_box_space(shape=(2,), low=0, high=1)
        )
    # Load checkpoint
    ppo.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert ppo.net_config == {
        "arch": "mlp",
        "hidden_size": [64, 64],
        "mlp_output_activation": "Softmax",
        "mlp_activation": "Tanh",
    }
    assert isinstance(ppo.actor, EvolvableMLP)
    assert isinstance(ppo.critic, EvolvableMLP)
    assert ppo.lr == 1e-4
    assert ppo.batch_size == 64
    assert ppo.gamma == 0.99
    assert ppo.mut is None
    assert ppo.action_std_init == 0.6
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
        "arch": "cnn",
        "hidden_size": [8],
        "channel_size": [3],
        "kernel_size": [3],
        "stride_size": [1],
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
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "critic_init_dict" in checkpoint
    assert "critic_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "net_config" in checkpoint
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
        action_space=generate_random_box_space(shape=(2,), low=0, high=1)
        )
    # Load checkpoint
    ppo.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert ppo.net_config == net_config_cnn
    assert isinstance(ppo.actor, EvolvableCNN)
    assert isinstance(ppo.critic, EvolvableCNN)
    assert ppo.lr == 1e-4
    assert ppo.batch_size == 64
    assert ppo.gamma == 0.99
    assert ppo.mut is None
    assert ppo.action_std_init == 0.6
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
    observation_space = generate_random_box_space(shape=input_tensor.shape[1:], low=0, high=1)
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
    assert "actor_init_dict" in checkpoint
    assert "actor_state_dict" in checkpoint
    assert "critic_init_dict" in checkpoint
    assert "critic_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "net_config" in checkpoint
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
        action_space=generate_random_box_space(shape=(2,), low=0, high=1)
        )
    # Load checkpoint
    ppo.load_checkpoint(checkpoint_path)

    # Check if properties and weights are loaded correctly
    assert ppo.net_config is None
    assert isinstance(ppo.actor, nn.Module)
    assert isinstance(ppo.critic, nn.Module)
    assert ppo.lr == 1e-4
    assert ppo.batch_size == 64
    assert ppo.gamma == 0.99
    assert ppo.mut is None
    assert ppo.action_std_init == 0.6
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
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained(device, accelerator, tmpdir):
    # Initialize the ppo agent
    ppo = PPO(
        observation_space=generate_random_box_space(shape=(4,), low=0, high=1),
        action_space=generate_random_box_space(shape=(2,), low=0, high=1)
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
    assert new_ppo.net_config == ppo.net_config
    assert isinstance(new_ppo.actor, EvolvableMLP)
    assert isinstance(new_ppo.critic, EvolvableMLP)
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


@pytest.mark.parametrize(
    "device, accelerator",
    [
        ("cpu", None),
        ("cpu", Accelerator()),
    ],
)
# The saved checkpoint file contains the correct data and format.
def test_load_from_pretrained_cnn(device, accelerator, tmpdir):
    # Initialize the ppo agent
    ppo = PPO(
        observation_space=generate_random_box_space(shape=(3, 32, 32), low=0, high=1),
        action_space=generate_random_box_space(shape=(2,), low=0, high=1),
        net_config={
            "arch": "cnn",
            "hidden_size": [8],
            "channel_size": [3],
            "kernel_size": [3],
            "stride_size": [1]
        }
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
    assert new_ppo.net_config == ppo.net_config
    assert isinstance(new_ppo.actor, EvolvableCNN)
    assert isinstance(new_ppo.critic, EvolvableCNN)
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


@pytest.mark.parametrize(
    "observation_space, actor_network, input_tensor",
    [
        (generate_random_box_space(shape=(4,), low=0, high=1), "simple_mlp", torch.randn(1, 4)),
        (generate_random_box_space(shape=(3, 64, 64), low=0, high=1), "simple_cnn", torch.randn(1, 3, 64, 64)),
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
    assert new_ppo.net_config == ppo.net_config
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
