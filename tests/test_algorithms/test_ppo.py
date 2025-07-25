import copy

import gymnasium
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces

from agilerl.algorithms.ppo import PPO
from agilerl.components.rollout_buffer import RolloutBuffer
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.rollouts import collect_rollouts, collect_rollouts_recurrent
from agilerl.typing import BPTTSequenceType
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
    assert ppo.num_envs == 1


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
    assert clone_agent.num_envs == ppo.num_envs
    assert clone_agent.index == ppo.index

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
    assert clone_agent.num_envs == ppo.num_envs
    assert clone_agent.index == ppo.index

    accelerator = (
        Accelerator(cpu=True) if torch.backends.mps.is_available() else Accelerator()
    )
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
    assert clone_agent.num_envs == ppo.num_envs


def test_clone_new_index(vector_space, discrete_space):
    ppo = PPO(vector_space, discrete_space)
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
def test_clone_after_learning(
    device, use_rollout_buffer, recurrent, share_encoders, vector_space
):
    # accept if recurrent and no rollout buffer
    if recurrent and not use_rollout_buffer:
        return

    # check if device is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    observation_space = vector_space
    action_space = vector_space
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
        # Use the correct rollout collection function based on whether the policy is recurrent
        if recurrent:
            collect_rollouts_recurrent(ppo, dummy_env)
        else:
            collect_rollouts(ppo, dummy_env)
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
    assert_state_dicts_equal(clone_agent.actor.state_dict(), ppo.actor.state_dict())

    if share_encoders and recurrent:
        # the critic might be different if share_encoders is True
        # (the encoder state might be different because of the logic in the share_encoder_parameters)
        # The important thing is that the head_net is the same as the encoder is neither ran during
        # the forward of the exploration, nor the learning step.
        assert_state_dicts_equal(
            clone_agent.critic.head_net.state_dict(), ppo.critic.head_net.state_dict()
        )
    else:
        assert_state_dicts_equal(
            clone_agent.critic.state_dict(), ppo.critic.state_dict()
        )

    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(), ppo.optimizer.state_dict()
    )
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores
    assert clone_agent.num_envs == ppo.num_envs
    assert clone_agent.index == ppo.index


# Test the RolloutBuffer implementation
def test_rollout_buffer_initialization(vector_space, discrete_space):
    observation_space = vector_space
    action_space = discrete_space

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
    assert buffer.buffer.get("hidden_states") is None


# Test the RolloutBuffer implementation
def test_rollout_buffer_initialization_recurrent(vector_space, discrete_space):
    observation_space = vector_space
    action_space = discrete_space

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
    assert buffer.buffer.get("hidden_states") is not None

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
    assert buffer.buffer.get("hidden_states") is not None


# Test adding samples to the buffer
def test_rollout_buffer_add(vector_space, discrete_space):
    observation_space = vector_space
    action_space = discrete_space
    device = "cpu"

    buffer = RolloutBuffer(
        capacity=100,
        num_envs=1,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
    )

    # Add a single sample
    obs = np.random.rand(*observation_space.shape).astype(observation_space.dtype)
    action = np.array([action_space.sample()])  # Ensure action is within space
    reward = 1.0
    done = False
    value = 0.5
    log_prob = -0.5
    next_obs = np.random.rand(*observation_space.shape).astype(observation_space.dtype)

    buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    assert buffer.pos == 1
    assert not buffer.full
    # where X is the environment index, and Y is the position in the buffer (pos-1 for last added)
    # Data is stored at buffer.pos - 1
    current_pos_idx = buffer.pos - 1
    assert np.array_equal(
        buffer.buffer.get("observations")[current_pos_idx, 0].cpu().numpy(), obs
    )
    assert np.array_equal(
        buffer.buffer.get("actions")[current_pos_idx, 0].cpu().numpy(), action[0]
    )
    assert buffer.buffer.get("rewards")[current_pos_idx, 0].item() == reward
    assert buffer.buffer.get("dones")[current_pos_idx, 0].item() == float(done)
    assert buffer.buffer.get("values")[current_pos_idx, 0].item() == value
    assert buffer.buffer.get("log_probs")[current_pos_idx, 0].item() == log_prob
    assert np.array_equal(
        buffer.buffer.get("next_observations")[current_pos_idx, 0].cpu().numpy(),
        next_obs,
    )

    # Add samples until buffer is full
    for i in range(buffer.capacity - 1):  # Already added one sample
        buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    assert buffer.pos == buffer.capacity  # pos is next insertion point
    assert buffer.full is True  # Buffer is full when pos reaches capacity

    buffer.reset()

    assert buffer.pos == 0
    assert buffer.full is False


# Test computing returns and advantages
def test_rollout_buffer_compute_returns_and_advantages(vector_space, discrete_space):
    observation_space = vector_space
    action_space = discrete_space
    device = "cpu"
    capacity = 5

    buffer = RolloutBuffer(
        capacity=capacity,
        num_envs=1,
        observation_space=observation_space,
        action_space=action_space,
        gamma=0.99,
        gae_lambda=0.95,
        device=device,
    )

    # Add samples
    for i in range(capacity):
        obs = np.random.rand(*observation_space.shape).astype(observation_space.dtype)
        action = np.array([action_space.sample()])
        reward = 1.0
        done = i == (capacity - 1)  # Last step is done
        value = 0.5
        log_prob = -0.5
        next_obs = np.random.rand(*observation_space.shape).astype(
            observation_space.dtype
        )

        buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    # Compute returns and advantages
    last_value = torch.tensor([[0.0]], device=device)  # Shape (num_envs, 1)
    last_done = torch.tensor([[0.0]], device=device)  # Shape (num_envs, 1)
    buffer.compute_returns_and_advantages(last_value, last_done)

    # Check that returns and advantages are computed
    # Slicing [:, 0] gets data for the first (and only) environment
    assert not np.array_equal(
        buffer.buffer.get("returns")[:, 0].cpu().numpy(), np.zeros((capacity, 1))
    )
    assert not np.array_equal(
        buffer.buffer.get("advantages")[:, 0].cpu().numpy(), np.zeros((capacity, 1))
    )

    # Check that returns are higher for earlier steps (due to discounting)
    assert (
        buffer.buffer.get("returns")[0, 0].item()
        > buffer.buffer.get("returns")[capacity - 1, 0].item()
    )


# Test getting batch from buffer
def test_rollout_buffer_get_batch(vector_space, discrete_space):
    observation_space = vector_space
    action_space = discrete_space
    device = "cpu"
    num_samples = 10

    buffer = RolloutBuffer(
        capacity=100,
        num_envs=1,
        observation_space=observation_space,
        action_space=action_space,
        device=device,
    )

    # Add samples
    for i in range(num_samples):
        obs = np.random.rand(*observation_space.shape).astype(observation_space.dtype)
        action = np.array([action_space.sample()])
        reward = 1.0
        done = i == (num_samples - 1)  # Last step is done
        value = 0.5
        log_prob = -0.5
        next_obs = np.random.rand(*observation_space.shape).astype(
            observation_space.dtype
        )

        buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    # Compute returns and advantages
    last_value = torch.tensor([[0.0]], device=device)
    last_done = torch.tensor([[0.0]], device=device)
    buffer.compute_returns_and_advantages(last_value, last_done)

    # Get all data (up to current pos)
    batch = buffer.get()  # Gets all data up to buffer.pos

    assert len(batch["observations"]) == num_samples
    assert len(batch["actions"]) == num_samples
    # Rewards, dones, values, log_probs are (num_samples, 1) after get() flattens num_envs
    assert len(batch["rewards"]) == num_samples
    assert len(batch["dones"]) == num_samples
    assert len(batch["values"]) == num_samples
    assert len(batch["log_probs"]) == num_samples
    assert len(batch["advantages"]) == num_samples
    assert len(batch["returns"]) == num_samples

    # Get batch of specific size
    batch_size = 5
    batch = buffer.get(batch_size=batch_size)

    assert len(batch["observations"]) == batch_size
    assert len(batch["actions"]) == batch_size

    # Get tensor batch
    tensor_batch = buffer.get_tensor_batch(batch_size=batch_size)

    assert isinstance(tensor_batch["observations"], torch.Tensor)
    assert isinstance(tensor_batch["actions"], torch.Tensor)
    assert isinstance(tensor_batch["advantages"], torch.Tensor)
    assert tensor_batch["observations"].shape[0] == batch_size


# Test PPO initialization with rollout buffer
@pytest.mark.parametrize("observation_space", ["vector_space", "image_space"])
@pytest.mark.parametrize("action_space", ["discrete_space", "vector_space"])
def test_ppo_with_rollout_buffer(observation_space, action_space, request):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)

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

    # Build an encoder configuration that matches the observation space type
    if len(observation_space.shape) == 3:  # Image observations – use CNN
        base_net_config = {
            "encoder_config": {
                "channel_size": [16, 32],
                "kernel_size": [3, 3],
                "stride_size": [1, 1],
            }
        }
        expected_shared = {}
        expected_separate = {}
    else:  # Vector observations – use LSTM
        base_net_config = {
            "encoder_config": {
                "hidden_state_size": 64,
                "max_seq_len": 10,
            }
        }
        expected_shared = {
            "shared_encoder_h": (1, 1, 64),
            "shared_encoder_c": (1, 1, 64),
        }
        expected_separate = {
            "actor_encoder_h": (1, 1, 64),
            "actor_encoder_c": (1, 1, 64),
            "critic_encoder_h": (1, 1, 64),
            "critic_encoder_c": (1, 1, 64),
        }

    # Recurrent only when hidden states are expected (vector observations)
    recurrent_flag = len(expected_shared) > 0
    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        recurrent=recurrent_flag,
        use_rollout_buffer=True,
        net_config=base_net_config,
    )

    if recurrent_flag:
        assert ppo.recurrent
        assert ppo.rollout_buffer.recurrent
        assert ppo.rollout_buffer.hidden_state_architecture == expected_shared

        # Test with separated encoders when hidden states exist
        base_net_config_share = base_net_config.copy()
        ppo = PPO(
            observation_space=observation_space,
            action_space=action_space,
            recurrent=True,
            use_rollout_buffer=True,
            share_encoders=False,
            net_config=base_net_config_share,
        )

        assert ppo.rollout_buffer.hidden_state_architecture == expected_separate
        assert not ppo.share_encoders

    # Test with hidden states / separated encoders
    if expected_separate:
        base_net_config_share = base_net_config.copy()
        ppo = PPO(
            observation_space=observation_space,
            action_space=action_space,
            recurrent=True,
            use_rollout_buffer=True,
            share_encoders=False,
            net_config=base_net_config_share,
        )

        assert ppo.recurrent
        assert ppo.rollout_buffer.hidden_state_architecture == expected_separate
        assert ppo.rollout_buffer.recurrent
        assert not ppo.share_encoders


# Test PPO learning with rollout buffer
@pytest.mark.parametrize("observation_space", ["vector_space"])
@pytest.mark.parametrize("action_space", ["discrete_space", "vector_space"])
@pytest.mark.parametrize("recurrent", [True, False])
@pytest.mark.parametrize(
    "bptt_sequence_type",
    [
        BPTTSequenceType.CHUNKED,
        BPTTSequenceType.MAXIMUM,
        BPTTSequenceType.FIFTY_PERCENT_OVERLAP,
    ],
)
def test_ppo_learn_with_rollout_buffer(
    observation_space, action_space, bptt_sequence_type, recurrent, request
):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)

    batch_size = 32
    learn_step = 64

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
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        learn_step=learn_step,
        batch_size=batch_size,
        bptt_sequence_type=bptt_sequence_type,
        recurrent=recurrent,
        net_config=net_config,
    )

    # Fill the buffer manually
    for i in range(learn_step):
        obs = np.random.rand(*observation_space.shape).astype(observation_space.dtype)
        action = np.array([action_space.sample()])
        reward = 1.0
        done = i == (learn_step - 1)  # Last step is done
        value = 0.5
        log_prob = -0.5
        next_obs = np.random.rand(*observation_space.shape).astype(
            observation_space.dtype
        )
        if recurrent:
            hidden_state = ppo.get_initial_hidden_state()
            ppo.rollout_buffer.add(
                obs, action, reward, done, value, log_prob, next_obs, hidden_state
            )
        else:
            ppo.rollout_buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    # Compute returns and advantages (normally called by collect_rollouts)
    # For manual filling, we might need to call it if not implicitly handled by learn()
    # However, PPO.learn() calls buffer.compute_returns_and_advantages if experiences are None
    # So, this explicit call might not be strictly necessary if learn() is called without experiences.
    # For clarity in testing buffer functionality, it's fine.
    last_value = torch.zeros((ppo.num_envs, 1), device=ppo.device)
    last_done = torch.zeros((ppo.num_envs, 1), device=ppo.device)
    ppo.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

    # Learn from rollout buffer
    loss = ppo.learn()

    assert isinstance(loss, float)
    assert loss >= 0.0


# Test PPO with hidden states
@pytest.mark.parametrize("use_rollout_buffer", [True, False])
@pytest.mark.parametrize("max_seq_len", [None, 10])
def test_ppo_with_hidden_states(
    vector_space, discrete_space, use_rollout_buffer, max_seq_len
):
    observation_space = vector_space
    action_space = discrete_space

    net_config = {
        "encoder_config": {
            "hidden_state_size": 64,
            "max_seq_len": max_seq_len,
        }
    }

    def make_ppo():
        return PPO(
            observation_space=observation_space,
            action_space=action_space,
            use_rollout_buffer=use_rollout_buffer,
            recurrent=True,
            net_config=net_config,
        )

    if use_rollout_buffer:
        if max_seq_len is None:
            with pytest.raises(ValueError):
                ppo = make_ppo()
            return
        else:
            ppo = make_ppo()
    else:
        with pytest.raises(ValueError):
            ppo = make_ppo()
        return

    # Get action with hidden state
    obs = np.random.rand(1, *observation_space.shape).astype(
        observation_space.dtype
    )  # Add batch dim for num_envs=1
    hidden_state = ppo.get_initial_hidden_state()

    action, log_prob, entropy, value, next_hidden = ppo.get_action(
        obs, hidden_state=hidden_state
    )

    assert action.shape[0] == 1
    assert isinstance(log_prob, np.ndarray)
    assert isinstance(entropy, np.ndarray)
    assert isinstance(value, np.ndarray)
    assert next_hidden is not None
    assert next_hidden.get("shared_encoder_h", None).shape == (
        1,
        1,
        64,
    )  # (directions, num_envs, hidden_size)
    assert next_hidden.get("shared_encoder_c", None).shape == (1, 1, 64)


# Test PPO with hidden states
def test_ppo_with_hidden_states_multiple_obs(vector_space, discrete_space):
    observation_space = vector_space
    action_space = discrete_space
    num_envs = 2

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
    obs = np.zeros((num_envs, *observation_space.shape), dtype=observation_space.dtype)
    hidden_state = ppo.get_initial_hidden_state(num_envs=num_envs)

    action, log_prob, entropy, value, next_hidden = ppo.get_action(
        obs, hidden_state=hidden_state
    )

    assert action.shape[0] == num_envs
    assert isinstance(log_prob, np.ndarray)
    assert isinstance(entropy, np.ndarray)
    assert isinstance(value, np.ndarray)
    assert next_hidden is not None
    assert next_hidden.get("shared_encoder_h", None).shape == (1, num_envs, 64)
    assert next_hidden.get("shared_encoder_c", None).shape == (1, num_envs, 64)


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

    assert action.shape[0] == num_envs
    assert isinstance(log_prob, np.ndarray)
    assert isinstance(entropy, np.ndarray)
    assert isinstance(value, np.ndarray)
    assert next_hidden is not None
    assert next_hidden.get("shared_encoder_h", None).shape == (1, num_envs, 64)
    assert next_hidden.get("shared_encoder_c", None).shape == (1, num_envs, 64)
    env.close()


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
    collect_rollouts_recurrent(ppo, env)

    # Check buffer contents
    assert ppo.rollout_buffer.pos == -(ppo.learn_step // -ppo.num_envs)
    assert ppo.rollout_buffer.recurrent is True
    # Check observation for the first env at the first timestep
    assert not np.array_equal(
        ppo.rollout_buffer.buffer.get("observations")[0, 0].cpu().numpy(),
        np.zeros(observation_space.shape, dtype=observation_space.dtype),
    )
    # Check actions for all envs at the first timestep
    assert ppo.rollout_buffer.buffer.get("actions") is not None

    hidden_states = ppo.rollout_buffer.buffer.get("hidden_states")
    assert hidden_states is not None
    assert hidden_states.get("shared_encoder_h") is not None
    assert hidden_states.get("shared_encoder_c") is not None

    # Verify hidden states were properly stored (first step's hidden state)
    assert hidden_states.get("shared_encoder_h")[0].shape == (
        num_envs,
        1,  # num_layers * directions
        64,  # hidden_size
    )
    assert hidden_states.get("shared_encoder_c")[0].shape == (
        num_envs,
        1,
        64,
    )

    # Learn from collected rollouts
    loss = ppo.learn()

    assert isinstance(loss, float)
    assert loss >= 0.0
    env.close()


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
    collect_rollouts_recurrent(ppo, env)

    # Check buffer contents
    assert ppo.rollout_buffer.pos == -(ppo.learn_step // -ppo.num_envs)
    assert ppo.rollout_buffer.recurrent is True
    assert not np.array_equal(
        ppo.rollout_buffer.buffer.get("observations")[0, 0].cpu().numpy(),
        np.zeros(observation_space.shape, dtype=observation_space.dtype),
    )
    assert ppo.rollout_buffer.buffer.get("actions")[0].cpu().numpy() is not None

    assert ppo.rollout_buffer.buffer.get("hidden_states") is not None

    # Verify hidden states were properly stored
    assert ppo.rollout_buffer.buffer.get("hidden_states").get("shared_encoder_h")[
        0
    ].shape == (
        num_envs,
        1,
        64,
    )
    assert ppo.rollout_buffer.buffer.get("hidden_states").get("shared_encoder_c")[
        0
    ].shape == (
        num_envs,
        1,
        64,
    )

    # Learn from collected rollouts
    loss = ppo.learn()

    assert isinstance(loss, float)
    assert loss >= 0.0

    # Test test loop
    ppo.test(test_env)
    env.close()
    test_env.close()


# Test PPO collect_rollouts method
@pytest.mark.parametrize(
    "observation_space",
    [
        "vector_space",
        "image_space",
        # "dict_space",
    ],
)
@pytest.mark.parametrize(
    "action_space",
    [
        "discrete_space",
        "vector_space",
        "multidiscrete_space",
        "multibinary_space",
    ],
)
@pytest.mark.parametrize(
    "bptt_sequence_type",
    [
        BPTTSequenceType.CHUNKED,
        BPTTSequenceType.MAXIMUM,
        BPTTSequenceType.FIFTY_PERCENT_OVERLAP,
    ],
)
def test_ppo_collect_rollouts(
    observation_space, action_space, bptt_sequence_type, request
):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)

    learn_step = 5

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        use_rollout_buffer=True,
        learn_step=learn_step,
        num_envs=1,  # Explicitly set num_envs for clarity
        bptt_sequence_type=bptt_sequence_type,
    )

    env = DummyEnv(state_size=observation_space.shape, vect=True, num_envs=ppo.num_envs)

    # Collect rollouts
    collect_rollouts(ppo, env, n_steps=learn_step)

    # Check if properties and weights are loaded correctly
    assert ppo.observation_space == ppo.observation_space
    assert ppo.action_space == ppo.action_space
    assert isinstance(ppo.actor, nn.Module)
    assert isinstance(ppo.critic, nn.Module)
    assert ppo.lr == ppo.lr
    assert str(ppo.actor.to("cpu").state_dict()) == str(ppo.actor.state_dict())
    assert str(ppo.critic.to("cpu").state_dict()) == str(ppo.critic.state_dict())
    assert ppo.batch_size == ppo.batch_size
    assert ppo.gamma == ppo.gamma
    assert ppo.mut == ppo.mut
    assert ppo.index == ppo.index
    assert ppo.scores == ppo.scores
    assert ppo.fitness == ppo.fitness
    assert ppo.steps == ppo.steps
