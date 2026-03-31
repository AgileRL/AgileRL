import copy

import gymnasium
import numpy as np
import pytest
import torch
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces
from tensordict import TensorDict
from torch import nn, optim

from agilerl.algorithms.ppo import PPO
from agilerl.components.rollout_buffer import RolloutBuffer
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput
from agilerl.rollouts import collect_rollouts, collect_rollouts_recurrent
from agilerl.typing import BPTTSequenceType
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import assert_not_equal_state_dict, assert_state_dicts_equal


def get_eval_action_for_space(action_space: spaces.Space, device: torch.device):
    """Build a valid batched action tensor (batch=1) for evaluate_actions."""
    if isinstance(action_space, spaces.Discrete):
        # (1,) or (1, 1) integer indices in [0, n-1]
        return torch.zeros(1, dtype=torch.long, device=device).clamp(
            0, action_space.n - 1
        )
    if isinstance(action_space, spaces.MultiDiscrete):
        return torch.stack(
            [
                torch.randint(0, int(high), (1,), device=device)
                for high in action_space.nvec
            ],
            dim=1,
        )
    if isinstance(action_space, spaces.MultiBinary):
        n = action_space.n
        return torch.randint(0, 2, (1, n), device=device).float()
    if isinstance(action_space, spaces.Box):
        return torch.zeros(1, *action_space.shape, device=device)
    raise NotImplementedError(f"Unsupported action space: {type(action_space)}")


def get_batch_states(observation_space, num_steps) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(observation_space, spaces.Discrete):
        states = torch.randint(0, observation_space.n, (num_steps,)).float()
        next_states = torch.randint(0, observation_space.n, (1,)).float()
    elif isinstance(observation_space, spaces.MultiDiscrete):
        states = torch.stack(
            [
                torch.randint(0, high, (num_steps,))
                for high in observation_space.nvec.tolist()
            ],
            dim=1,
        ).float()
        next_states = torch.stack(
            [torch.randint(0, high, (1,)) for high in observation_space.nvec.tolist()],
            dim=1,
        ).float()
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
    else:
        raise NotImplementedError(
            f"Unsupported observation space: {type(observation_space)}"
        )
    return states, next_states


def _obs_batch_at(
    observation_space: spaces.Space,
    states,
    index: int,
) -> np.ndarray | dict[str, np.ndarray] | tuple[np.ndarray, ...]:
    """Single env observation batch (shape (1, ...)) at time ``index`` from stacked tensors."""
    if isinstance(observation_space, spaces.Dict):
        return {
            key: val[index : index + 1].detach().cpu().numpy().astype(np.float32)
            for key, val in states.items()
        }
    if isinstance(observation_space, spaces.Tuple):
        return tuple(
            t[index : index + 1].detach().cpu().numpy().astype(np.float32)
            for t in states
        )
    arr = states[index : index + 1].detach().cpu().numpy()
    dt = getattr(observation_space, "dtype", np.float32)
    return arr.astype(dt)


def _bootstrap_next_obs(
    observation_space: spaces.Space,
    next_states,
) -> np.ndarray | dict[str, np.ndarray] | tuple[np.ndarray, ...]:
    """Bootstrap next observation batch already shaped for ``num_envs == 1``."""
    if isinstance(observation_space, spaces.Dict):
        return {
            key: val.detach().cpu().numpy().astype(np.float32)
            for key, val in next_states.items()
        }
    if isinstance(observation_space, spaces.Tuple):
        return tuple(t.detach().cpu().numpy().astype(np.float32) for t in next_states)
    dt = getattr(observation_space, "dtype", np.float32)
    return next_states.detach().cpu().numpy().astype(dt)


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
            3,
            16,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # Input channels: 3 (for RGB images), Output channels: 16
        self.relu1 = nn.ReLU()
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            16,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
        )  # Input channels: 16, Output channels: 32
        self.relu2 = nn.ReLU()
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()  # Flatten the 2D feature map to a 1D vector
        self.linear1 = nn.Linear(
            32 * 16 * 16,
            128,
        )  # Fully connected layer with 128 output features
        self.relu3 = nn.ReLU()
        self.linear2 = nn.Linear(
            128,
            128,
        )  # Fully connected layer with 128 output features

    def forward(self, x, xc):
        x = self.mp1(self.relu1(self.conv1(x)))
        x = self.mp2(self.relu2(self.conv2(x)))
        x = self.flat(x)
        x = self.relu3(self.linear1(x))
        return self.relu3(self.linear2(x))


@pytest.fixture(scope="function")
def build_ppo(observation_space, action_space, recurrent, accelerator_flag, request):
    accelerator = Accelerator() if accelerator_flag else None
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)
    return PPO(
        observation_space,
        action_space,
        recurrent=recurrent,
        accelerator=accelerator,
    )


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
    [
        "vector_space",
        "discrete_space",
        "multidiscrete_space",
        "multibinary_space",
    ],
)
@pytest.mark.parametrize("accelerator_flag", [False, True])
def test_initialize_ppo(
    observation_space,
    action_space,
    encoder_cls,
    accelerator_flag,
    request,
):
    accelerator = Accelerator() if accelerator_flag else None
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)
    ppo = PPO(
        observation_space,
        action_space,
        accelerator=accelerator,
        recurrent=False,
    )

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
    assert ppo.steps == 0
    assert isinstance(ppo.actor.encoder, encoder_cls)
    assert isinstance(ppo.critic.encoder, encoder_cls)
    expected_optimizer = AcceleratedOptimizer if accelerator else optim.Adam
    assert isinstance(ppo.optimizer.optimizer, expected_optimizer)
    ppo.clean_up()


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
    assert ppo.steps == 0
    assert isinstance(ppo.optimizer.optimizer, optim.Adam)
    assert ppo.num_envs == 1
    ppo.clean_up()


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
        PPO(
            observation_space,
            discrete_space,
            actor_network=actor_network,
            critic_network=critic_network,
        )


@pytest.mark.parametrize(
    "observation_space",
    ["vector_space", "image_space", "dict_space"],
)
@pytest.mark.parametrize(
    "action_space",
    ["vector_space", "discrete_space", "multidiscrete_space", "multibinary_space"],
)
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("recurrent", [False])
# Returns the expected action when given a state observation.
def test_returns_expected_action(
    observation_space,
    action_space,
    build_ppo,
    request,
    recurrent,
    accelerator_flag,
):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)
    state = observation_space.sample()

    act_ret = build_ppo.get_action(state)
    action, action_logprob, dist_entropy, state_values = act_ret

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

    # Now with grad=True, and eval_action (must match action_space)
    eval_action = get_eval_action_for_space(action_space, build_ppo.device)
    action_logprob, dist_entropy, state_values = build_ppo.evaluate_actions(
        obs=state,
        actions=eval_action,
    )

    assert isinstance(action_logprob, torch.Tensor)
    assert isinstance(dist_entropy, torch.Tensor)
    assert isinstance(state_values, torch.Tensor)
    build_ppo.clean_up()


@pytest.mark.parametrize(
    "observation_space",
    [
        "vector_space",
        "discrete_space",
        "multidiscrete_space",
        "multibinary_space",
    ],
)
@pytest.mark.parametrize(
    "action_space",
    [
        "vector_space",
        "discrete_space",
        "multidiscrete_space",
        "multibinary_space",
    ],
)
@pytest.mark.parametrize("recurrent", [True])
@pytest.mark.parametrize("accelerator_flag", [False])
# Returns the expected action when given a state observation.
def test_returns_expected_action_recurrent(
    observation_space,
    action_space,
    recurrent,
    accelerator_flag,
    build_ppo,
    request,
):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)
    state = observation_space.sample()

    act_ret = build_ppo.get_action(
        state,
        hidden_state=build_ppo.get_initial_hidden_state(),
    )
    action, action_logprob, dist_entropy, state_values, hidden_state = act_ret

    assert isinstance(action, np.ndarray)
    assert isinstance(action_logprob, np.ndarray)
    assert isinstance(dist_entropy, np.ndarray)
    assert isinstance(state_values, np.ndarray)
    assert hidden_state is not None

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

    eval_action = get_eval_action_for_space(action_space, build_ppo.device)
    action_logprob, dist_entropy, state_values = build_ppo.evaluate_actions(
        obs=state,
        actions=eval_action,
        hidden_state=hidden_state,
    )

    assert isinstance(action_logprob, torch.Tensor)
    assert isinstance(dist_entropy, torch.Tensor)
    assert isinstance(state_values, torch.Tensor)
    build_ppo.clean_up()


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
    ppo.clean_up()


@pytest.mark.parametrize("observation_space", ["vector_space"])
@pytest.mark.parametrize("action_space", ["discrete_space"])
@pytest.mark.parametrize("accelerator_flag", [False])
@pytest.mark.parametrize("recurrent", [False])
def test_returns_expected_action_mask_vectorized(
    build_ppo,
    observation_space,
    action_space,
    recurrent,
    request,
    accelerator_flag,
):
    observation_space = request.getfixturevalue(observation_space)
    request.getfixturevalue(action_space)  # for parametrization
    state = np.stack([observation_space.sample(), observation_space.sample()])
    action_mask = np.stack([np.array([0, 1]), np.array([1, 0])])
    action, _, _, _ = build_ppo.get_action(state, action_mask=action_mask)
    assert np.array_equal(action, [1, 0]), action
    build_ppo.clean_up()


@pytest.mark.parametrize(
    "observation_space",
    ["vector_space", "image_space", "dict_space"],
)
@pytest.mark.parametrize("accelerator_flag", [False, True])
def test_learns_from_rollout_buffer_updates_weights(
    observation_space,
    discrete_space,
    accelerator_flag,
    request,
):
    """PPO.learn() updates actor weights after the rollout buffer is filled manually."""
    accelerator = Accelerator() if accelerator_flag else None
    batch_size = 45
    observation_space = request.getfixturevalue(observation_space)
    ppo = PPO(
        observation_space=observation_space,
        action_space=discrete_space,
        batch_size=batch_size,
        accelerator=accelerator,
    )

    actor = ppo.actor
    actor_pre_learn_sd = copy.deepcopy(ppo.actor.state_dict())

    num_steps = batch_size + 1
    states, next_states = get_batch_states(observation_space, num_steps)
    actions = torch.randint(0, discrete_space.n, (num_steps,))
    log_probs = torch.randn(num_steps)
    rewards = torch.randn(num_steps)
    dones = torch.randint(0, 2, (num_steps,))
    values = torch.randn(num_steps)

    for i in range(num_steps):
        obs = _obs_batch_at(observation_space, states, i)
        nxt = (
            _bootstrap_next_obs(observation_space, next_states)
            if i == num_steps - 1
            else _obs_batch_at(observation_space, states, i + 1)
        )
        action = np.array([[int(actions[i].item())]], dtype=discrete_space.dtype)
        ppo.rollout_buffer.add(
            obs,
            action,
            float(rewards[i]),
            bool(dones[i].item()),
            float(values[i]),
            float(log_probs[i]),
            nxt,
        )

    last_value = torch.zeros((ppo.num_envs, 1), device=ppo.device)
    last_done = torch.zeros((ppo.num_envs, 1), device=ppo.device)
    ppo.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

    loss = ppo.learn()

    assert isinstance(loss, float)
    assert loss >= 0.0
    assert actor == ppo.actor
    assert_not_equal_state_dict(actor_pre_learn_sd, ppo.actor.state_dict())
    ppo.clean_up()


# Runs algorithm test loop
@pytest.mark.parametrize(
    "observation_space",
    ["vector_space", "discrete_space", "image_space"],
)
@pytest.mark.parametrize("num_envs", [1, 3])
def test_algorithm_test_loop(observation_space, discrete_space, num_envs, request):
    observation_space = request.getfixturevalue(observation_space)

    # Create a vectorised environment & test loop
    vect = num_envs > 1
    env = DummyEnv(state_size=observation_space.shape, vect=vect, num_envs=num_envs)
    agent = PPO(observation_space=observation_space, action_space=discrete_space)
    mean_score = agent.test(env, max_steps=10)
    assert isinstance(mean_score, float)
    agent.clean_up()


# Clones the agent and returns an identical agent.
@pytest.mark.parametrize("observation_space", ["vector_space"])
def test_clone_returns_identical_agent(observation_space, discrete_space, request):
    observation_space = request.getfixturevalue(observation_space)
    ppo = DummyPPO(observation_space, discrete_space)
    ppo.fitness = [200, 200, 200]
    ppo.scores = [94, 94, 94]
    ppo.steps = 2500
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
        clone_agent.optimizer.state_dict(),
        ppo.optimizer.state_dict(),
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
        clone_agent.optimizer.state_dict(),
        ppo.optimizer.state_dict(),
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
        clone_agent.optimizer.state_dict(),
        ppo.optimizer.state_dict(),
    )
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores
    assert clone_agent.num_envs == ppo.num_envs


def test_clone_new_index(vector_space, discrete_space):
    ppo = PPO(vector_space, discrete_space)
    clone_agent = ppo.clone(index=100)
    assert clone_agent.index == 100
    ppo.clean_up()
    clone_agent.clean_up()


@pytest.mark.parametrize("device", ["cpu", "cuda"], ids=lambda d: f"device={d}")
@pytest.mark.parametrize("recurrent", [True, False], ids=lambda r: f"recurrent={r}")
@pytest.mark.parametrize(
    "share_encoders",
    [True, False],
    ids=lambda s: f"share_encoders={s}",
)
def test_clone_after_learning(
    device,
    recurrent,
    share_encoders,
    vector_space,
):
    # check if device is available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    observation_space = vector_space
    action_space = vector_space
    num_vec_envs = 2

    net_config = {"encoder_config": {"hidden_state_size": 64}} if recurrent else {}

    ppo = PPO(
        observation_space,
        action_space,
        device=torch.device(device),
        recurrent=recurrent,
        net_config=net_config,
        num_envs=num_vec_envs,
        share_encoders=share_encoders,
        max_seq_len=None,
    )

    dummy_env = DummyEnv(observation_space.shape, vect=True, num_envs=num_vec_envs)
    collect_function = collect_rollouts_recurrent if recurrent else collect_rollouts
    collect_function(ppo, dummy_env)
    ppo.learn()

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
            clone_agent.critic.head_net.state_dict(),
            ppo.critic.head_net.state_dict(),
        )
    else:
        assert_state_dicts_equal(
            clone_agent.critic.state_dict(),
            ppo.critic.state_dict(),
        )

    assert_state_dicts_equal(
        clone_agent.optimizer.state_dict(),
        ppo.optimizer.state_dict(),
    )
    assert clone_agent.fitness == ppo.fitness
    assert clone_agent.steps == ppo.steps
    assert clone_agent.scores == ppo.scores
    assert clone_agent.num_envs == ppo.num_envs
    assert clone_agent.index == ppo.index


# Test PPO rollout buffer wiring at initialization
@pytest.mark.parametrize(
    "observation_space",
    [
        "vector_space",
        "discrete_space",
        "multidiscrete_space",
        "multibinary_space",
        "image_space",
        "dict_space",
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
def test_ppo_rollout_buffer_initialization(observation_space, action_space, request):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        learn_step=100,
    )

    assert hasattr(ppo, "rollout_buffer")
    assert isinstance(ppo.rollout_buffer, RolloutBuffer)
    assert ppo.rollout_buffer.capacity == ppo.learn_step
    assert not ppo.rollout_buffer.recurrent

    if not isinstance(
        observation_space,
        (spaces.Box, spaces.Discrete, spaces.MultiDiscrete, spaces.MultiBinary),
    ):
        pytest.skip("Recurrent PPO with non-vector space is not supported yet!")

    # Build an encoder configuration that matches the observation space type
    if len(observation_space.shape) == 3:  # Image observations – use CNN
        base_net_config = {
            "encoder_config": {
                "channel_size": [16, 32],
                "kernel_size": [3, 3],
                "stride_size": [1, 1],
            },
        }
        expected_shared = {}
        expected_separate = {}
    else:  # Vector observations – use LSTM
        base_net_config = {
            "encoder_config": {
                "hidden_state_size": 64,
            },
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
        max_seq_len=10,
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
            share_encoders=False,
            max_seq_len=10,
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
            share_encoders=False,
            max_seq_len=10,
            net_config=base_net_config_share,
        )

        assert ppo.recurrent
        assert ppo.rollout_buffer.hidden_state_architecture == expected_separate
        assert ppo.rollout_buffer.recurrent
        assert not ppo.share_encoders
        ppo.clean_up()


# Test PPO learning from a manually filled rollout buffer (BPTT / flat paths)
@pytest.mark.parametrize(
    "observation_space",
    [
        "vector_space",
        "discrete_space",
        "multidiscrete_space",
        "multibinary_space",
        "image_space",
        "dict_space",
    ],
)
@pytest.mark.parametrize(
    "action_space",
    ["discrete_space", "vector_space", "multidiscrete_space", "multibinary_space"],
)
@pytest.mark.parametrize("recurrent", [True, False])
@pytest.mark.parametrize("max_seq_len", [None, 10])
@pytest.mark.parametrize(
    "bptt_sequence_type",
    [
        BPTTSequenceType.CHUNKED,
        BPTTSequenceType.MAXIMUM,
        BPTTSequenceType.FIFTY_PERCENT_OVERLAP,
    ],
)
def test_ppo_learn_from_filled_rollout_buffer(
    observation_space,
    action_space,
    bptt_sequence_type,
    recurrent,
    max_seq_len,
    request,
):
    supported_spaces = [
        "vector_space",
        "discrete_space",
        "multidiscrete_space",
        "multibinary_space",
    ]
    if recurrent and observation_space not in supported_spaces:
        pytest.skip("Recurrent PPO with non-vector space is not supported yet!")

    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)

    batch_size = 32
    learn_step = 64

    net_config = {"encoder_config": {"hidden_state_size": 64}} if recurrent else {}

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        learn_step=learn_step,
        batch_size=batch_size,
        bptt_sequence_type=bptt_sequence_type,
        recurrent=recurrent,
        max_seq_len=max_seq_len,
        net_config=net_config,
    )

    # Fill the buffer manually
    for i in range(learn_step):
        obs, next_obs = get_batch_states(observation_space, 1)
        action = np.array([action_space.sample()], dtype=action_space.dtype)
        reward = 1.0
        done = i == (learn_step - 1)  # Last step is done
        value = 0.5
        log_prob = -0.5
        if recurrent:
            hidden_state = ppo.get_initial_hidden_state()
            ppo.rollout_buffer.add(
                obs,
                action,
                reward,
                done,
                value,
                log_prob,
                next_obs,
                hidden_state,
            )
        else:
            ppo.rollout_buffer.add(obs, action, reward, done, value, log_prob, next_obs)

    # Compute returns and advantages (normally called by collect_rollouts)
    last_value = torch.zeros((ppo.num_envs, 1), device=ppo.device)
    last_done = torch.zeros((ppo.num_envs, 1), device=ppo.device)
    ppo.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

    # Learn from rollout buffer
    loss = ppo.learn()

    assert isinstance(loss, float)
    assert loss >= 0.0
    ppo.clean_up()


# Test PPO with hidden states
@pytest.mark.parametrize("max_seq_len", [None, 10])
def test_ppo_with_hidden_states(
    vector_space,
    discrete_space,
    max_seq_len,
):
    observation_space = vector_space
    action_space = discrete_space

    net_config = {
        "encoder_config": {
            "hidden_state_size": 64,
        },
    }

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        recurrent=True,
        max_seq_len=max_seq_len,
        net_config=net_config,
    )

    # Get action with hidden state
    obs = np.random.rand(1, *observation_space.shape).astype(
        observation_space.dtype,
    )  # Add batch dim for num_envs=1
    hidden_state = ppo.get_initial_hidden_state()

    action, log_prob, entropy, value, next_hidden = ppo.get_action(
        obs,
        hidden_state=hidden_state,
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
    ppo.clean_up()


# Test PPO with hidden states
def test_ppo_with_hidden_states_multiple_obs(vector_space, discrete_space):
    observation_space = vector_space
    action_space = discrete_space
    num_envs = 2

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        recurrent=True,
        num_envs=num_envs,
        net_config={
            "encoder_config": {
                "hidden_state_size": 64,
            },
        },
    )

    # Get action with hidden state (multiple observations)
    obs = np.zeros((num_envs, *observation_space.shape), dtype=observation_space.dtype)
    hidden_state = ppo.get_initial_hidden_state(num_envs=num_envs)

    action, log_prob, entropy, value, next_hidden = ppo.get_action(
        obs,
        hidden_state=hidden_state,
    )

    assert action.shape[0] == num_envs
    assert isinstance(log_prob, np.ndarray)
    assert isinstance(entropy, np.ndarray)
    assert isinstance(value, np.ndarray)
    assert next_hidden is not None
    assert next_hidden.get("shared_encoder_h", None).shape == (1, num_envs, 64)
    assert next_hidden.get("shared_encoder_c", None).shape == (1, num_envs, 64)
    ppo.clean_up()


# Test PPO with hidden states
def test_ppo_with_hidden_states_multiple_envs():
    num_envs = 2
    env = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make("CartPole-v1")] * num_envs,
    )

    observation_space = env.single_observation_space  # Use single env space
    action_space = env.single_action_space  # Use single env space

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        recurrent=True,
        num_envs=num_envs,
        max_seq_len=10,
        net_config={
            "encoder_config": {
                "hidden_state_size": 64,
            },
        },
    )

    # Get action with hidden state (multiple observations)
    obs, _ = env.reset()
    hidden_state = ppo.get_initial_hidden_state(num_envs=num_envs)

    action, log_prob, entropy, value, next_hidden = ppo.get_action(
        obs,
        hidden_state=hidden_state,
    )

    assert action.shape[0] == num_envs
    assert isinstance(log_prob, np.ndarray)
    assert isinstance(entropy, np.ndarray)
    assert isinstance(value, np.ndarray)
    assert next_hidden is not None
    assert next_hidden.get("shared_encoder_h", None).shape == (1, num_envs, 64)
    assert next_hidden.get("shared_encoder_c", None).shape == (1, num_envs, 64)
    ppo.clean_up()
    env.close()


# Test PPO with hidden states and collect_rollouts
def test_ppo_with_hidden_states_multiple_envs_collect_rollouts():
    num_envs = 2
    env = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make("CartPole-v1")] * num_envs,
    )

    observation_space = env.single_observation_space  # Use single env space
    action_space = env.single_action_space  # Use single env space

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        recurrent=True,
        num_envs=num_envs,
        batch_size=num_envs,
        learn_step=5,
        max_seq_len=5,
        net_config={
            "encoder_config": {
                "hidden_state_size": 64,
            },
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
    ppo.clean_up()
    env.close()


# Test PPO with hidden states and collect_rollouts
def test_ppo_with_hidden_states_multiple_envs_collect_rollouts_and_test():
    num_envs = 8
    env = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make("CartPole-v1")] * num_envs,
    )
    num_test_envs = 2
    test_env = gymnasium.vector.SyncVectorEnv(
        [lambda: gymnasium.make("CartPole-v1")] * num_test_envs,
    )

    observation_space = env.single_observation_space  # Use single env space
    action_space = env.single_action_space  # Use single env space

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        recurrent=True,
        num_envs=num_envs,
        batch_size=num_envs,
        learn_step=5,
        max_seq_len=5,
        net_config={
            "encoder_config": {
                "hidden_state_size": 64,
            },
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
    ppo.clean_up()
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
    observation_space,
    action_space,
    bptt_sequence_type,
    request,
):
    observation_space = request.getfixturevalue(observation_space)
    action_space = request.getfixturevalue(action_space)

    learn_step = 5

    ppo = PPO(
        observation_space=observation_space,
        action_space=action_space,
        learn_step=learn_step,
        num_envs=1,  # Explicitly set num_envs for clarity
        bptt_sequence_type=bptt_sequence_type,
    )

    env = DummyEnv(state_size=observation_space.shape, vect=True, num_envs=ppo.num_envs)

    # Collect rollouts
    collect_rollouts(ppo, env, n_steps=learn_step)

    # Check if properties and weights are loaded correctly
    assert ppo.observation_space is not None
    assert ppo.action_space is not None
    assert isinstance(ppo.actor, nn.Module)
    assert isinstance(ppo.critic, nn.Module)
    assert ppo.lr is not None
    assert str(ppo.actor.to("cpu").state_dict()) == str(ppo.actor.state_dict())
    assert str(ppo.critic.to("cpu").state_dict()) == str(ppo.critic.state_dict())
    assert ppo.batch_size is not None
    assert ppo.gamma is not None
    assert ppo.mut is None
    assert ppo.index is not None
    assert ppo.scores is not None
    assert ppo.fitness is not None
    assert ppo.steps is not None
    ppo.clean_up()


def test_collect_rollouts_populates_buffer(vector_space, discrete_space):
    """collect_rollouts fills the agent rollout buffer without error."""
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        learn_step=5,
        num_envs=1,
    )
    env = DummyEnv(state_size=vector_space.shape, vect=True, num_envs=1)
    collect_rollouts(ppo, env, n_steps=5)
    assert ppo.rollout_buffer.pos > 0 or ppo.rollout_buffer.full
    ppo.clean_up()


def test_ppo_init_negative_target_kl_assert(vector_space, discrete_space):
    with pytest.raises(
        AssertionError,
        match="Target KL divergence threshold must be greater than or equal to zero",
    ):
        PPO(
            observation_space=vector_space,
            action_space=discrete_space,
            target_kl=-1.0,
        )


def test_ppo_init_head_config_path(vector_space, discrete_space):
    net_config = {
        "head_config": {"hidden_size": [8], "output_activation": "Tanh"},
        "squash_output": True,
    }
    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        net_config=net_config,
    )
    assert ppo.critic.head_net.net_config["output_activation"] is None
    ppo.clean_up()


def test_share_encoder_parameters_warns_for_non_evolvable(vector_space, discrete_space):
    ppo = PPO(vector_space, discrete_space)
    ppo.actor = nn.Linear(4, 2)
    ppo.critic = nn.Linear(4, 1)
    with pytest.warns(UserWarning, match="Encoder sharing is disabled"):
        ppo.share_encoder_parameters()
    ppo.clean_up()


def test_evaluate_actions_uses_negative_log_prob_when_entropy_none(
    vector_space,
    discrete_space,
    monkeypatch,
):
    ppo = PPO(vector_space, discrete_space)
    monkeypatch.setattr(
        ppo,
        "_get_action_and_values",
        lambda *args, **kwargs: (
            torch.zeros(1),
            torch.zeros(1),
            None,
            torch.zeros(1),
            None,
        ),
    )
    monkeypatch.setattr(
        ppo.actor,
        "action_log_prob",
        lambda actions: torch.tensor([2.0]),
    )
    log_prob, entropy, _ = ppo.evaluate_actions(
        obs=np.zeros((1, *vector_space.shape), dtype=np.float32),
        actions=torch.tensor([0]),
    )
    assert log_prob.item() == 2.0
    assert entropy.item() == -2.0
    ppo.clean_up()


def test_get_action_clips_unsquashed_box_actions(vector_space):
    box_action = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    ppo = PPO(vector_space, box_action)
    ppo.set_training_mode(False)
    ppo.actor.squash_output = False

    ppo._get_action_and_values = lambda *args, **kwargs: (
        torch.tensor([[5.0, -5.0]]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        None,
    )
    action, *_ = ppo.get_action(np.zeros((1, vector_space.shape[0]), dtype=np.float32))
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)
    ppo.clean_up()


def test_rollout_buffer_flat_external_empty_returns_zero(vector_space, discrete_space):
    ppo = PPO(vector_space, discrete_space)
    empty_td = TensorDict({}, batch_size=[0])
    with pytest.warns(UserWarning, match="Buffer data is empty"):
        loss = ppo._learn_from_rollout_buffer_flat(buffer_td_external=empty_td)
    assert loss == 0.0
    ppo.clean_up()


def test_rollout_buffer_flat_external_uses_accelerator_and_early_stops(
    vector_space,
    discrete_space,
    monkeypatch,
):
    class DummyAccel:
        def __init__(self):
            self.calls = 0

        def backward(self, loss):
            self.calls += 1
            loss.backward()

    ppo = PPO(
        vector_space,
        discrete_space,
        target_kl=1e-6,
        update_epochs=2,
        batch_size=2,
    )
    ppo.accelerator = DummyAccel()
    ppo.rollout_buffer = type("RB", (), {"size": lambda self: 4})()

    td = TensorDict(
        {
            "observations": torch.randn(4, *vector_space.shape),
            "actions": torch.randint(0, discrete_space.n, (4, 1)),
            "log_probs": torch.zeros(4),
            "advantages": torch.ones(4),
            "returns": torch.zeros(4),
            "values": torch.zeros(4),
        },
        batch_size=[4],
    )

    monkeypatch.setattr(
        ppo,
        "evaluate_actions",
        lambda obs, actions, hidden_state=None, action_mask=None: (
            torch.full((actions.shape[0],), 10.0, requires_grad=True),
            torch.ones(actions.shape[0], requires_grad=True),
            torch.zeros(actions.shape[0], requires_grad=True),
        ),
    )
    loss = ppo._learn_from_rollout_buffer_flat(buffer_td_external=td)
    assert isinstance(loss, float)
    assert ppo.accelerator.calls > 0
    ppo.clean_up()


def test_rollout_buffer_bptt_kl_warning_and_break_paths(
    vector_space,
    discrete_space,
    monkeypatch,
):
    class DummyAccel:
        def __init__(self):
            self.calls = 0

        def backward(self, loss):
            self.calls += 1
            loss.backward()

    class FakeRolloutBuffer:
        def __init__(self):
            self.capacity = 1
            self.pos = 1
            self.full = False
            self.buffer = TensorDict(
                {"advantages": torch.ones(1, 2)}, batch_size=[1, 2]
            )

        def prepare_sequence_tensors(self, device="cpu"):
            return None

        def get_minibatch_sequences(self, batch_size=1):
            padded = TensorDict(
                {
                    "observations": torch.randn(2, *vector_space.shape),
                    "actions": torch.randint(0, discrete_space.n, (2, 1)),
                    "pad_mask": torch.tensor([True, True]),
                },
                batch_size=[2],
            )
            unpadded = TensorDict(
                {
                    "log_probs": torch.zeros(2),
                    "advantages": torch.ones(2),
                    "values": torch.zeros(2),
                    "returns": torch.zeros(2),
                },
                batch_size=[2],
            )
            return [(padded, unpadded)]

    ppo = PPO(
        vector_space,
        discrete_space,
        recurrent=True,
        target_kl=0.5,
        update_epochs=2,
        batch_size=1,
    )
    ppo.rollout_buffer = FakeRolloutBuffer()
    ppo.accelerator = DummyAccel()

    monkeypatch.setattr(
        ppo,
        "evaluate_actions",
        lambda obs, actions, hidden_state=None, action_mask=None: (
            torch.full((actions.shape[0],), 10.0, requires_grad=True),
            torch.ones(actions.shape[0], requires_grad=True),
            torch.zeros(actions.shape[0], requires_grad=True),
        ),
    )

    with pytest.warns(UserWarning, match="KL divergence .* exceeded target"):
        loss = ppo._learn_from_rollout_buffer_bptt()
    assert isinstance(loss, float)
    assert ppo.accelerator.calls > 0
    ppo.clean_up()


def test_rollout_buffer_bptt_epoch_avg_kl_warning_branch(
    vector_space,
    discrete_space,
    monkeypatch,
):
    class FakeRolloutBuffer:
        def __init__(self):
            self.capacity = 1
            self.pos = 1
            self.full = False
            self.buffer = TensorDict(
                {"advantages": torch.ones(1, 2)}, batch_size=[1, 2]
            )

        def prepare_sequence_tensors(self, device="cpu"):
            return None

        def get_minibatch_sequences(self, batch_size=1):
            padded = TensorDict(
                {
                    "observations": torch.randn(2, *vector_space.shape),
                    "actions": torch.randint(0, discrete_space.n, (2, 1)),
                    "pad_mask": torch.tensor([True, True]),
                },
                batch_size=[2],
            )
            unpadded = TensorDict(
                {
                    "log_probs": torch.zeros(2),
                    "advantages": torch.ones(2),
                    "values": torch.zeros(2),
                    "returns": torch.zeros(2),
                },
                batch_size=[2],
            )
            return [(padded, unpadded)]

    ppo = PPO(
        vector_space,
        discrete_space,
        recurrent=True,
        target_kl=0.5,
        update_epochs=1,
        batch_size=1,
    )
    ppo.rollout_buffer = FakeRolloutBuffer()

    monkeypatch.setattr(
        ppo,
        "evaluate_actions",
        lambda obs, actions, hidden_state=None, action_mask=None: (
            torch.zeros(actions.shape[0], requires_grad=True),
            torch.ones(actions.shape[0], requires_grad=True),
            torch.zeros(actions.shape[0], requires_grad=True),
        ),
    )
    values = iter([0.1, 1.0, 0.1])  # minibatch KL, epoch avg KL, latest minibatch KL
    monkeypatch.setattr(
        "agilerl.algorithms.ppo.np.mean", lambda *_args, **_kwargs: next(values)
    )

    with pytest.warns(UserWarning, match="Average KL divergence .* exceeded target"):
        _ = ppo._learn_from_rollout_buffer_bptt()
    ppo.clean_up()


def test_get_action_and_values_share_encoders_false(vector_space, discrete_space):
    ppo = PPO(
        vector_space,
        discrete_space,
        share_encoders=False,
    )
    obs = np.zeros((1, *vector_space.shape), dtype=np.float32)
    action, log_prob, entropy, values, next_hidden = ppo._get_action_and_values(
        obs, sample=True
    )
    assert action is not None
    assert values is not None
    assert next_hidden is None
    ppo.clean_up()


def test_ppo_test_loop_masks_callbacks_and_non_vectorized_paths(
    vector_space, discrete_space
):
    class VecEnv:
        def __init__(self):
            self.num_envs = 2
            self._steps = 0

        def reset(self):
            self._steps = 0
            obs = np.zeros((self.num_envs, *vector_space.shape), dtype=np.float32)
            info = [{"action_mask": np.array([1, 0])}, {"action_mask": None}]
            return obs, info

        def step(self, _action):
            self._steps += 1
            obs = np.zeros((self.num_envs, *vector_space.shape), dtype=np.float32)
            reward = np.array([1.0, 1.0])
            done = (
                np.array([False, False]) if self._steps == 1 else np.array([True, True])
            )
            trunc = np.array([False, False])
            info = {"action_mask": np.array([[1, 0], [0, 1]])}
            return obs, reward, done, trunc, info

    class NonVecEnv:
        def __init__(self):
            self._done = False

        def reset(self):
            self._done = False
            return np.zeros(vector_space.shape, dtype=np.float32), {
                "action_mask": np.array([1, 0])
            }

        def step(self, _action):
            self._done = True
            return (
                np.zeros(vector_space.shape, dtype=np.float32),
                1.0,
                True,
                False,
                {"k": 1.0},
            )

    ppo = PPO(vector_space, discrete_space)
    callback_calls = []
    with pytest.warns(
        UserWarning, match="Action masks not provided for all vectorized environments"
    ):
        _ = ppo.test(
            VecEnv(),
            swap_channels=True,
            vectorized=True,
            loop=1,
            callback=lambda score, info: callback_calls.append((score, info)),
        )
    assert len(callback_calls) == 1
    assert isinstance(callback_calls[0][1], dict)

    callback_calls.clear()
    _ = ppo.test(
        NonVecEnv(),
        vectorized=False,
        loop=1,
        callback=lambda score, info: callback_calls.append((score, info)),
    )
    assert len(callback_calls) == 1
    assert "final_info" in callback_calls[0][1]
    ppo.clean_up()


@pytest.mark.parametrize(
    "action_space",
    ["discrete_space", "multidiscrete_space"],
)
def test_action_masks_applied_during_learning(action_space, vector_space, request):
    """Verify that action masks stored in the rollout buffer are used during learning.

    This ensures that log_probs computed during evaluate_actions (in learn) use the
    same masked distribution as get_action, preventing the PPO ratio from being biased
    by probability mass assigned to illegal actions.
    """
    action_space = request.getfixturevalue(action_space)
    learn_step = 32
    batch_size = 16

    ppo = PPO(
        observation_space=vector_space,
        action_space=action_space,
        learn_step=learn_step,
        batch_size=batch_size,
    )

    # Determine mask size based on action space
    if isinstance(action_space, spaces.Discrete):
        mask_size = action_space.n
    elif isinstance(action_space, spaces.MultiDiscrete):
        mask_size = int(sum(action_space.nvec))
    else:
        pytest.skip("Action masks only apply to discrete action spaces")

    # Fill the buffer with experiences that include action masks
    for i in range(learn_step):
        obs = np.array([vector_space.sample()], dtype=np.float32)
        next_obs = np.array([vector_space.sample()], dtype=np.float32)

        # Create a mask that blocks some actions
        action_mask = np.ones(mask_size, dtype=bool)
        action_mask[0] = False  # Block the first action

        # Get action with the mask applied
        action, log_prob, _, value = ppo.get_action(obs, action_mask=action_mask)

        ppo.rollout_buffer.add(
            obs=obs,
            action=action,
            reward=1.0,
            done=i == learn_step - 1,
            value=float(value),
            log_prob=float(log_prob),
            next_obs=next_obs,
            action_mask=np.array([action_mask]),
        )

    # Compute returns and advantages
    last_value = torch.zeros((ppo.num_envs, 1), device=ppo.device)
    last_done = torch.zeros((ppo.num_envs, 1), device=ppo.device)
    ppo.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

    # Verify action masks are stored in the buffer
    buffer_td = ppo.rollout_buffer.get_tensor_batch(device=ppo.device)
    assert "action_masks" in buffer_td.keys(), (
        "action_masks should be stored in rollout buffer"
    )

    # Compare log_probs: evaluate_actions with mask should match get_action with mask
    sample_obs = buffer_td["observations"][:4]
    sample_actions = buffer_td["actions"][:4]
    sample_masks = buffer_td["action_masks"][:4]

    if isinstance(action_space, spaces.Discrete):
        sample_actions = sample_actions.squeeze(-1)

    # Evaluate with mask (correct behavior)
    with torch.no_grad():
        log_prob_masked, _, _ = ppo.evaluate_actions(
            obs=sample_obs,
            actions=sample_actions,
            action_mask=sample_masks,
        )
        # Evaluate without mask (old buggy behavior)
        log_prob_unmasked, _, _ = ppo.evaluate_actions(
            obs=sample_obs,
            actions=sample_actions,
        )

    # The masked log_probs should differ from unmasked ones because the masked
    # distribution renormalizes probability mass over legal actions only.
    # If they were the same, that would mean masking had no effect during learning.
    assert not torch.allclose(log_prob_masked, log_prob_unmasked, atol=1e-6), (
        "Action masks should affect log_probs during evaluate_actions. "
        "Masked and unmasked log_probs should differ."
    )

    # Learn should complete without errors when masks are in the buffer
    loss = ppo.learn()
    assert isinstance(loss, float)
    ppo.clean_up()


@pytest.mark.parametrize("recurrent", [True, False])
def test_action_masks_in_rollout_buffer_learn(vector_space, discrete_space, recurrent):
    """Test that the full learn pipeline works with action masks in the rollout buffer."""
    learn_step = 32
    batch_size = 16

    net_config = {"encoder_config": {"hidden_state_size": 32}} if recurrent else {}

    ppo = PPO(
        observation_space=vector_space,
        action_space=discrete_space,
        learn_step=learn_step,
        batch_size=batch_size,
        recurrent=recurrent,
        net_config=net_config,
    )

    mask_size = discrete_space.n
    hidden_state = ppo.get_initial_hidden_state() if recurrent else None

    for i in range(learn_step):
        obs = np.array([vector_space.sample()], dtype=np.float32)
        next_obs = np.array([vector_space.sample()], dtype=np.float32)

        # Alternate which action is blocked
        action_mask = np.ones(mask_size, dtype=bool)
        action_mask[i % mask_size] = False

        if recurrent:
            action, log_prob, _, value, hidden_state = ppo.get_action(
                obs, action_mask=action_mask, hidden_state=hidden_state
            )
        else:
            action, log_prob, _, value = ppo.get_action(obs, action_mask=action_mask)

        add_kwargs = {
            "obs": obs,
            "action": action,
            "reward": 1.0,
            "done": i == learn_step - 1,
            "value": float(value),
            "log_prob": float(log_prob),
            "next_obs": next_obs,
            "action_mask": np.array([action_mask]),
        }
        if recurrent:
            add_kwargs["hidden_state"] = hidden_state
        ppo.rollout_buffer.add(**add_kwargs)

    last_value = torch.zeros((ppo.num_envs, 1), device=ppo.device)
    last_done = torch.zeros((ppo.num_envs, 1), device=ppo.device)
    ppo.rollout_buffer.compute_returns_and_advantages(last_value, last_done)

    loss = ppo.learn()
    assert isinstance(loss, float)
    ppo.clean_up()


def test_continuous_action_space_no_mask_buffer(vector_space):
    """Verify that continuous action spaces do not allocate action mask buffers."""
    continuous_action_space = vector_space  # Box space

    ppo = PPO(
        observation_space=vector_space,
        action_space=continuous_action_space,
        learn_step=16,
        batch_size=8,
    )

    assert "action_masks" not in ppo.rollout_buffer.buffer.keys(), (
        "Continuous action spaces should not have action_masks in the buffer"
    )
    ppo.clean_up()
