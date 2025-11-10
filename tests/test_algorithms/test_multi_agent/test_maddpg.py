import copy
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv
from torch._dynamo import OptimizedModule

from agilerl.algorithms.maddpg import MADDPG
from agilerl.modules import EvolvableCNN, EvolvableMLP, EvolvableMultiInput, ModuleDict
from agilerl.modules.custom_components import GumbelSoftmax
from agilerl.networks.actors import DeterministicActor
from agilerl.networks.q_networks import ContinuousQNetwork
from agilerl.utils.algo_utils import concatenate_spaces
from agilerl.utils.evolvable_networks import get_default_encoder_config
from agilerl.utils.utils import make_multi_agent_vect_envs
from agilerl.wrappers.make_evolvable import MakeEvolvable
from tests.helper_functions import assert_not_equal_state_dict, assert_state_dicts_equal


class DummyMultiEnv(ParallelEnv):
    def __init__(self, observation_spaces, action_spaces):
        self.observation_spaces = observation_spaces
        self.action_spaces = action_spaces
        self.agents = ["agent_0", "agent_1", "other_agent_0"]
        self.possible_agents = ["agent_0", "agent_1", "other_agent_0"]
        self.metadata = None
        self.render_mode = None

    def action_space(self, agent):
        return Discrete(self.action_spaces[0].n)

    def observation_space(self, agent):
        return Box(0, 1, self.observation_spaces.shape)

    def reset(self, seed=None, options=None):
        return {
            agent: np.random.rand(*self.observation_spaces.shape)
            for agent in self.agents
        }, {
            "agent_0": {"env_defined_actions": np.array([1])},
            "agent_1": {"env_defined_actions": np.array([1])},
            "other_agent_0": {"env_defined_actions": None},
        }

    def step(self, action):
        return (
            {
                agent: np.random.rand(*self.observation_spaces.shape)
                for agent in self.agents
            },
            {agent: np.random.randint(0, 5) for agent in self.agents},
            {agent: 1 for agent in self.agents},
            {agent: np.random.randint(0, 2) for agent in self.agents},
            self.reset()[1],
        )


class MultiAgentCNNActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=3, out_channels=16, kernel_size=(1, 3, 3), stride=4
        )
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=2
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(288, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()
        self.output_activation = GumbelSoftmax()

    def forward(self, state_tensor):
        x = self.relu(self.conv1(state_tensor))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.output_activation(self.fc2(x))

        return x


class MultiAgentCNNCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=3, out_channels=16, kernel_size=(2, 3, 3), stride=4
        )
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=2
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(290, 256)
        self.fc2 = nn.Linear(256, 2)
        self.relu = nn.ReLU()

    def forward(self, state_tensor, action_tensor):
        x = self.relu(self.conv1(state_tensor))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action_tensor], dim=1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class DummyContinuousQNetwork(ContinuousQNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def no_sync(self):
        class DummyNoSync:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass  # Add cleanup or handling if needed

        return DummyNoSync()


class DummyDeterministicActor(DeterministicActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def no_sync(self):
        class DummyNoSync:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                pass  # Add cleanup or handling if needed

        return DummyNoSync()


@pytest.fixture(scope="function")
def mlp_actor(observation_spaces, action_spaces, request):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    return nn.Sequential(
        nn.Linear(observation_spaces[0].shape[0], 64),
        nn.ReLU(),
        nn.Linear(64, action_spaces[0].n),
        GumbelSoftmax(),
    )


@pytest.fixture(scope="function")
def mlp_critic(action_spaces, observation_spaces, request):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    return nn.Sequential(
        nn.Linear(observation_spaces[0].shape[0] + action_spaces[0].n, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )


@pytest.fixture(scope="function")
def cnn_actor():
    return MultiAgentCNNActor()


@pytest.fixture(scope="function")
def cnn_critic():
    return MultiAgentCNNCritic()


@pytest.fixture
def mocked_accelerator():
    MagicMock(spec=Accelerator)


@pytest.fixture(scope="function")
def accelerated_experiences(
    batch_size, observation_spaces, action_spaces, agent_ids, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    one_hot = all(isinstance(space, Discrete) for space in observation_spaces)
    discrete_actions = all(isinstance(space, Discrete) for space in action_spaces)
    state_size = (
        observation_spaces[0].shape if not one_hot else (observation_spaces[0].n,)
    )
    action_size = action_spaces[0].n if discrete_actions else action_spaces[0].shape[0]
    if one_hot:
        states = {
            agent: torch.randint(0, state_size[0], (batch_size, 1)).float()
            for agent in agent_ids
        }
    else:
        states = {agent: torch.randn(batch_size, *state_size) for agent in agent_ids}

    actions = {agent: torch.randn(batch_size, action_size) for agent in agent_ids}
    rewards = {agent: torch.randn(batch_size, 1) for agent in agent_ids}
    dones = {agent: torch.randint(0, 2, (batch_size, 1)) for agent in agent_ids}
    if one_hot:
        next_states = {
            agent: torch.randint(0, state_size[0], (batch_size, 1)).float()
            for agent in agent_ids
        }
    else:
        next_states = {
            agent: torch.randn(batch_size, *state_size) for agent in agent_ids
        }

    return states, actions, rewards, next_states, dones


@pytest.fixture(scope="function")
def experiences(
    batch_size, observation_spaces, action_spaces, agent_ids, device, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    one_hot = all(isinstance(space, Discrete) for space in observation_spaces)
    discrete_actions = all(isinstance(space, Discrete) for space in action_spaces)
    state_size = (
        observation_spaces[0].shape if not one_hot else (observation_spaces[0].n,)
    )
    action_size = action_spaces[0].n if discrete_actions else action_spaces[0].shape[0]
    if one_hot:
        states = {
            agent: torch.randint(0, state_size[0], (batch_size, 1)).float().to(device)
            for agent in agent_ids
        }
    else:
        states = {
            agent: torch.randn(batch_size, *state_size).to(device)
            for agent in agent_ids
        }

    actions = {
        agent: torch.randn(batch_size, action_size).to(device) for agent in agent_ids
    }
    rewards = {agent: torch.randn(batch_size, 1).to(device) for agent in agent_ids}
    dones = {
        agent: torch.randint(0, 2, (batch_size, 1)).to(device) for agent in agent_ids
    }
    if one_hot:
        next_states = {
            agent: torch.randint(0, state_size[0], (batch_size, 1)).float().to(device)
            for agent in agent_ids
        }
    else:
        next_states = {
            agent: torch.randn(batch_size, *state_size).to(device)
            for agent in agent_ids
        }

    return states, actions, rewards, next_states, dones


@pytest.mark.parametrize(
    "observation_spaces",
    [
        "ma_vector_space",
        "ma_image_space",
        "ma_dict_space",
        "ma_multidiscrete_space",
    ],
)
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_maddpg_with_net_config(
    accelerator_flag, observation_spaces, ma_vector_space, device, compile_mode, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    net_config = {
        "encoder_config": get_default_encoder_config(observation_spaces[0]),
        "head_config": {"hidden_size": [16]},
    }
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    expl_noise = 0.1
    batch_size = 64
    accelerator = Accelerator() if accelerator_flag else None

    # Initialize MADDPG
    maddpg = MADDPG(
        observation_spaces=observation_spaces,
        net_config=net_config,
        action_spaces=ma_vector_space,
        agent_ids=agent_ids,
        accelerator=accelerator,
        device=device,
        torch_compiler=compile_mode,
    )

    # Check if the MADDPG agent is initialized correctly
    net_config.update({"output_activation": "Softmax"})
    assert maddpg.observation_spaces == observation_spaces
    assert maddpg.action_spaces == ma_vector_space
    assert maddpg.n_agents == len(agent_ids)
    assert maddpg.agent_ids == agent_ids
    for noise_vec in maddpg.expl_noise.values():
        assert torch.all(noise_vec == expl_noise)
    assert maddpg.batch_size == batch_size
    assert maddpg.scores == []
    assert maddpg.fitness == []
    assert maddpg.steps == [0]

    if compile_mode is not None and accelerator is None:
        assert all(
            isinstance(actor, OptimizedModule) for actor in maddpg.actors.values()
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in maddpg.critics.values()
        )
    else:
        for agent_id in maddpg.agent_ids:
            actor = maddpg.actors[agent_id]
            critic = maddpg.critics[agent_id]
            assert isinstance(actor, DeterministicActor)
            assert isinstance(critic, ContinuousQNetwork)

    expected_optimizer_cls = optim.Adam if accelerator is None else AcceleratedOptimizer
    for agent_id, actor_optimizer in maddpg.actor_optimizers.items():
        critic_optimizer = maddpg.critic_optimizers[agent_id]
        assert isinstance(actor_optimizer, expected_optimizer_cls)
        assert isinstance(critic_optimizer, expected_optimizer_cls)

    assert isinstance(maddpg.criterion, nn.MSELoss)


# TODO: This will be deprecated in the future
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("observation_spaces", ["ma_vector_space"])
@pytest.mark.parametrize("action_spaces", ["ma_discrete_space"])
def test_initialize_maddpg_with_mlp_networks(
    mlp_actor,
    mlp_critic,
    observation_spaces,
    action_spaces,
    request,
    accelerator_flag,
    device,
    compile_mode,
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    accelerator = Accelerator() if accelerator_flag else None
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    evo_actors = ModuleDict(
        {
            agent_id: MakeEvolvable(
                network=mlp_actor, input_tensor=torch.randn(1, 6), device=device
            )
            for agent_id in agent_ids
        }
    )
    evo_critics = ModuleDict(
        {
            agent_id: MakeEvolvable(
                network=mlp_critic, input_tensor=torch.randn(1, 8), device=device
            )
            for agent_id in agent_ids
        }
    )
    maddpg = MADDPG(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )

    if compile_mode is not None and accelerator is None:
        assert all(
            isinstance(actor, OptimizedModule) for actor in maddpg.actors.values()
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in maddpg.critics.values()
        )
    else:
        for agent_id in maddpg.agent_ids:
            actor = maddpg.actors[agent_id]
            critic = maddpg.critics[agent_id]
            assert isinstance(actor, MakeEvolvable)
            assert isinstance(critic, MakeEvolvable)

    assert maddpg.observation_spaces == observation_spaces
    assert maddpg.action_spaces == action_spaces
    assert maddpg.n_agents == len(agent_ids)
    assert maddpg.agent_ids == agent_ids
    assert maddpg.scores == []
    assert maddpg.fitness == []
    assert maddpg.steps == [0]
    expected_optimizer_cls = optim.Adam if accelerator is None else AcceleratedOptimizer
    assert all(
        isinstance(actor_optimizer, expected_optimizer_cls)
        for actor_optimizer in maddpg.actor_optimizers.values()
    )
    assert all(
        isinstance(critic_optimizer, expected_optimizer_cls)
        for critic_optimizer in maddpg.critic_optimizers.values()
    )
    assert isinstance(maddpg.criterion, nn.MSELoss)


@pytest.mark.parametrize("observation_spaces", ["ma_vector_space"])
@pytest.mark.parametrize("action_spaces", ["ma_discrete_space"])
def test_initialize_maddpg_with_mlp_networks_gumbel_softmax(
    mlp_actor,
    mlp_critic,
    observation_spaces,
    action_spaces,
    device,
    request,
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    compile_mode = "reduce-overhead"
    net_config = {
        "encoder_config": {
            "hidden_size": [64, 64],
            "min_hidden_layers": 1,
            "max_hidden_layers": 3,
            "min_mlp_nodes": 64,
            "max_mlp_nodes": 500,
            "output_activation": "GumbelSoftmax",
            "activation": "ReLU",
        }
    }
    maddpg = MADDPG(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        net_config=net_config,
        device=device,
        torch_compiler=compile_mode,
    )
    assert maddpg.torch_compiler == "default"


# TODO: This will be deprecated in the future
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_maddpg_with_cnn_networks(
    cnn_actor,
    cnn_critic,
    accelerator_flag,
    device,
    compile_mode,
    ma_image_space,
    ma_discrete_space,
):
    accelerator = Accelerator() if accelerator_flag else None
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    evo_actors = ModuleDict(
        {
            agent_id: MakeEvolvable(
                network=cnn_actor,
                input_tensor=torch.randn(1, 3, 1, 32, 32),
                device=device,
            )
            for agent_id in agent_ids
        }
    )
    evo_critics = ModuleDict(
        {
            agent_id: MakeEvolvable(
                network=cnn_critic,
                input_tensor=torch.randn(1, 3, 3, 32, 32),
                secondary_input_tensor=torch.randn(1, 2),
                device=device,
            )
            for agent_id in agent_ids
        }
    )
    maddpg = MADDPG(
        observation_spaces=ma_image_space,
        action_spaces=ma_discrete_space,
        agent_ids=agent_ids,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )
    if compile_mode is not None and accelerator is None:
        assert all(
            isinstance(actor, OptimizedModule) for actor in maddpg.actors.values()
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in maddpg.critics.values()
        )
    else:
        for agent_id in maddpg.agent_ids:
            actor = maddpg.actors[agent_id]
            critic = maddpg.critics[agent_id]
            assert isinstance(actor, MakeEvolvable)
            assert isinstance(critic, MakeEvolvable)

    assert maddpg.observation_spaces == ma_image_space
    assert maddpg.action_spaces == ma_discrete_space
    assert maddpg.n_agents == len(agent_ids)
    assert maddpg.agent_ids == agent_ids
    assert maddpg.scores == []
    assert maddpg.fitness == []
    assert maddpg.steps == [0]
    expected_optimizer_cls = optim.Adam if accelerator is None else AcceleratedOptimizer
    assert all(
        isinstance(actor_optimizer, expected_optimizer_cls)
        for actor_optimizer in maddpg.actor_optimizers.values()
    )
    assert all(
        isinstance(critic_optimizer, expected_optimizer_cls)
        for critic_optimizer in maddpg.critic_optimizers.values()
    )
    assert isinstance(maddpg.criterion, nn.MSELoss)


@pytest.mark.parametrize("accelerator", [None, Accelerator()])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize(
    "observation_spaces, encoder_cls",
    [
        ("ma_vector_space", EvolvableMLP),
        ("ma_image_space", EvolvableCNN),
    ],
)
def test_initialize_maddpg_with_evo_networks(
    observation_spaces,
    encoder_cls,
    device,
    compile_mode,
    accelerator,
    ma_discrete_space,
    request,
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    observation_spaces = request.getfixturevalue(observation_spaces)
    observation_space = spaces.Dict(
        {agent_id: observation_spaces[idx] for idx, agent_id in enumerate(agent_ids)}
    )

    evo_actors = ModuleDict(
        {
            agent_id: DeterministicActor(
                observation_spaces[i],
                ma_discrete_space[i],
                device=device,
            )
            for i, agent_id in enumerate(agent_ids)
        }
    )
    evo_critics = ModuleDict(
        {
            agent_id: ContinuousQNetwork(
                observation_space=observation_space,
                action_space=concatenate_spaces(ma_discrete_space),
                device=device,
            )
            for agent_id in agent_ids
        }
    )

    maddpg = MADDPG(
        observation_spaces=observation_spaces,
        action_spaces=ma_discrete_space,
        agent_ids=agent_ids,
        actor_networks=evo_actors,
        critic_networks=evo_critics,
        device=device,
        torch_compiler=compile_mode,
        accelerator=accelerator,
    )
    if compile_mode is not None and accelerator is None:
        assert all(
            isinstance(actor, OptimizedModule) for actor in maddpg.actors.values()
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in maddpg.critics.values()
        )
    else:
        for agent_id in maddpg.agent_ids:
            actor = maddpg.actors[agent_id]
            critic = maddpg.critics[agent_id]
            assert isinstance(actor.encoder, encoder_cls)
            assert isinstance(critic.encoder, EvolvableMultiInput)

    assert maddpg.observation_spaces == observation_spaces
    assert maddpg.action_spaces == ma_discrete_space
    assert maddpg.n_agents == len(agent_ids)
    assert maddpg.agent_ids == agent_ids
    assert maddpg.scores == []
    assert maddpg.fitness == []
    assert maddpg.steps == [0]

    expected_optimizer_cls = optim.Adam if accelerator is None else AcceleratedOptimizer
    assert all(
        isinstance(actor_optimizer, expected_optimizer_cls)
        for actor_optimizer in maddpg.actor_optimizers.values()
    )
    assert all(
        isinstance(critic_optimizer, expected_optimizer_cls)
        for critic_optimizer in maddpg.critic_optimizers.values()
    )
    assert isinstance(maddpg.criterion, nn.MSELoss)


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_initialize_maddpg_with_incorrect_evo_networks(
    compile_mode, ma_vector_space, ma_discrete_space
):
    evo_actors = []
    evo_critics = []
    with pytest.raises(AssertionError):
        maddpg = MADDPG(
            observation_spaces=ma_vector_space,
            action_spaces=ma_discrete_space,
            agent_ids=["agent_0", "agent_1", "other_agent_0"],
            actor_networks=evo_actors,
            critic_networks=evo_critics,
            torch_compiler=compile_mode,
        )
        assert maddpg


@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("observation_spaces", ["ma_vector_space"])
@pytest.mark.parametrize("action_spaces", ["ma_discrete_space"])
def test_maddpg_init_warning(
    mlp_actor, device, compile_mode, observation_spaces, action_spaces, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    warning_string = "Actor and critic network must both be supplied to use custom networks. Defaulting to net config."
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    evo_actors = [
        MakeEvolvable(network=mlp_actor, input_tensor=torch.randn(1, 6), device=device)
        for _ in range(len(agent_ids))
    ]
    with pytest.warns(UserWarning, match=warning_string):
        MADDPG(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=agent_ids,
            actor_networks=evo_actors,
            device=device,
            torch_compiler=compile_mode,
        )


@pytest.mark.parametrize(
    "mode", (None, 0, False, "default", "reduce-overhead", "max-autotune")
)
def test_maddpg_init_torch_compiler_no_error(
    mode, ma_vector_space, ma_discrete_space, device
):
    maddpg = MADDPG(
        observation_spaces=ma_vector_space,
        action_spaces=ma_discrete_space,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        device=device,
        torch_compiler=mode,
    )
    if isinstance(mode, str):
        assert all(
            isinstance(actor, OptimizedModule) for actor in maddpg.actors.values()
        )
        assert all(
            isinstance(critic, OptimizedModule) for critic in maddpg.critics.values()
        )
        assert all(
            isinstance(actor_target, OptimizedModule)
            for actor_target in maddpg.actor_targets.values()
        )
        assert all(
            isinstance(critic_target, OptimizedModule)
            for critic_target in maddpg.critic_targets.values()
        )
        assert maddpg.torch_compiler == "default"
    else:
        assert isinstance(maddpg, MADDPG)


@pytest.mark.parametrize("mode", (1, True, "max-autotune-no-cudagraphs"))
def test_maddpg_init_torch_compiler_error(
    mode, ma_vector_space, ma_discrete_space, device
):
    err_string = (
        "Choose between torch compiler modes: "
        "default, reduce-overhead, max-autotune or None"
    )
    with pytest.raises(AssertionError, match=err_string):
        MADDPG(
            observation_spaces=ma_vector_space,
            action_spaces=ma_discrete_space,
            agent_ids=["agent_0", "agent_1", "other_agent_0"],
            device=device,
            torch_compiler=mode,
        )


@pytest.mark.parametrize(
    "observation_spaces", ["ma_vector_space", "ma_discrete_space", "ma_image_space"]
)
@pytest.mark.parametrize("action_spaces", ["ma_vector_space", "ma_discrete_space"])
@pytest.mark.parametrize("training", [0, 1])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_maddpg_get_action(
    training, observation_spaces, action_spaces, device, compile_mode, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    if all(isinstance(space, spaces.Discrete) for space in observation_spaces):
        state = {
            agent: np.random.randint(0, observation_spaces[idx].n, 1)
            for idx, agent in enumerate(agent_ids)
        }
    else:
        state = {
            agent: np.random.randn(*observation_spaces[idx].shape)
            for idx, agent in enumerate(agent_ids)
        }

    maddpg = MADDPG(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )
    maddpg.set_training_mode(bool(training))
    processed_action, raw_action = maddpg.get_action(state)
    discrete_actions = all(
        isinstance(space, spaces.Discrete) for space in action_spaces
    )
    for idx, env_actions in enumerate(list(raw_action.values())):
        action_dim = (
            action_spaces[idx].shape[0]
            if isinstance(action_spaces[idx], spaces.Box)
            else action_spaces[idx].n
        )
        for action in env_actions:
            assert len(action) == action_dim
            if discrete_actions:
                torch.testing.assert_close(
                    sum(action),
                    1.0,
                    atol=0.1,
                    rtol=1e-3,
                )
            assert action.dtype == np.float32
            assert -1 <= action.all() <= 1

    if discrete_actions:
        for idx, env_action in enumerate(list(processed_action.values())):
            for action in env_action:
                assert action <= action_spaces[idx].n - 1
    maddpg = None


@pytest.mark.parametrize("training", [False, True])
def test_maddpg_get_action_action_masking_exception(
    training, device, ma_vector_space, ma_discrete_space
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {
        agent: {
            "observation": np.random.randn(*ma_vector_space[idx].shape),
            "action_mask": [0, 1, 0, 1],
        }
        for idx, agent in enumerate(agent_ids)
    }
    maddpg = MADDPG(
        ma_vector_space,
        ma_discrete_space,
        agent_ids=agent_ids,
        device=device,
    )
    with pytest.raises(AssertionError):
        maddpg.set_training_mode(training)
        _, raw_action = maddpg.get_action(state)


@pytest.mark.parametrize("training", [False, True])
def test_maddpg_get_action_action_masking(
    training, device, ma_vector_space, ma_discrete_space
):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {
        agent: np.random.randn(*ma_vector_space[idx].shape)
        for idx, agent in enumerate(agent_ids)
    }
    info = {
        agent: {
            "action_mask": [0, 1],
        }
        for idx, agent in enumerate(agent_ids)
    }
    maddpg = MADDPG(
        ma_vector_space,
        ma_discrete_space,
        agent_ids=agent_ids,
        device=device,
    )
    maddpg.set_training_mode(training)
    action, _ = maddpg.get_action(state, info)
    assert all(i in [1, 3] for i in action.values())


@pytest.mark.parametrize("observation_spaces", ["ma_vector_space", "ma_image_space"])
@pytest.mark.parametrize("action_spaces", ["ma_discrete_space", "ma_vector_space"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_get_action_distributed(
    training, observation_spaces, action_spaces, compile_mode, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    accelerator = Accelerator()
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {
        agent: np.random.randn(*observation_spaces[idx].shape)
        for idx, agent in enumerate(agent_ids)
    }
    from agilerl.algorithms.maddpg import MADDPG

    maddpg = MADDPG(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )
    new_actors = ModuleDict(
        {
            agent_id: DummyDeterministicActor(
                observation_space=actor.observation_space,
                action_space=actor.action_space,
                encoder_config=actor.encoder.net_config,
                head_config=actor.head_net.net_config,
                device=actor.device,
            )
            for agent_id, actor in maddpg.actors.items()
        }
    )
    maddpg.actors = new_actors
    maddpg.set_training_mode(training)
    processed_action, raw_action = maddpg.get_action(state)
    discrete_actions = all(
        isinstance(space, spaces.Discrete) for space in action_spaces
    )
    for idx, env_actions in enumerate(list(raw_action.values())):
        action_dim = (
            action_spaces[idx].shape[0]
            if isinstance(action_spaces[idx], spaces.Box)
            else action_spaces[idx].n
        )
        for action in env_actions:
            assert len(action) == action_dim
            if discrete_actions:
                torch.testing.assert_close(
                    sum(action),
                    1.0,
                    atol=0.1,
                    rtol=1e-3,
                )
            assert action.dtype == np.float32
            assert -1 <= action.all() <= 1

    if discrete_actions:
        for idx, env_action in enumerate(list(processed_action.values())):
            action_dim = (
                action_spaces[idx].shape[0]
                if isinstance(action_spaces[idx], spaces.Box)
                else action_spaces[idx].n
            )
            for action in env_action:
                assert action <= action_dim - 1


@pytest.mark.parametrize("observation_spaces", ["ma_vector_space"])
@pytest.mark.parametrize("action_spaces", ["ma_vector_space", "ma_discrete_space"])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_maddpg_get_action_agent_masking(
    training, observation_spaces, action_spaces, device, compile_mode, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {
        agent: np.random.randn(*observation_spaces[0].shape) for agent in agent_ids
    }
    discrete_actions = all(
        isinstance(space, spaces.Discrete) for space in action_spaces
    )
    if discrete_actions:
        info = {
            "agent_0": {"env_defined_actions": 1},
            "agent_1": {"env_defined_actions": 1},
            "other_agent_0": {"env_defined_actions": None},
        }
    else:
        info = {
            "agent_0": {"env_defined_actions": np.array([0, 1, 0, 1, 0, 1])},
            "agent_1": {"env_defined_actions": np.array([0, 1, 0, 1, 0, 1])},
            "other_agent_0": {"env_defined_actions": None},
        }
    maddpg = MADDPG(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
        torch_compiler=compile_mode,
    )
    maddpg.set_training_mode(training)
    action, _ = maddpg.get_action(state, infos=info)
    if discrete_actions:
        assert np.array_equal(action["agent_0"], np.array([1])), action["agent_0"]
    else:
        assert np.array_equal(
            action["agent_0"], np.array([[0, 1, 0, 1, 0, 1]])
        ), action["agent_0"]


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("observation_spaces", ["ma_vector_space"])
@pytest.mark.parametrize("action_spaces", ["ma_vector_space", "ma_discrete_space"])
def test_maddpg_get_action_vectorized_agent_masking(
    training, observation_spaces, action_spaces, device, request
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    num_envs = 6
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    state = {
        agent: np.array(
            [np.random.randn(*observation_spaces[0].shape) for _ in range(num_envs)]
        )
        for agent in agent_ids
    }
    discrete_actions = all(
        isinstance(space, spaces.Discrete) for space in action_spaces
    )
    if discrete_actions:
        env_defined_action = np.array(
            [
                np.random.randint(0, observation_spaces[0].shape[0] + 1)
                for _ in range(num_envs)
            ]
        )
    else:
        env_defined_action = np.array(
            [np.random.randn(*observation_spaces[0].shape) for _ in range(num_envs)]
        )
    nan_array = np.zeros(env_defined_action.shape)
    nan_array[:] = np.nan
    info = {
        "agent_0": {"env_defined_actions": env_defined_action},
        "agent_1": {"env_defined_actions": env_defined_action},
        "other_agent_0": {"env_defined_actions": nan_array},
    }
    maddpg = MADDPG(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
    )
    maddpg.set_training_mode(training)
    action, raw_action = maddpg.get_action(state, infos=info)
    if discrete_actions:
        assert np.array_equal(
            action["agent_0"].squeeze(), info["agent_0"]["env_defined_actions"]
        ), action["agent_0"]
    else:
        assert np.isclose(
            action["agent_0"], info["agent_0"]["env_defined_actions"]
        ).all(), action["agent_0"]


@pytest.mark.parametrize(
    "observation_spaces", ["ma_vector_space", "ma_discrete_space", "ma_image_space"]
)
@pytest.mark.parametrize("action_spaces", ["ma_vector_space", "ma_discrete_space"])
@pytest.mark.parametrize("agent_ids", [["agent_0", "agent_1", "other_agent_0"]])
@pytest.mark.parametrize("compile_mode", [None])
@pytest.mark.parametrize("accelerator", [None, Accelerator(device_placement=False)])
@pytest.mark.parametrize("batch_size", [64])
def test_maddpg_learns_from_experiences(
    observation_spaces,
    action_spaces,
    experiences,
    batch_size,
    agent_ids,
    device,
    compile_mode,
    accelerator,
    request,
):
    observation_spaces = request.getfixturevalue(observation_spaces)
    action_spaces = request.getfixturevalue(action_spaces)
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    maddpg = MADDPG(
        observation_spaces,
        action_spaces,
        agent_ids=agent_ids,
        device=device,
        accelerator=accelerator,
        torch_compiler=compile_mode,
    )
    if accelerator is not None:
        for agent_id, actor in maddpg.actors.items():
            critic = maddpg.critics[agent_id]
            actor_target = maddpg.actor_targets[agent_id]
            critic_target = maddpg.critic_targets[agent_id]
            actor.no_sync = no_sync.__get__(actor)
            critic.no_sync = no_sync.__get__(critic)
            actor_target.no_sync = no_sync.__get__(actor_target)
            critic_target.no_sync = no_sync.__get__(critic_target)

    actors_pre_learn_sd = {
        agent_id: copy.deepcopy(actor.state_dict())
        for agent_id, actor in maddpg.actors.items()
    }
    critics_pre_learn_sd = {
        agent_id: copy.deepcopy(critic.state_dict())
        for agent_id, critic in maddpg.critics.items()
    }

    for _ in range(2):
        maddpg.scores.append(0)
        loss = maddpg.learn(experiences)

    assert isinstance(loss, dict)

    for agent_id in maddpg.agent_ids:
        assert loss[agent_id][-1] >= 0.0

        updated_actor = maddpg.actors[agent_id]
        assert_not_equal_state_dict(
            actors_pre_learn_sd[agent_id], updated_actor.state_dict()
        )

        updated_critic = maddpg.critics[agent_id]
        assert_not_equal_state_dict(
            critics_pre_learn_sd[agent_id], updated_critic.state_dict()
        )


def no_sync(self):
    class DummyNoSync:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass  # Add cleanup or handling if needed

    return DummyNoSync()


@pytest.mark.parametrize("compile_mode", [None])
def test_maddpg_soft_update(device, compile_mode, ma_vector_space, ma_discrete_space):
    maddpg = MADDPG(
        ma_vector_space,
        ma_discrete_space,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        accelerator=None,
        device=device,
        torch_compiler=compile_mode,
    )

    for agent_id, actor in maddpg.actors.items():
        actor_target = maddpg.actor_targets[agent_id]
        # Check actors
        maddpg.soft_update(actor, actor_target)
        eval_params = list(actor.parameters())
        target_params = list(actor_target.parameters())
        expected_params = [
            maddpg.tau * eval_param + (1.0 - maddpg.tau) * target_param
            for eval_param, target_param in zip(eval_params, target_params)
        ]
        assert all(
            torch.allclose(expected_param, target_param)
            for expected_param, target_param in zip(expected_params, target_params)
        )

    for agent_id, critic in maddpg.critics.items():
        critic_target = maddpg.critic_targets[agent_id]
        maddpg.soft_update(critic, critic_target)
        eval_params = list(critic.parameters())
        target_params = list(critic_target.parameters())
        expected_params = [
            maddpg.tau * eval_param + (1.0 - maddpg.tau) * target_param
            for eval_param, target_param in zip(eval_params, target_params)
        ]

        assert all(
            torch.allclose(expected_param, target_param)
            for expected_param, target_param in zip(expected_params, target_params)
        )


@pytest.mark.parametrize("observation_spaces", ["ma_vector_space", "ma_image_space"])
@pytest.mark.parametrize("sum_score", [True, False])
@pytest.mark.parametrize("compile_mode", [None])
@pytest.mark.parametrize("vectorized", [True, False])
def test_maddpg_algorithm_test_loop(
    ma_discrete_space,
    device,
    sum_score,
    compile_mode,
    observation_spaces,
    vectorized,
    request,
):
    observation_spaces = request.getfixturevalue(observation_spaces)

    # Define environment and algorithm
    if vectorized:
        env = make_multi_agent_vect_envs(
            DummyMultiEnv,
            2,
            **dict(
                observation_spaces=observation_spaces[0],
                action_spaces=ma_discrete_space,
            )
        )
    else:
        env = DummyMultiEnv(observation_spaces[0], ma_discrete_space)

    maddpg = MADDPG(
        observation_spaces,
        ma_discrete_space,
        agent_ids=["agent_0", "agent_1", "other_agent_0"],
        device=device,
        torch_compiler=compile_mode,
    )
    mean_score = maddpg.test(env, max_steps=10, sum_scores=sum_score)
    if sum_score:
        assert isinstance(mean_score, float)
    else:
        assert isinstance(mean_score, np.ndarray)
        assert len(mean_score) == 3
    env.close()


@pytest.mark.parametrize("observation_spaces", ["ma_vector_space"])
@pytest.mark.parametrize("compile_mode", [None, "default"])
@pytest.mark.parametrize("accelerator_flag", [False, True])
@pytest.mark.parametrize("wrap", [True, False])
def test_maddpg_clone_returns_identical_agent(
    accelerator_flag, wrap, compile_mode, observation_spaces, ma_vector_space, request
):
    # Clones the agent and returns an identical copy.
    observation_spaces = request.getfixturevalue(observation_spaces)
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    expl_noise = 0.1
    index = 0
    batch_size = 64
    lr_actor = 0.001
    lr_critic = 0.01
    learn_step = 5
    gamma = 0.95
    tau = 0.01
    mut = None
    actor_networks = None
    critic_networks = None
    device = "cpu"
    accelerator = Accelerator(device_placement=False) if accelerator_flag else None

    maddpg = MADDPG(
        observation_spaces,
        ma_vector_space,
        agent_ids,
        expl_noise=expl_noise,
        index=index,
        batch_size=batch_size,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        learn_step=learn_step,
        gamma=gamma,
        tau=tau,
        mut=mut,
        actor_networks=actor_networks,
        critic_networks=critic_networks,
        device=device,
        accelerator=accelerator,
        wrap=wrap,
        torch_compiler=compile_mode,
    )
    clone_agent = maddpg.clone(wrap=wrap)

    assert isinstance(clone_agent, MADDPG)
    assert clone_agent.observation_spaces == maddpg.observation_spaces
    assert clone_agent.action_spaces == maddpg.action_spaces
    assert clone_agent.n_agents == maddpg.n_agents
    assert clone_agent.agent_ids == maddpg.agent_ids
    assert all(
        torch.equal(clone_expl_noise, expl_noise)
        for clone_expl_noise, expl_noise in zip(
            clone_agent.expl_noise.values(), maddpg.expl_noise.values()
        )
    )
    assert clone_agent.index == maddpg.index
    assert clone_agent.batch_size == maddpg.batch_size
    assert clone_agent.lr_actor == maddpg.lr_actor
    assert clone_agent.lr_critic == maddpg.lr_critic
    assert clone_agent.learn_step == maddpg.learn_step
    assert clone_agent.gamma == maddpg.gamma
    assert clone_agent.tau == maddpg.tau
    assert clone_agent.device == maddpg.device
    assert clone_agent.accelerator == maddpg.accelerator

    for agent_id in clone_agent.agent_ids:
        actor = maddpg.actors[agent_id]
        clone_actor = clone_agent.actors[agent_id]
        assert_state_dicts_equal(clone_actor.state_dict(), actor.state_dict())

        critic = maddpg.critics[agent_id]
        clone_critic = clone_agent.critics[agent_id]
        assert_state_dicts_equal(clone_critic.state_dict(), critic.state_dict())

        actor_target = maddpg.actor_targets[agent_id]
        clone_actor_target = clone_agent.actor_targets[agent_id]
        assert_state_dicts_equal(
            clone_actor_target.state_dict(), actor_target.state_dict()
        )

        critic_target = maddpg.critic_targets[agent_id]
        clone_critic_target = clone_agent.critic_targets[agent_id]
        assert_state_dicts_equal(
            clone_critic_target.state_dict(), critic_target.state_dict()
        )


@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_clone_new_index(compile_mode, ma_vector_space, ma_discrete_space):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]

    maddpg = MADDPG(
        ma_vector_space,
        ma_discrete_space,
        agent_ids,
        torch_compiler=compile_mode,
    )
    clone_agent = maddpg.clone(index=100)

    assert clone_agent.index == 100


@pytest.mark.parametrize("compile_mode", [None])
def test_clone_after_learning(compile_mode, ma_vector_space):
    agent_ids = ["agent_0", "agent_1", "other_agent_0"]
    batch_size = 8

    maddpg = MADDPG(
        ma_vector_space,
        copy.deepcopy(ma_vector_space),
        agent_ids,
        batch_size=batch_size,
        torch_compiler=compile_mode,
    )

    states = {
        agent_id: torch.randn(batch_size, ma_vector_space[idx].shape[0])
        for idx, agent_id in enumerate(agent_ids)
    }
    actions = {
        agent_id: torch.randn(batch_size, ma_vector_space[idx].shape[0])
        for idx, agent_id in enumerate(agent_ids)
    }
    rewards = {agent_id: torch.randn(batch_size, 1) for agent_id in agent_ids}
    next_states = {
        agent_id: torch.randn(batch_size, ma_vector_space[idx].shape[0])
        for idx, agent_id in enumerate(agent_ids)
    }
    dones = {agent_id: torch.zeros(batch_size, 1) for agent_id in agent_ids}

    experiences = states, actions, rewards, next_states, dones
    maddpg.learn(experiences)
    clone_agent = maddpg.clone()
    assert isinstance(clone_agent, MADDPG)
    assert clone_agent.observation_spaces == maddpg.observation_spaces
    assert clone_agent.action_spaces == maddpg.action_spaces
    assert clone_agent.n_agents == maddpg.n_agents
    assert clone_agent.agent_ids == maddpg.agent_ids
    assert all(
        torch.equal(clone_expl_noise, expl_noise)
        for clone_expl_noise, expl_noise in zip(
            clone_agent.expl_noise.values(), maddpg.expl_noise.values()
        )
    )
    assert clone_agent.index == maddpg.index
    assert clone_agent.batch_size == maddpg.batch_size
    assert clone_agent.lr_actor == maddpg.lr_actor
    assert clone_agent.lr_critic == maddpg.lr_critic
    assert clone_agent.learn_step == maddpg.learn_step
    assert clone_agent.gamma == maddpg.gamma
    assert clone_agent.tau == maddpg.tau
    assert clone_agent.device == maddpg.device
    assert clone_agent.accelerator == maddpg.accelerator

    for agent_id in clone_agent.agent_ids:
        actor = maddpg.actors[agent_id]
        clone_actor = clone_agent.actors[agent_id]
        assert_state_dicts_equal(clone_actor.state_dict(), actor.state_dict())

        critic = maddpg.critics[agent_id]
        clone_critic = clone_agent.critics[agent_id]
        assert_state_dicts_equal(clone_critic.state_dict(), critic.state_dict())

        clone_actor_target = clone_agent.actor_targets[agent_id]
        actor_target = maddpg.actor_targets[agent_id]
        assert_state_dicts_equal(
            clone_actor_target.state_dict(), actor_target.state_dict()
        )

        clone_critic_target = clone_agent.critic_targets[agent_id]
        critic_target = maddpg.critic_targets[agent_id]
        assert_state_dicts_equal(
            clone_critic_target.state_dict(), critic_target.state_dict()
        )

        clone_actor_opt = clone_agent.actor_optimizers.optimizer[agent_id]
        actor_opt = maddpg.actor_optimizers.optimizer[agent_id]
        assert str(clone_actor_opt) == str(actor_opt)

        clone_critic_opt = clone_agent.critic_optimizers.optimizer[agent_id]
        critic_opt = maddpg.critic_optimizers.optimizer[agent_id]
        assert str(clone_critic_opt) == str(critic_opt)
